"""
train.py — QLoRA Fine-Tuning Entry Point
=========================================
Fine-tunes CodeLlama-7B-hf on Spider + WikiSQL using 4-bit NF4 QLoRA.
Optimised for RTX 3050 (6 GB VRAM) and Kaggle T4 x2 (16 GB × 2).

VRAM budget (RTX 3050, 6 GB):
  Base model 4-bit NF4  : ~3.5 GB
  LoRA adapters         : ~40 MB
  Activations (seq=384) : ~1.5 GB  (gradient_checkpointing reduces ~50 %)
  Optimizer 8-bit paged : ~120 MB
  Total                 : ~5.5 GB  ← fits with gradient_checkpointing=True

Usage:
    python train.py                             # full training run
    python train.py --dry_run                   # 10 steps, verify pipeline
    python train.py --max_train_samples 5000    # quick smoke test
    python train.py --max_seq_length 384        # reduce if OOM on 6 GB GPU
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from config import Config
from data.dataset_loader import merge_datasets, SQLDataset
from utils.schema_linker import HeuristicSchemaLinker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# VRAM utility
# ──────────────────────────────────────────────────────────────────────────────

def _log_vram(label: str) -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        logger.info(
            "%s — VRAM: %.2f GB allocated / %.2f GB reserved", label, allocated, reserved
        )


def _configure_for_ampere() -> None:
    """Enable TF32 matmul on Ampere GPUs (RTX 30xx / A100) for ~17 %% speedup."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32  = True
            torch.backends.cudnn.allow_tf32         = True
            logger.info(
                "Ampere GPU detected (compute %d.%d) — TF32 enabled.", *cap
            )


# ──────────────────────────────────────────────────────────────────────────────
# Model / training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_bnb_config(cfg: Config) -> BitsAndBytesConfig:
    q = cfg.quantization
    return BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=q.bnb_4bit_compute_dtype,
        llm_int8_enable_fp32_cpu_offload=q.llm_int8_enable_fp32_cpu_offload,
    )


def _load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"  # SFTTrainer expects right-padding during training
    return tok


def _load_base_model(
    model_id: str,
    bnb_cfg: BitsAndBytesConfig,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )
    model.config.use_cache        = False  # required for gradient checkpointing
    model.config.pretraining_tp   = 1
    return model


def _apply_lora(model, cfg: Config):
    lc = cfg.lora
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    lora_config = LoraConfig(
        r=lc.r,
        lora_alpha=lc.lora_alpha,
        lora_dropout=lc.lora_dropout,
        bias=lc.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lc.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _build_training_args(
    cfg: Config,
    dry_run: bool = False,
    max_seq_length: Optional[int] = None,
) -> SFTConfig:
    tc         = cfg.training
    seq_len    = max_seq_length or tc.max_seq_length
    max_steps  = 10 if dry_run else -1
    save_steps = 5  if dry_run else tc.save_steps
    log_steps  = 2  if dry_run else tc.logging_steps

    if dry_run:
        logger.info("DRY RUN — max_steps=10, save_steps=5, logging_steps=2")
    if seq_len != tc.max_seq_length:
        logger.info("max_seq_length overridden: %d → %d", tc.max_seq_length, seq_len)

    os.makedirs(tc.output_dir, exist_ok=True)
    os.makedirs(tc.logging_dir, exist_ok=True)

    return SFTConfig(
        # SFT-specific
        max_length=seq_len,
        dataset_text_field="text",
        packing=False,
        completion_only_loss=True,
        # Run identity
        run_name=tc.run_name,
        output_dir=tc.output_dir,
        # Steps / epochs
        num_train_epochs=tc.num_train_epochs,
        max_steps=max_steps,
        # Batch / memory
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        gradient_checkpointing=tc.gradient_checkpointing,
        # Optimiser
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        optim=tc.optim,
        # Precision — fp16=True, bf16=False (T4 safe; RTX 3050 can use bf16 but fp16 is universal)
        fp16=tc.fp16,
        bf16=tc.bf16,
        fp16_opt_level="O1",
        # Logging
        logging_steps=log_steps,
        logging_dir=tc.logging_dir,
        report_to=tc.report_to,
        # Evaluation
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        # DataLoader
        dataloader_num_workers=tc.dataloader_num_workers,
        dataloader_pin_memory=tc.dataloader_pin_memory,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg = Config()

    if args.max_train_samples:
        cfg.data.max_train_samples = args.max_train_samples

    # Ampere TF32 acceleration (RTX 3050 is Ampere sm_86)
    _configure_for_ampere()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_examples, dev_examples, _ = merge_datasets(
        spider_name=cfg.data.spider_name,
        wikisql_name=cfg.data.wikisql_name,
        include_wikisql_validation=cfg.data.include_wikisql_validation,
        max_train_samples=cfg.data.max_train_samples,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", cfg.model_id)
    tokenizer     = _load_tokenizer(cfg.model_id)
    schema_linker = HeuristicSchemaLinker()

    seq_len = args.max_seq_length or cfg.training.max_seq_length

    train_dataset = SQLDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        schema_linker=schema_linker,
        prompt_template=cfg.prompt_template,
        max_length=seq_len,
        is_training=True,
    )

    eval_subset = dev_examples[: cfg.data.max_eval_samples]
    eval_dataset = SQLDataset(
        examples=eval_subset,
        tokenizer=tokenizer,
        schema_linker=schema_linker,
        prompt_template=cfg.prompt_template,
        max_length=seq_len,
        is_training=True,
    )

    logger.info(
        "Train: %d | Eval: %d | seq_len: %d",
        len(train_dataset), len(eval_dataset), seq_len,
    )

    hf_train = train_dataset.as_hf_dataset()
    hf_eval  = eval_dataset.as_hf_dataset()

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading base model: %s", cfg.model_id)
    _log_vram("Before model load")

    bnb_cfg = _build_bnb_config(cfg)
    try:
        model = _load_base_model(cfg.model_id, bnb_cfg)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "OOM loading model. Try: --max_seq_length 256 or reduce "
                "per_device_train_batch_size in config.py."
            )
        raise

    _log_vram("After model load")
    model = _apply_lora(model, cfg)
    _log_vram("After LoRA applied")

    torch.cuda.empty_cache()

    # ── Trainer ───────────────────────────────────────────────────────────────
    training_args = _build_training_args(cfg, dry_run=args.dry_run, max_seq_length=seq_len)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        processing_class=tokenizer,  # TRL 1.x replaces deprecated tokenizer= param
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    try:
        trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "OOM during training. Try: --max_seq_length 256 to reduce "
                "activation memory (6 GB GPU limit)."
            )
        raise

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg.training.lora_weights_dir, exist_ok=True)
    model.save_pretrained(cfg.training.lora_weights_dir)
    tokenizer.save_pretrained(cfg.training.lora_weights_dir)
    logger.info("LoRA adapters saved to %s", cfg.training.lora_weights_dir)

    _log_vram("After training complete")
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from typing import Optional

    parser = argparse.ArgumentParser(
        description="Fine-tune CodeLlama with QLoRA (RTX 3050 / Kaggle T4)"
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Cap training examples (useful for smoke tests)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Run 10 steps only to verify pipeline fires without crashing.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=None,
        help=(
            "Override max sequence length from config. "
            "Use 384 or 256 if you hit OOM on 6 GB VRAM. "
            "Default: uses config.training.max_seq_length (512)."
        ),
    )
    main(parser.parse_args())
