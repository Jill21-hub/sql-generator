"""
train.py — QLoRA Fine-Tuning Entry Point
=========================================
Loads CodeLlama-7B in 4-bit NF4, applies LoRA adapters, and fine-tunes on
the merged Spider + WikiSQL dataset using SFTTrainer.

Usage (Kaggle / terminal):
    python train.py
    python train.py --max_train_samples 5000   # quick smoke-test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from datasets import disable_progress_bar
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from config import Config
from data.dataset import merge_datasets, SQLDataset
from utils.schema_linker import HeuristicSchemaLinker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
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
    # CodeLlama has no default pad token — use eos as pad
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"   # SFTTrainer expects right-padding during training
    return tok


def _load_base_model(model_id: str, bnb_cfg: BitsAndBytesConfig) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",          # shard across T4 x2 automatically
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1
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


def _build_training_args(cfg: Config) -> SFTConfig:
    tc = cfg.training
    os.makedirs(tc.output_dir, exist_ok=True)
    os.makedirs(tc.logging_dir, exist_ok=True)
    return SFTConfig(
        # SFT-specific
        max_length=tc.max_seq_length,
        dataset_text_field="text",
        packing=False,
        completion_only_loss=True,   # replaces DataCollatorForCompletionOnlyLM
        # Standard training args
        output_dir=tc.output_dir,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        gradient_checkpointing=tc.gradient_checkpointing,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        optim=tc.optim,
        fp16=tc.fp16,
        bf16=tc.bf16,
        logging_steps=tc.logging_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        report_to=tc.report_to,
        logging_dir=tc.logging_dir,
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

    # ── Data ──────────────────────────────────────────────────────────────────
    train_examples, dev_examples, _ = merge_datasets(
        spider_name=cfg.data.spider_name,
        wikisql_name=cfg.data.wikisql_name,
        include_wikisql_validation=cfg.data.include_wikisql_validation,
        max_train_samples=cfg.data.max_train_samples,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", cfg.model_id)
    tokenizer = _load_tokenizer(cfg.model_id)

    schema_linker = HeuristicSchemaLinker()

    train_dataset = SQLDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        schema_linker=schema_linker,
        prompt_template=cfg.prompt_template,
        max_length=cfg.training.max_seq_length,
        is_training=True,
    )

    eval_subset = dev_examples[: cfg.data.max_eval_samples]
    eval_dataset = SQLDataset(
        examples=eval_subset,
        tokenizer=tokenizer,
        schema_linker=schema_linker,
        prompt_template=cfg.prompt_template,
        max_length=cfg.training.max_seq_length,
        is_training=True,   # eval loss uses same label-masked format
    )

    logger.info("Train: %d | Eval: %d", len(train_dataset), len(eval_dataset))

    # SFTTrainer expects a HuggingFace Dataset with a 'text' column
    hf_train = train_dataset.as_hf_dataset()
    hf_eval  = eval_dataset.as_hf_dataset()

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading base model: %s", cfg.model_id)
    bnb_cfg = _build_bnb_config(cfg)
    model = _load_base_model(cfg.model_id, bnb_cfg)
    model = _apply_lora(model, cfg)

    torch.cuda.empty_cache()

    # ── Trainer ───────────────────────────────────────────────────────────────
    training_args = _build_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        processing_class=tokenizer,   # TRL 1.x: replaces deprecated tokenizer= param
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg.training.lora_weights_dir, exist_ok=True)
    model.save_pretrained(cfg.training.lora_weights_dir)
    tokenizer.save_pretrained(cfg.training.lora_weights_dir)
    logger.info("LoRA adapters saved to %s", cfg.training.lora_weights_dir)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeLlama with QLoRA")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training examples (useful for smoke tests)",
    )
    main(parser.parse_args())
