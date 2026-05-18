"""
Central configuration for the LLM-Powered SQL Query Generator.
All hyper-parameters, paths, and runtime toggles live here so that
train.py / eval.py / app.py never contain magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ──────────────────────────────────────────────────────────────────────────────
# LoRA / PEFT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 4-bit Quantization (QLoRA)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    # Kaggle T4 does NOT support bfloat16 — use float16
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    # Allows CPU/disk offload when VRAM is insufficient (local dev / small GPUs).
    # Safe to leave enabled on Kaggle T4 — has no effect when everything fits in VRAM.
    llm_int8_enable_fp32_cpu_offload: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    lora_weights_dir: str = "./results/lora_weights"
    logging_dir: str = "./results/logs"

    num_train_epochs: int = 3
    max_seq_length: int = 512

    # T4 x2 memory budget:
    #   effective batch size = per_device_bs (1) × grad_accum (4) × n_gpus (2) = 8
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    gradient_checkpointing: bool = True   # trades compute for VRAM on T4
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # 8-bit paged optimizer — essential for QLoRA on 16 GB T4
    optim: str = "paged_adamw_8bit"

    # T4: fp16 yes / bf16 no
    fp16: bool = True
    bf16: bool = False

    logging_steps: int = 25
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    report_to: str = "none"          # set to "wandb" if needed
    dataloader_num_workers: int = 2  # Kaggle allows a few workers
    dataloader_pin_memory: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    spider_name: str = "xlangai/spider"
    wikisql_name: str = "wikisql"

    # Include WikiSQL validation split (~8 K) to push combined total toward 87 K
    include_wikisql_validation: bool = True

    # Path where Spider SQLite databases live (needed for Execution Accuracy).
    # Download from https://drive.google.com/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m
    # and unzip to this directory.
    spider_db_path: str = "./spider_databases"

    # Hard caps — set to None to use the full split
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 500   # keep eval fast on Kaggle


# ──────────────────────────────────────────────────────────────────────────────
# Inference / Generation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    max_new_tokens: int = 128
    temperature: float = 0.1        # near-greedy — SQL generation is deterministic
    do_sample: bool = False
    repetition_penalty: float = 1.1
    num_beams: int = 1              # greedy; increase for beam search at the cost of speed


# ──────────────────────────────────────────────────────────────────────────────
# Root Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    model_id: str = "codellama/CodeLlama-7b-hf"

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Prompt fed to the model during both training (+ SQL answer appended)
    # and inference (SQL answer is generated).
    prompt_template: str = (
        "### Task: Convert the natural language question to a SQL query.\n\n"
        "### Schema:\n{schema}\n\n"
        "### Question:\n{question}\n\n"
        "### SQL:\n"
    )

    # Sentinel that marks end-of-SQL during generation
    sql_end_token: str = ";"

    @property
    def device_map(self) -> str:
        # "auto" lets accelerate shard across both T4 GPUs automatically
        return "auto"

    @property
    def effective_batch_size(self) -> int:
        return (
            self.training.per_device_train_batch_size
            * self.training.gradient_accumulation_steps
        )


# Module-level singleton — import and use directly, or override per-file.
cfg = Config()
