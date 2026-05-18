"""
eval.py — Evaluation & Error Analysis
=======================================
Runs Exact Match (EM) and Execution Accuracy (EX) on Spider Dev/Test splits,
grouped by difficulty level.

Usage:
    # Fine-tuned model
    python eval.py

    # Zero-shot baseline (base model, no LoRA)
    python eval.py --zero_shot

    # Evaluate on test split
    python eval.py --split test

    # Limit examples for quick check
    python eval.py --max_samples 100
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import Config
from data.dataset import merge_datasets
from utils.schema_linker import HeuristicSchemaLinker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra hard"]


# ──────────────────────────────────────────────────────────────────────────────
# SQL normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_sql(sql: str) -> str:
    """Lower-case, collapse whitespace, strip trailing semicolons."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql.lower())
    return sql


# ──────────────────────────────────────────────────────────────────────────────
# Execution accuracy
# ──────────────────────────────────────────────────────────────────────────────

def _execute_sql(db_path: str, sql: str) -> Optional[List]:
    """Run sql against a SQLite database and return result rows, or None on error."""
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [tuple(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception:
        return None


def _execution_match(
    db_id: str,
    pred_sql: str,
    gold_sql: str,
    spider_db_path: str,
) -> bool:
    """Return True if predicted and gold SQL return identical result sets."""
    db_file = os.path.join(spider_db_path, db_id, f"{db_id}.sqlite")
    pred_rows = _execute_sql(db_file, pred_sql)
    gold_rows = _execute_sql(db_file, gold_sql)

    if pred_rows is None or gold_rows is None:
        return False
    return set(map(tuple, pred_rows)) == set(map(tuple, gold_rows))


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"   # left-pad for generation
    return tok


def _load_model(cfg: Config, zero_shot: bool):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=cfg.quantization.bnb_4bit_compute_dtype,
        llm_int8_enable_fp32_cpu_offload=cfg.quantization.llm_int8_enable_fp32_cpu_offload,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = True

    if not zero_shot:
        if not os.path.isdir(cfg.training.lora_weights_dir):
            raise FileNotFoundError(
                f"LoRA weights not found at '{cfg.training.lora_weights_dir}'. "
                "Run train.py first, or pass --zero_shot for baseline evaluation."
            )
        logger.info("Loading LoRA adapters from %s", cfg.training.lora_weights_dir)
        model = PeftModel.from_pretrained(model, cfg.training.lora_weights_dir)
        model = model.merge_and_unload()

    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# SQL generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_sql(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    cfg: Config,
) -> str:
    ic = cfg.inference
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.training.max_seq_length,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=ic.max_new_tokens,
            do_sample=ic.do_sample,
            temperature=ic.temperature,
            repetition_penalty=ic.repetition_penalty,
            num_beams=ic.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Trim at the first semicolon (end-of-SQL sentinel)
    if ";" in decoded:
        decoded = decoded[: decoded.index(";") + 1]

    return decoded.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    tokenizer: AutoTokenizer,
    examples: List[Dict],
    split_name: str,
    cfg: Config,
    schema_linker: HeuristicSchemaLinker,
) -> Dict:
    """
    Iterate over examples, generate SQL, compute EM and EX per example,
    then aggregate metrics by difficulty.

    Returns
    -------
    dict with keys:
        "results"       : list of per-example result dicts
        "overall"       : {"em": float, "ex": float, "n": int}
        "by_difficulty" : {difficulty: {"em": float, "ex": float, "n": int}}
    """
    results: List[Dict] = []

    for i, ex in enumerate(examples):
        filtered_schema = schema_linker.filter_schema(ex["schema"], ex["question"])
        prompt = cfg.prompt_template.format(
            schema=filtered_schema,
            question=ex["question"],
        )

        pred_sql = _generate_sql(model, tokenizer, prompt, cfg)
        gold_sql = ex.get("query", "")
        difficulty = ex.get("difficulty", "medium")
        db_id = ex.get("db_id", "")

        em = int(_normalize_sql(pred_sql) == _normalize_sql(gold_sql))
        ex_match = int(_execution_match(
            db_id, pred_sql, gold_sql, cfg.data.spider_db_path
        ))

        results.append({
            "db_id":      db_id,
            "difficulty": difficulty,
            "question":   ex["question"],
            "gold_sql":   gold_sql,
            "pred_sql":   pred_sql,
            "em":         em,
            "ex":         ex_match,
        })

        if (i + 1) % 50 == 0:
            running_em = sum(r["em"] for r in results) / len(results)
            running_ex = sum(r["ex"] for r in results) / len(results)
            logger.info(
                "[%s] %d/%d — EM: %.3f | EX: %.3f",
                split_name, i + 1, len(examples), running_em, running_ex,
            )

        torch.cuda.empty_cache()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _agg(subset: List[Dict]) -> Dict:
        n = len(subset)
        if n == 0:
            return {"em": 0.0, "ex": 0.0, "n": 0}
        return {
            "em": round(sum(r["em"] for r in subset) / n, 4),
            "ex": round(sum(r["ex"] for r in subset) / n, 4),
            "n":  n,
        }

    by_difficulty: Dict[str, Dict] = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r)

    return {
        "results":       results,
        "overall":       _agg(results),
        "by_difficulty": {k: _agg(v) for k, v in by_difficulty.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report printing
# ──────────────────────────────────────────────────────────────────────────────

def _print_report(metrics: Dict, split_name: str, mode: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Evaluation Report — split={split_name}  mode={mode}")
    print(sep)

    ov = metrics["overall"]
    print(f"\n  Overall  (n={ov['n']:>5})   EM: {ov['em']:.4f}   EX: {ov['ex']:.4f}")

    print("\n  By Difficulty:")
    print(f"  {'Difficulty':<14} {'N':>6}   {'EM':>8}   {'EX':>8}")
    print("  " + "-" * 42)

    bd = metrics["by_difficulty"]
    for diff in DIFFICULTY_ORDER:
        if diff in bd:
            d = bd[diff]
            print(f"  {diff:<14} {d['n']:>6}   {d['em']:>8.4f}   {d['ex']:>8.4f}")

    # Any unexpected difficulty labels
    for diff, d in sorted(bd.items()):
        if diff not in DIFFICULTY_ORDER:
            print(f"  {diff:<14} {d['n']:>6}   {d['em']:>8.4f}   {d['ex']:>8.4f}")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg = Config()
    mode = "zero_shot" if args.zero_shot else "fine_tuned"

    logger.info("Loading tokenizer: %s", cfg.model_id)
    tokenizer = _load_tokenizer(cfg.model_id)

    logger.info("Loading model (zero_shot=%s) …", args.zero_shot)
    model = _load_model(cfg, zero_shot=args.zero_shot)
    torch.cuda.empty_cache()

    schema_linker = HeuristicSchemaLinker()

    _, dev_examples, test_examples = merge_datasets(
        spider_name=cfg.data.spider_name,
        wikisql_name=cfg.data.wikisql_name,
        include_wikisql_validation=False,
    )

    splits_to_eval: List[Tuple[str, List[Dict]]] = []
    if args.split in ("dev", "both"):
        subset = dev_examples[: args.max_samples] if args.max_samples else dev_examples
        splits_to_eval.append(("dev", subset))
    if args.split in ("test", "both"):
        subset = test_examples[: args.max_samples] if args.max_samples else test_examples
        splits_to_eval.append(("test", subset))

    for split_name, examples in splits_to_eval:
        if not examples:
            logger.warning("Split '%s' is empty — skipping.", split_name)
            continue
        logger.info("Evaluating on '%s' split (%d examples) …", split_name, len(examples))
        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            split_name=split_name,
            cfg=cfg,
            schema_linker=schema_linker,
        )
        _print_report(metrics, split_name, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SQL generation model")
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Evaluate base model without LoRA adapters (baseline comparison)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="dev",
        help="Which Spider split to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap examples per split (useful for quick runs)",
    )
    main(parser.parse_args())
