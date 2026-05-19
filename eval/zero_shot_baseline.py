"""
eval/zero_shot_baseline.py — Local Zero-Shot Baseline Evaluation
=================================================================
# PROPOSAL_REF: Enhancement 3 — Zero-Shot vs Fine-Tuned Comparative Evaluation

Loads CodeLlama-7B in 8-bit (GPU) or pure CPU mode, evaluates on a Spider dev
subset, and reports Exact Match per complexity level.

The script is intentionally lightweight — it evaluates up to 10 samples by
default so it finishes in a few minutes even on CPU, producing presentation-
ready output and a JSON artefact for slides.

Usage:
    python eval/zero_shot_baseline.py                        # 10 samples, auto device
    python eval/zero_shot_baseline.py --max_samples 5        # 5 samples only
    python eval/zero_shot_baseline.py --mock                 # instant mock mode (no model)
    python eval/zero_shot_baseline.py --output my_results.json
"""

from __future__ import annotations

# PROPOSAL_REF: Enhancement 3
import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows terminals default to cp1252 — force UTF-8 so emoji/box chars render
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.loader_local import load_spider
from utils.schema_linker import HeuristicSchemaLinker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra hard"]

# ──────────────────────────────────────────────────────────────────────────────
# Mock responses — used when --mock flag is set or model fails to load
# Provides instant, presentation-ready output without any GPU/download
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_RESPONSES = {
    "easy":       "SELECT COUNT(*) FROM singer",
    "medium":     "SELECT name, country FROM singer WHERE age > 25",
    "hard":       "SELECT T1.name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id GROUP BY T1.singer_id HAVING COUNT(*) > 1",
    "extra hard": "SELECT name FROM singer WHERE age = (SELECT MAX(age) FROM singer) INTERSECT SELECT name FROM singer WHERE country = 'France'",
}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(model_id: str):
    """
    Load CodeLlama in 8-bit (GPU) or fp32 (CPU) depending on what's available.
    Returns (model, tokenizer) or raises RuntimeError on failure.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        logger.info("CUDA available — loading in 8-bit quantization")
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
        )
    else:
        logger.info("No CUDA — loading in CPU fp32 mode (slow but functional)")
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tok


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = (
    "### Task: Convert the natural language question to a SQL query.\n\n"
    "### Schema:\n{schema}\n\n"
    "### Question:\n{question}\n\n"
    "### SQL:\n"
)


def _generate_sql(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    import torch
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if ";" in decoded:
        decoded = decoded[: decoded.index(";") + 1]
    return decoded


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").strip().lower())


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    examples:    List[Dict],
    model,
    tokenizer,
    linker:      HeuristicSchemaLinker,
    mock:        bool = False,
) -> List[Dict]:
    results = []

    for i, ex in enumerate(examples):
        filtered_schema = linker.filter_schema(ex["schema"], ex["question"])
        prompt          = _PROMPT_TEMPLATE.format(
            schema=filtered_schema, question=ex["question"]
        )

        if mock:
            diff        = ex.get("difficulty", "easy")
            pred_sql    = _MOCK_RESPONSES.get(diff, "SELECT * FROM table")
        else:
            pred_sql    = _generate_sql(model, tokenizer, prompt)

        gold_sql = ex.get("query", "")
        em       = int(_normalize_sql(pred_sql) == _normalize_sql(gold_sql))

        results.append({
            "db_id":           ex.get("db_id", ""),
            "difficulty":      ex.get("difficulty", "unknown"),
            "question":        ex["question"],
            "gold_sql":        gold_sql,
            "predicted_sql":   pred_sql,
            "filtered_schema": filtered_schema,
            "exact_match":     em,
            "mode":            "mock" if mock else "zero_shot",
        })

        status = "✅" if em else "❌"
        logger.info(
            "[%d/%d] %s %-12s | %s",
            i + 1, len(examples),
            status,
            ex.get("difficulty", "?"),
            ex["question"][:55],
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report printer — markdown table for presentation slides
# ──────────────────────────────────────────────────────────────────────────────

def print_report(results: List[Dict], mode: str) -> None:
    by_diff: Dict[str, List] = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)

    total_em = sum(r["exact_match"] for r in results)
    total_n  = len(results)

    print()
    print("=" * 60)
    print(f"  Zero-Shot Baseline — Exact Match Report  [{mode}]")
    print("  PROPOSAL_REF: Enhancement 3")
    print("=" * 60)
    print()
    print(f"  {'Difficulty':<14} {'Correct':>8} {'Total':>7} {'EM %':>8}")
    print("  " + "-" * 42)

    for diff in DIFFICULTY_ORDER:
        rows = by_diff.get(diff, [])
        if not rows:
            continue
        correct = sum(r["exact_match"] for r in rows)
        n       = len(rows)
        pct     = correct / n * 100 if n else 0.0
        bar     = "█" * int(pct / 10)
        print(f"  {diff:<14} {correct:>8} {n:>7} {pct:>7.1f}%  {bar}")

    for diff in sorted(by_diff):
        if diff not in DIFFICULTY_ORDER:
            rows    = by_diff[diff]
            correct = sum(r["exact_match"] for r in rows)
            n       = len(rows)
            pct     = correct / n * 100 if n else 0.0
            print(f"  {diff:<14} {correct:>8} {n:>7} {pct:>7.1f}%")

    print("  " + "-" * 42)
    overall_pct = total_em / total_n * 100 if total_n else 0.0
    print(f"  {'OVERALL':<14} {total_em:>8} {total_n:>7} {overall_pct:>7.1f}%")
    print()
    print("  (Zero-shot baseline — no fine-tuning applied)")
    print("=" * 60)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot baseline evaluation on Spider dev"
    )
    parser.add_argument("--model_id",    default="codellama/CodeLlama-7b-hf",
                        help="HuggingFace model ID")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Number of Spider dev examples to evaluate (default 10)")
    parser.add_argument("--mock",        action="store_true",
                        help="Use mock SQL responses — no model download needed")
    parser.add_argument("--output",      default="baseline_results.json",
                        help="Path to save JSON results")
    args = parser.parse_args()

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading Spider dev split …")
    examples = load_spider(split="dev", subset=args.max_samples)
    logger.info("Evaluating %d examples", len(examples))

    linker = HeuristicSchemaLinker()

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = None, None
    mode = "mock"

    if args.mock:
        logger.info("Mock mode — skipping model download")
    else:
        try:
            model, tokenizer = _load_model_and_tokenizer(args.model_id)
            mode = "zero_shot"
        except Exception as exc:
            logger.warning(
                "Model load failed (%s). Falling back to mock mode.", exc
            )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate(examples, model, tokenizer, linker, mock=(mode == "mock"))

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(results, mode)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"  💾 Results saved → {out_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
