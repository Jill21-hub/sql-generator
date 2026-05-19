"""
demo/presentation_demo.py — One-Click Presentation Demo
=========================================================
# PRESENTATION_READY
# PROPOSAL_REF: Enhancements 1, 2, 3, 4

Orchestrates the full Text-to-SQL pipeline locally:
  1. Loads 5 diverse Spider examples (one per complexity level)
  2. Applies HeuristicSchemaLinker → prints filtered schema
  3. Runs zero-shot inference (or uses fast mock responses)
  4. Compares predicted SQL vs gold SQL
  5. Exports demo_output.json for presentation slides
  6. Prints a slide-ready summary table

Usage:
    python demo/presentation_demo.py              # mock mode (instant, no model)
    python demo/presentation_demo.py --real       # real model inference (slow)
    python demo/presentation_demo.py --output slides/demo_output.json
"""

from __future__ import annotations

# PRESENTATION_READY
import argparse
import json
import logging
import re
import sys
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

_PROMPT_TEMPLATE = (
    "### Task: Convert the natural language question to a SQL query.\n\n"
    "### Schema:\n{schema}\n\n"
    "### Question:\n{question}\n\n"
    "### SQL:\n"
)

# ──────────────────────────────────────────────────────────────────────────────
# Mock SQL responses — realistic enough for slides, instant to produce
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_SQL: Dict[str, str] = {
    "easy":       "SELECT COUNT(*) FROM singer ;",
    "medium":     "SELECT name, country FROM singer WHERE age > 25 ;",
    "hard":       (
        "SELECT T1.name FROM singer AS T1 "
        "JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id "
        "GROUP BY T1.singer_id HAVING COUNT(*) > 1 ;"
    ),
    "extra hard": (
        "SELECT name FROM singer WHERE age = (SELECT MAX(age) FROM singer) "
        "INTERSECT SELECT name FROM singer WHERE country = 'France' ;"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Pick one example per difficulty level
# ──────────────────────────────────────────────────────────────────────────────

def _pick_one_per_difficulty(examples: List[Dict]) -> List[Dict]:
    seen: Dict[str, Dict] = {}
    for ex in examples:
        d = ex.get("difficulty", "easy")
        if d not in seen:
            seen[d] = ex
        if len(seen) == len(DIFFICULTY_ORDER):
            break

    # Return in canonical order; fill any missing difficulties from what we have
    result = []
    for d in DIFFICULTY_ORDER:
        if d in seen:
            result.append(seen[d])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").strip().lower())


def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    if torch.cuda.is_available():
        bnb   = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32,
            device_map="cpu", low_cpu_mem_usage=True,
        )
    model.eval()
    return model, tok


def _infer(model, tokenizer, prompt: str) -> str:
    import torch
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new = out[0, inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(new, skip_special_tokens=True).strip()
    if ";" in sql:
        sql = sql[: sql.index(";") + 1]
    return sql


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(examples: List[Dict], use_real_model: bool, model_id: str) -> List[Dict]:
    linker = HeuristicSchemaLinker()
    model, tokenizer = None, None

    if use_real_model:
        try:
            logger.info("Loading model %s …", model_id)
            model, tokenizer = _load_model(model_id)
        except Exception as exc:
            logger.warning("Model load failed (%s) — switching to mock mode.", exc)
            use_real_model = False

    sep = "─" * 66
    print()
    print("=" * 66)
    print("  Text-to-SQL Pipeline Demo")
    print("  PROPOSAL_REF: Enhancements 1 · 2 · 3 · 4")
    print(f"  Mode: {'Real model inference' if use_real_model else 'Mock responses (no model)'}")
    print("=" * 66)

    records = []
    for i, ex in enumerate(examples, 1):
        diff   = ex.get("difficulty", "easy")
        q      = ex["question"]
        gold   = ex.get("query", "")
        schema = ex["schema"]

        # Step 2 — Schema Linking
        filtered = linker.filter_schema(schema, q)
        prompt   = _PROMPT_TEMPLATE.format(schema=filtered, question=q)

        # Step 3 — Inference
        if use_real_model and model is not None:
            pred = _infer(model, tokenizer, prompt)
        else:
            pred = _MOCK_SQL.get(diff, "SELECT * FROM table ;")

        # Step 4 — Compare
        match = int(_normalize_sql(pred) == _normalize_sql(gold))

        # ── Slide-ready per-example printout ──────────────────────────────────
        print()
        print(sep)
        print(f"  Example {i}/5  [{diff.upper()}]")
        print(sep)
        print(f"  Question        : {q}")
        print()

        # Full schema (before linking)
        all_tables = schema.get("tables", [])
        full_repr  = "  |  ".join(
            f"{t['name']}({', '.join(c['name'] for c in t.get('columns',[])[:4])}{'…' if len(t.get('columns',[]))>4 else ''})"
            for t in all_tables
        )
        print(f"  Schema (full)   : {full_repr}")
        print(f"  Schema (linked) : {filtered}")
        print()
        print(f"  Gold SQL        : {gold}")
        print(f"  Predicted SQL   : {pred}")
        print()
        print(f"  Match           : {'✅ EXACT MATCH' if match else '❌ No match'}")

        records.append({
            "question":        q,
            "db_id":           ex.get("db_id", ""),
            "complexity":      diff,
            "filtered_schema": filtered,
            "predicted_sql":   pred,
            "gold_sql":        gold,
            "match":           bool(match),
            "mode":            "real" if use_real_model else "mock",
        })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(records: List[Dict]) -> None:
    total   = len(records)
    correct = sum(1 for r in records if r["match"])

    print()
    print("=" * 66)
    print("  📊 Demo Summary  (slide-ready)")
    print("=" * 66)
    print()
    print(f"  {'#':<4} {'Complexity':<14} {'Match':<8} {'Question'}")
    print("  " + "-" * 62)
    for i, r in enumerate(records, 1):
        icon = "✅" if r["match"] else "❌"
        q    = r["question"][:45] + ("…" if len(r["question"]) > 45 else "")
        print(f"  {i:<4} {r['complexity']:<14} {icon:<8} {q}")
    print("  " + "-" * 62)
    pct = correct / total * 100 if total else 0.0
    print(f"  {'TOTAL':<4} {'':<14} {correct}/{total} ({pct:.0f}% exact match)")
    print()
    print("  Enhancement Coverage:")
    print("    ✅ Enhancement 1 — Multi-dataset training (Spider + WikiSQL)")
    print("    ✅ Enhancement 2 — Schema linking (keyword heuristics)")
    print("    ✅ Enhancement 3 — Zero-shot baseline comparison")
    print("    ✅ Enhancement 4 — Error analysis by SQL complexity")
    print("=" * 66)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="One-click presentation demo")
    parser.add_argument("--real",      action="store_true",
                        help="Use real model inference (slow; requires HF download)")
    parser.add_argument("--model_id",  default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--output",    default="demo_output.json",
                        help="Path to save demo_output.json")
    args = parser.parse_args()

    # Step 1 — Load data
    logger.info("Loading Spider dev split …")
    all_dev  = load_spider(split="dev", subset=500)   # 500 enough to find all difficulties
    examples = _pick_one_per_difficulty(all_dev)
    logger.info("Selected %d examples (%s)", len(examples),
                ", ".join(e["difficulty"] for e in examples))

    # Steps 2–4 — Run pipeline
    records = run_demo(examples, use_real_model=args.real, model_id=args.model_id)

    # Summary
    print_summary(records)

    # Step 5 — Export
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  💾 Saved → {out.resolve()}")
    print()


if __name__ == "__main__":
    main()
