"""
test_dataset.py — Dataset Pipeline Smoke Test
==============================================
Loads 10 train + 5 dev examples, prints schema structure, question,
gold SQL, and confirms caching works. Runs in <30 seconds locally.

Usage:
    python test_dataset.py
    python test_dataset.py --max_samples 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.dataset_loader import merge_and_cache_datasets, _CACHE_DIR


def _print_example(i: int, ex: dict, label: str) -> None:
    sep = "─" * 64
    print(f"\n  [{label} {i}]  db: {ex['db_id']}  difficulty: {ex['difficulty']}")
    print(sep)
    print(f"  Q  : {ex['question']}")
    print(f"  SQL: {ex['query']}")
    schema = ex.get("schema", {})
    tables = schema.get("tables", [])
    schema_str = " | ".join(
        f"{t['name']}({', '.join(c['name'] for c in t.get('columns', [])[:4])}{'…' if len(t.get('columns', [])) > 4 else ''})"
        for t in tables[:3]
    )
    print(f"  SCH: {schema_str or '(empty)'}")


def main(max_samples: int = 10) -> None:
    sep = "=" * 64
    print()
    print(sep)
    print("  Dataset Loader — Smoke Test")
    print(sep)

    # ── First load (may hit HF) ───────────────────────────────────────────────
    t0 = time.time()
    train, dev, test = merge_and_cache_datasets(include_wikisql_validation=True)
    elapsed = time.time() - t0
    print(f"\n  Load time : {elapsed:.1f}s")
    print(f"  Train     : {len(train):,} examples")
    print(f"  Dev       : {len(dev):,} examples")
    print(f"  Test      : {len(test):,} examples")

    # ── Source breakdown ──────────────────────────────────────────────────────
    from collections import Counter
    sources = Counter(e["source"] for e in train)
    print(f"\n  Train sources: {dict(sources)}")

    # ── Print examples ────────────────────────────────────────────────────────
    print(f"\n  ── Train examples (first {max_samples}) ──────────────────")
    for i, ex in enumerate(train[:max_samples], 1):
        _print_example(i, ex, "TRAIN")

    print(f"\n  ── Dev examples (first 5) ───────────────────────────────")
    for i, ex in enumerate(dev[:5], 1):
        _print_example(i, ex, "DEV")

    # ── Cache verification ────────────────────────────────────────────────────
    cache_files = list(_CACHE_DIR.glob("*.json"))
    print(f"\n  ── Cache verification ───────────────────────────────────")
    print(f"  Cache dir : {_CACHE_DIR}")
    for cf in sorted(cache_files):
        size_kb = cf.stat().st_size / 1024
        print(f"  {cf.name:<35}  {size_kb:>8.1f} KB")

    # ── Second load (must hit cache) ──────────────────────────────────────────
    t1 = time.time()
    train2, dev2, _ = merge_and_cache_datasets()
    cache_elapsed   = time.time() - t1
    print(f"\n  Cache reload time: {cache_elapsed:.2f}s  (should be <1s)")
    assert len(train2) == len(train), "Cache mismatch in train"
    assert len(dev2)   == len(dev),   "Cache mismatch in dev"
    print("  Cache integrity: ✅ OK")

    print()
    print(sep)
    print("  ✅ All checks passed")
    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset pipeline smoke test")
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()
    main(args.max_samples)
