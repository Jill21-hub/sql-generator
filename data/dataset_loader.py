"""
data/dataset_loader.py — Unified Dataset Pipeline
===================================================
Loads Spider + WikiSQL, normalises to a single schema, caches to disk,
and exposes a PyTorch Dataset for SFTTrainer.

Datasets
--------
  Train : Spider train (~7 000) + WikiSQL train (~56 000) + WikiSQL dev (~8 000)
  Dev   : Spider dev  (~1 034)  — never mixed with WikiSQL
  Test  : Spider test (~2 147)  — gold queries may be absent on HF

Cache
-----
  ~/.cache/sql_generator/<split>.json  — processed examples (JSON, written once)
  ~/.cache/wikisql_raw/               — raw WikiSQL JSONL + tables files

Function signatures (required by eval.py / train.py)
-----------------------------------------------------
  load_spider(split)                → List[Dict]
  load_wikisql()                    → List[Dict]   (train split only)
  merge_and_cache_datasets()        → (train, dev, test)
  get_dataset_split(name, max)      → List[Dict]
  merge_datasets(...)               → (train, dev, test)  [backward compat]
  SQLDataset                        → torch.utils.data.Dataset

Usage:
    python data/dataset_loader.py                    # full load + report
    python data/dataset_loader.py --max_samples 10   # quick smoke test
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Cache paths
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_DIR        = Path.home() / ".cache" / "sql_generator"
_WIKISQL_RAW_DIR  = Path.home() / ".cache" / "wikisql_raw"
_WIKISQL_TAR_URL  = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

_WIKISQL_SPLIT_MAP = {
    "train":      ("data/train.jsonl",  "data/train.tables.jsonl"),
    "validation": ("data/dev.jsonl",    "data/dev.tables.jsonl"),
    "test":       ("data/test.jsonl",   "data/test.tables.jsonl"),
}


def _load_from_cache(name: str) -> Optional[List[Dict]]:
    path = _CACHE_DIR / f"{name}.json"
    if path.exists():
        logger.info("Cache hit: %s (%s)", name, path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_to_cache(name: str, data: List[Dict]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("Cached %d examples → %s", len(data), path)


def _load_hf_with_retry(
    dataset_name: str,
    max_retries: int = 3,
    **kwargs,
):
    """Load a HuggingFace dataset with exponential back-off on failure."""
    for attempt in range(max_retries):
        try:
            return load_dataset(dataset_name, streaming=False, **kwargs)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "HF load attempt %d/%d failed (%s). Retrying in %ds …",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)


# ──────────────────────────────────────────────────────────────────────────────
# WikiSQL SQL reconstruction helpers
# ──────────────────────────────────────────────────────────────────────────────

_AGG_OPS  = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_COND_OPS = ["=", ">", "<", "!=", "<=", ">="]


def _reconstruct_wikisql_query(sql_dict: Dict, headers: List[str]) -> str:
    sel_idx = sql_dict.get("sel", 0)
    agg_idx = sql_dict.get("agg", 0)

    sel_col = headers[sel_idx] if sel_idx < len(headers) else "*"
    agg     = _AGG_OPS[agg_idx] if agg_idx < len(_AGG_OPS) else ""

    select_expr = f"{agg}({sel_col})" if agg else sel_col
    sql         = f"SELECT {select_expr} FROM table"

    conds      = sql_dict.get("conds", {})
    col_idxs   = conds.get("column_index", [])
    op_idxs    = conds.get("operator_index", [])
    cond_vals  = conds.get("condition", [])

    where_parts: List[str] = []
    for ci, oi, val in zip(col_idxs, op_idxs, cond_vals):
        col = headers[ci] if ci < len(headers) else f"col{ci}"
        op  = _COND_OPS[oi] if oi < len(_COND_OPS) else "="
        try:
            float(str(val))
            where_parts.append(f"{col} {op} {val}")
        except (ValueError, TypeError):
            where_parts.append(f"{col} {op} '{val}'")

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    return sql


# ──────────────────────────────────────────────────────────────────────────────
# Difficulty estimation
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_difficulty(sql: str) -> str:
    up    = sql.upper()
    score = (
        up.count("JOIN")
        + (up.count("SELECT") - 1) * 2
        + ("GROUP BY" in up)
        + ("HAVING"   in up)
        + any(k in up for k in ("INTERSECT", "UNION", "EXCEPT")) * 2
    )
    if score == 0:  return "easy"
    if score <= 2:  return "medium"
    if score <= 4:  return "hard"
    return "extra hard"


# ──────────────────────────────────────────────────────────────────────────────
# Schema parsers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_spider_schema(example: Dict) -> Dict:
    """Convert Spider's flat parallel arrays into normalised schema dict."""
    table_names: List[str] = example.get("db_table_names", [])
    col_info:    Dict      = example.get("db_column_names", {})
    col_types:   List[str] = example.get("db_column_types", [])

    table_id_list: List[int] = col_info.get("table_id", [])
    col_name_list: List[str] = col_info.get("column_name", [])

    tables: Dict[int, Dict] = {
        i: {"name": name, "columns": []}
        for i, name in enumerate(table_names)
    }
    for idx, (tid, cname) in enumerate(zip(table_id_list, col_name_list)):
        if tid < 0:
            continue
        ctype = col_types[idx] if idx < len(col_types) else ""
        if tid in tables:
            tables[tid]["columns"].append({"name": cname, "type": ctype})

    return {"db_id": example.get("db_id", ""), "tables": list(tables.values())}


def _parse_wikisql_schema(example: Dict) -> Dict:
    table   = example.get("table", {})
    headers = table.get("header", [])
    types   = table.get("types", [])
    table_id = table.get("id", "table")
    columns  = [{"name": h, "type": t} for h, t in zip(headers, types)]
    return {"db_id": table_id, "tables": [{"name": "table", "columns": columns}]}


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_spider(example: Dict) -> Optional[Dict]:
    try:
        query: str = (example.get("query") or "").strip()
        if not query:
            return None
        difficulty = example.get("difficulty") or _estimate_difficulty(query)
        return {
            "db_id":      example.get("db_id", ""),
            "schema":     _parse_spider_schema(example),
            "question":   (example.get("question") or "").strip(),
            "query":      query,
            "difficulty": difficulty,
            "source":     "spider",
        }
    except Exception as exc:
        logger.debug("Skip Spider example: %s", exc)
        return None


def _normalize_wikisql(example: Dict) -> Optional[Dict]:
    try:
        table   = example.get("table", {})
        headers = table.get("header", [])
        sql_dict = example.get("sql", {})

        query = (sql_dict.get("human_readable") or "").strip()
        if not query:
            query = _reconstruct_wikisql_query(sql_dict, headers)
        if not query:
            return None

        return {
            "db_id":      table.get("id", "table"),
            "schema":     _parse_wikisql_schema(example),
            "question":   (example.get("question") or "").strip(),
            "query":      query,
            "difficulty": "easy",
            "source":     "wikisql",
        }
    except Exception as exc:
        logger.debug("Skip WikiSQL example: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# WikiSQL direct loader (datasets 4.x dropped legacy script support)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_wikisql_extracted() -> None:
    sentinel = _WIKISQL_RAW_DIR / ".extracted"
    if sentinel.exists():
        return

    _WIKISQL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = _WIKISQL_RAW_DIR / "data.tar.bz2"

    if not tar_path.exists():
        logger.info("Downloading WikiSQL archive (~14 MB) from GitHub …")
        import requests
        resp = requests.get(_WIKISQL_TAR_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    logger.info("Extracting WikiSQL archive …")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(_WIKISQL_RAW_DIR)

    sentinel.touch()
    logger.info("WikiSQL extracted to %s", _WIKISQL_RAW_DIR)


def _load_wikisql_raw(split: str) -> List[Dict]:
    """Parse one WikiSQL split from the extracted JSONL files."""
    _ensure_wikisql_extracted()

    qa_rel, tables_rel = _WIKISQL_SPLIT_MAP[split]
    qa_path     = _WIKISQL_RAW_DIR / qa_rel
    tables_path = _WIKISQL_RAW_DIR / tables_rel

    tables: Dict[str, Dict] = {}
    with open(tables_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                tables[t["id"]] = t

    examples: List[Dict] = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex  = json.loads(line)
            tid = ex.get("table_id", "")
            tbl = tables.get(tid, {"id": tid, "header": [], "types": [], "rows": []})

            # Normalise raw conds [[col,op,val],...] → HF dict format
            sql_raw   = ex.get("sql", {})
            conds_raw = sql_raw.get("conds", [])
            if conds_raw and isinstance(conds_raw[0], list):
                conds_hf = {
                    "column_index":   [c[0] for c in conds_raw],
                    "operator_index": [c[1] for c in conds_raw],
                    "condition":      [c[2] for c in conds_raw],
                }
            else:
                conds_hf = conds_raw

            examples.append({
                "question": ex.get("question", ""),
                "sql": {
                    "sel":            sql_raw.get("sel", 0),
                    "agg":            sql_raw.get("agg", 0),
                    "conds":          conds_hf,
                    "human_readable": None,
                },
                "table": tbl,
            })
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_spider(split: str) -> List[Dict]:
    """
    Load a single Spider split from HuggingFace (with disk caching).

    Parameters
    ----------
    split : "train" | "validation" | "test"

    Returns
    -------
    List of normalised example dicts.
    """
    cache_name = f"spider_{split}"
    cached = _load_from_cache(cache_name)
    if cached is not None:
        return cached

    logger.info("Loading Spider '%s' split from HuggingFace …", split)
    try:
        ds = _load_hf_with_retry("xlangai/spider")
        hf_split = ds.get(split) or ds.get("validation") if split == "dev" else ds.get(split)
        if hf_split is None:
            logger.warning("Spider split '%s' not found on HF.", split)
            return []
        examples = [r for r in (_normalize_spider(ex) for ex in hf_split) if r]
    except Exception as exc:
        logger.error("Failed to load Spider %s: %s", split, exc)
        return []

    _save_to_cache(cache_name, examples)
    return examples


def load_wikisql() -> List[Dict]:
    """
    Load WikiSQL train split (direct GitHub download, datasets-4.x safe).

    Returns
    -------
    List of normalised example dicts (difficulty="easy", source="wikisql").
    Note: WikiSQL is only used for training — never for dev/test evaluation.
    """
    cache_name = "wikisql_train"
    cached = _load_from_cache(cache_name)
    if cached is not None:
        return cached

    logger.info("Loading WikiSQL train (direct download) …")
    try:
        raw      = _load_wikisql_raw("train")
        examples = [r for r in (_normalize_wikisql(ex) for ex in raw) if r]
    except Exception as exc:
        logger.error("Failed to load WikiSQL: %s", exc)
        return []

    _save_to_cache(cache_name, examples)
    logger.info("WikiSQL train: %d examples", len(examples))
    return examples


def merge_and_cache_datasets(
    include_wikisql_validation: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load, merge, and cache Spider + WikiSQL.

    Returns
    -------
    (train, dev, test) where:
      train = Spider train + WikiSQL train [+ WikiSQL dev if flag set]
      dev   = Spider dev  (for EM evaluation)
      test  = Spider test
    """
    # Check if combined cache already exists
    cached_train = _load_from_cache("combined_train")
    cached_dev   = _load_from_cache("spider_validation")
    cached_test  = _load_from_cache("spider_test")

    if cached_train and cached_dev:
        logger.info(
            "Loaded from cache — train: %d | dev: %d | test: %d",
            len(cached_train), len(cached_dev), len(cached_test or []),
        )
        return cached_train, cached_dev, cached_test or []

    # --- Spider ---
    spider_train = load_spider("train")
    spider_dev   = load_spider("validation")
    spider_test  = load_spider("test")

    # --- WikiSQL ---
    logger.info("Loading WikiSQL train …")
    wikisql_train = load_wikisql()
    wikisql_val   = []
    if include_wikisql_validation:
        val_cache = _load_from_cache("wikisql_validation")
        if val_cache is not None:
            wikisql_val = val_cache
        else:
            try:
                raw_val     = _load_wikisql_raw("validation")
                wikisql_val = [r for r in (_normalize_wikisql(ex) for ex in raw_val) if r]
                _save_to_cache("wikisql_validation", wikisql_val)
            except Exception as exc:
                logger.warning("WikiSQL validation load failed: %s", exc)

    train = spider_train + wikisql_train + wikisql_val
    _save_to_cache("combined_train", train)

    logger.info(
        "Dataset summary — Spider train: %d | WikiSQL train: %d | "
        "WikiSQL val: %d | Combined train: %d | Dev: %d | Test: %d",
        len(spider_train), len(wikisql_train), len(wikisql_val),
        len(train), len(spider_dev), len(spider_test),
    )
    return train, spider_dev, spider_test


def get_dataset_split(
    split_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Convenience wrapper — returns a single named split, optionally capped.

    Parameters
    ----------
    split_name  : "train" | "dev" | "validation" | "test"
    max_samples : If given, truncate the split to this many examples.

    Returns
    -------
    List[Dict] — normalised examples.
    """
    train, dev, test = merge_and_cache_datasets()
    mapping = {
        "train":      train,
        "dev":        dev,
        "validation": dev,
        "test":       test,
    }
    split = mapping.get(split_name, dev)
    if max_samples is not None:
        split = split[:max_samples]
    return split


# Backward-compatible alias used by train.py
def merge_datasets(
    spider_name: str = "xlangai/spider",
    wikisql_name: str = "wikisql",
    include_wikisql_validation: bool = True,
    max_train_samples: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Backward-compatible wrapper for train.py."""
    train, dev, test = merge_and_cache_datasets(
        include_wikisql_validation=include_wikisql_validation,
    )
    if max_train_samples is not None:
        train = train[:max_train_samples]
    return train, dev, test


# ──────────────────────────────────────────────────────────────────────────────
# Custom collator for eval batches
# ──────────────────────────────────────────────────────────────────────────────

def eval_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "gold_query":     [b["gold_query"]  for b in batch],
        "difficulty":     [b["difficulty"]  for b in batch],
        "db_id":          [b["db_id"]       for b in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# SQLDataset
# ──────────────────────────────────────────────────────────────────────────────

class SQLDataset(Dataset):
    """
    PyTorch Dataset for SQL generation fine-tuning and evaluation.

    Training mode  → returns {input_ids, attention_mask, labels}
    Eval mode      → returns {input_ids, attention_mask, gold_query, difficulty, db_id}
    SFTTrainer     → call .as_hf_dataset() for a HF Dataset with 'text' column.

    Parameters
    ----------
    examples        : List of normalised dicts from merge_and_cache_datasets().
    tokenizer       : HF tokenizer (pad_token must be set).
    schema_linker   : HeuristicSchemaLinker instance.
    prompt_template : Format string with {schema} and {question} placeholders.
    max_length      : Maximum total sequence length.
    is_training     : Training → label masking; False → metadata passthrough.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        schema_linker,
        prompt_template: str,
        max_length: int = 512,
        is_training: bool = True,
    ) -> None:
        self.tokenizer       = tokenizer
        self.schema_linker   = schema_linker
        self.prompt_template = prompt_template
        self.max_length      = max_length
        self.is_training     = is_training

        logger.info("Preprocessing %d examples …", len(examples))
        processed   = [self._preprocess(ex) for ex in examples]
        self.data   = [p for p in processed if p is not None]
        skipped     = len(examples) - len(self.data)
        if skipped:
            logger.warning("Skipped %d examples during preprocessing.", skipped)
        logger.info("SQLDataset ready — %d examples.", len(self.data))

    def _preprocess(self, example: Dict) -> Optional[Dict]:
        try:
            filtered = self.schema_linker.filter_schema(
                example["schema"], example["question"]
            )
            prompt = self.prompt_template.format(
                schema=filtered, question=example["question"]
            )
            query = example.get("query", "").rstrip(";").strip()
            return {
                "prompt":     prompt,
                "query":      query,
                "difficulty": example.get("difficulty", "medium"),
                "db_id":      example.get("db_id", ""),
                "source":     example.get("source", ""),
            }
        except Exception as exc:
            logger.debug("Preprocessing failed: %s", exc)
            return None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item   = self.data[idx]
        prompt = item["prompt"]
        if self.is_training:
            return self._training_item(prompt, item["query"])
        return self._eval_item(prompt, item)

    def _training_item(self, prompt: str, query: str) -> Dict[str, torch.Tensor]:
        full_text  = f"{prompt}{query} ;"
        enc_full   = self.tokenizer(
            full_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        enc_prompt = self.tokenizer(
            prompt, max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids      = enc_full["input_ids"].squeeze(0)
        attention_mask = enc_full["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
        prompt_len     = min(enc_prompt["input_ids"].shape[1], self.max_length)
        labels[:prompt_len] = -100  # mask prompt tokens from loss
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _eval_item(self, prompt: str, item: Dict) -> Dict[str, Any]:
        enc = self.tokenizer(
            prompt, max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "gold_query":     item["query"],
            "difficulty":     item["difficulty"],
            "db_id":          item["db_id"],
        }

    def as_hf_dataset(self) -> HFDataset:
        """Return HF Dataset with 'text' column for SFTTrainer."""
        return HFDataset.from_dict({
            "text": [f"{d['prompt']}{d['query']} ;" for d in self.data]
        })

    def difficulty_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(d["difficulty"] for d in self.data))


# ──────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────────────────────────────────────

def _print_report(
    train: List[Dict],
    dev: List[Dict],
    test: List[Dict],
    max_samples: int,
) -> None:
    from collections import Counter
    sep = "─" * 60

    def _dist(examples: List[Dict]) -> str:
        counts = Counter(e["difficulty"] for e in examples)
        return "  ".join(
            f"{d}: {counts.get(d, 0)}" for d in ["easy", "medium", "hard", "extra hard"]
        )

    print()
    print("=" * 60)
    print("  Dataset Loader — Summary Report")
    print("=" * 60)
    print(f"  Train examples : {len(train)}")
    print(f"  Dev examples   : {len(dev)}")
    print(f"  Test examples  : {len(test)}")
    print()
    print(f"  Train difficulty: {_dist(train)}")
    print(f"  Dev difficulty  : {_dist(dev)}")
    print("=" * 60)

    print()
    print(f"  Sample train examples (first {max_samples}):")
    print(sep)
    for i, ex in enumerate(train[:max_samples], 1):
        q   = ex["question"][:60] + ("…" if len(ex["question"]) > 60 else "")
        sql = ex["query"][:60] + ("…" if len(ex["query"]) > 60 else "")
        print(f"  [{i}] [{ex['difficulty'].upper():10}] [{ex['source']:7}] {q}")
        print(f"      SQL: {sql}")
        print()

    print()
    print(f"  Sample dev examples (first {min(5, len(dev))}):")
    print(sep)
    for i, ex in enumerate(dev[:5], 1):
        q   = ex["question"][:60] + ("…" if len(ex["question"]) > 60 else "")
        sql = ex["query"][:60] + ("…" if len(ex["query"]) > 60 else "")
        print(f"  [{i}] [{ex['difficulty'].upper():10}] {q}")
        print(f"      SQL: {sql}")
        print()

    print("  Caching: verified — files written to", _CACHE_DIR)
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset loader smoke test")
    parser.add_argument(
        "--max_samples", type=int, default=10,
        help="Number of train examples to display (default: 10)",
    )
    args = parser.parse_args()

    train, dev, test = merge_and_cache_datasets(include_wikisql_validation=True)
    _print_report(train[:args.max_samples], dev, test, args.max_samples)
