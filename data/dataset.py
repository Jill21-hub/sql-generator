"""
Dataset Loading and Preprocessing
===================================
Handles two sources:
  - xlangai/spider  (train + validation + test)
  - wikisql         (train, optionally validation)

Both are normalised to a single internal format:

    {
        "db_id":      str,
        "schema":     {"db_id": str, "tables": [{"name": str, "columns": [...]}]},
        "question":   str,
        "query":      str,        # gold SQL
        "difficulty": str,        # easy / medium / hard / extra hard
        "source":     str,        # "spider" | "wikisql"
    }

SQLDataset wraps the normalised list and:
  1. Applies HeuristicSchemaLinker to compress the schema.
  2. Formats the prompt template.
  3. Tokenises with label masking (loss on SQL tokens only).
  4. Exposes as_hf_dataset() for SFTTrainer compatibility.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# WikiSQL SQL reconstruction helpers
# ──────────────────────────────────────────────────────────────────────────────

_AGG_OPS = ["", "max", "min", "count", "sum", "avg"]
_COND_OPS = ["=", ">", "<", "!=", "<=", ">="]


def _reconstruct_wikisql_query(sql_dict: Dict, headers: List[str]) -> str:
    """Build an SQL string from WikiSQL's structured {sel, agg, conds} dict."""
    sel_idx: int = sql_dict.get("sel", 0)
    agg_idx: int = sql_dict.get("agg", 0)

    sel_col = headers[sel_idx] if sel_idx < len(headers) else "*"
    agg = _AGG_OPS[agg_idx] if agg_idx < len(_AGG_OPS) else ""

    select_expr = f"{agg}({sel_col})" if agg else sel_col
    sql = f"SELECT {select_expr} FROM table"

    conds: Dict = sql_dict.get("conds", {})
    col_indices: List[int] = conds.get("column_index", [])
    op_indices: List[int] = conds.get("operator_index", [])
    cond_values: List[Any] = conds.get("condition", [])

    where_parts: List[str] = []
    for col_i, op_i, val in zip(col_indices, op_indices, cond_values):
        col = headers[col_i] if col_i < len(headers) else f"col{col_i}"
        op = _COND_OPS[op_i] if op_i < len(_COND_OPS) else "="
        # Numeric literals are unquoted; everything else is quoted
        try:
            float(str(val))
            where_parts.append(f"{col} {op} {val}")
        except (ValueError, TypeError):
            where_parts.append(f"{col} {op} '{val}'")

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    return sql


# ──────────────────────────────────────────────────────────────────────────────
# Difficulty estimation (fallback when the field is absent)
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_difficulty(sql: str) -> str:
    """
    Approximate Spider difficulty from SQL structure.
    Mirrors the original Spider annotation heuristics:
      easy   → simple SELECT with optional WHERE
      medium → JOIN or GROUP BY
      hard   → nested subquery or HAVING
      extra hard → set operations (INTERSECT / UNION / EXCEPT) or deeply nested
    """
    up = sql.upper()
    n_selects = up.count("SELECT")
    score = 0
    score += up.count("JOIN")
    score += (n_selects - 1) * 2          # nested subqueries are expensive
    score += ("GROUP BY" in up)
    score += ("HAVING" in up)
    score += any(k in up for k in ("INTERSECT", "UNION", "EXCEPT")) * 2

    if score == 0:
        return "easy"
    elif score <= 2:
        return "medium"
    elif score <= 4:
        return "hard"
    return "extra hard"


# ──────────────────────────────────────────────────────────────────────────────
# Schema parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_spider_schema(example: Dict) -> Dict:
    """
    Convert Spider's flat parallel arrays into the normalised schema dict.

    Spider stores schema as:
        db_table_names: ["stadium", "singer", ...]
        db_column_names: {
            "table_id":    [-1, 0, 0, 1, 1, ...],
            "column_name": ["*", "stadium_id", "location", ...]
        }
        db_column_types: ["text", "number", "text", ...]
    """
    table_names: List[str] = example.get("db_table_names", [])
    col_info: Dict = example.get("db_column_names", {})
    col_types: List[str] = example.get("db_column_types", [])

    table_id_list: List[int] = col_info.get("table_id", [])
    col_name_list: List[str] = col_info.get("column_name", [])

    tables: Dict[int, Dict] = {
        i: {"name": name, "columns": []}
        for i, name in enumerate(table_names)
    }

    for idx, (tid, cname) in enumerate(zip(table_id_list, col_name_list)):
        if tid < 0:           # skip the wildcard (*) pseudo-column
            continue
        ctype = col_types[idx] if idx < len(col_types) else ""
        if tid in tables:
            tables[tid]["columns"].append({"name": cname, "type": ctype})

    return {
        "db_id": example.get("db_id", ""),
        "tables": list(tables.values()),
    }


def _parse_wikisql_schema(example: Dict) -> Dict:
    """Convert WikiSQL's table dict into the normalised schema dict."""
    table: Dict = example.get("table", {})
    headers: List[str] = table.get("header", [])
    types: List[str] = table.get("types", [])
    table_id: str = table.get("id", "table")

    columns = [
        {"name": h, "type": t}
        for h, t in zip(headers, types)
    ]
    return {
        "db_id": table_id,
        "tables": [{"name": "table", "columns": columns}],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation functions
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_spider(example: Dict) -> Optional[Dict]:
    try:
        query: str = (example.get("query") or "").strip()
        if not query:
            return None

        # HuggingFace xlangai/spider may or may not include "difficulty"
        difficulty: str = (
            example.get("difficulty")
            or _estimate_difficulty(query)
        )

        return {
            "db_id":      example.get("db_id", ""),
            "schema":     _parse_spider_schema(example),
            "question":   (example.get("question") or "").strip(),
            "query":      query,
            "difficulty": difficulty,
            "source":     "spider",
        }
    except Exception as exc:
        logger.debug("Skipping Spider example: %s", exc)
        return None


def _normalize_wikisql(example: Dict) -> Optional[Dict]:
    try:
        table: Dict = example.get("table", {})
        headers: List[str] = table.get("header", [])
        sql_dict: Dict = example.get("sql", {})

        # Prefer the pre-built human-readable string
        query: str = (sql_dict.get("human_readable") or "").strip()
        if not query:
            query = _reconstruct_wikisql_query(sql_dict, headers)
        if not query:
            return None

        return {
            "db_id":      table.get("id", "table"),
            "schema":     _parse_wikisql_schema(example),
            "question":   (example.get("question") or "").strip(),
            "query":      query,
            "difficulty": "easy",   # WikiSQL is single-table, single-clause
            "source":     "wikisql",
        }
    except Exception as exc:
        logger.debug("Skipping WikiSQL example: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# WikiSQL direct loader (datasets 4.x dropped legacy script support)
# ──────────────────────────────────────────────────────────────────────────────

_WIKISQL_TAR_URL   = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
_WIKISQL_CACHE_DIR = Path(os.path.expanduser("~/.cache/wikisql_raw"))
_WIKISQL_SPLIT_MAP = {
    "train":      ("data/train.jsonl",  "data/train.tables.jsonl"),
    "validation": ("data/dev.jsonl",    "data/dev.tables.jsonl"),
    "test":       ("data/test.jsonl",   "data/test.tables.jsonl"),
}


def _ensure_wikisql_extracted() -> None:
    """Download and extract data.tar.bz2 once, then cache the JSONL files."""
    sentinel = _WIKISQL_CACHE_DIR / ".extracted"
    if sentinel.exists():
        return

    _WIKISQL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = _WIKISQL_CACHE_DIR / "data.tar.bz2"

    if not tar_path.exists():
        logger.info("Downloading WikiSQL archive (~14 MB) from GitHub …")
        import requests  # available via huggingface-hub dependency
        resp = requests.get(_WIKISQL_TAR_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    logger.info("Extracting WikiSQL archive …")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(_WIKISQL_CACHE_DIR)

    sentinel.touch()
    logger.info("WikiSQL extracted to %s", _WIKISQL_CACHE_DIR)


def _load_wikisql_split(split: str) -> List[Dict]:
    """
    Download+extract (once, cached) and parse a WikiSQL split.
    Returns a list of raw dicts compatible with _normalize_wikisql().
    """
    _ensure_wikisql_extracted()

    qa_rel, tables_rel = _WIKISQL_SPLIT_MAP[split]
    qa_path     = _WIKISQL_CACHE_DIR / qa_rel
    tables_path = _WIKISQL_CACHE_DIR / tables_rel

    # Build table_id → table dict
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
            ex = json.loads(line)
            tid = ex.get("table_id", "")
            table = tables.get(tid, {"id": tid, "header": [], "types": [], "rows": []})

            # Raw WikiSQL conds format: [[col_idx, op_idx, value], ...]
            # Normalize to the HF dict format that _normalize_wikisql / _reconstruct_wikisql_query expects
            sql_raw = ex.get("sql", {})
            conds_raw = sql_raw.get("conds", [])
            if conds_raw and isinstance(conds_raw[0], list):
                conds_hf = {
                    "column_index":   [c[0] for c in conds_raw],
                    "operator_index": [c[1] for c in conds_raw],
                    "condition":      [c[2] for c in conds_raw],
                }
            else:
                conds_hf = conds_raw  # already in dict form
            sql_hf = {
                "sel":            sql_raw.get("sel", 0),
                "agg":            sql_raw.get("agg", 0),
                "conds":          conds_hf,
                "human_readable": None,
            }
            examples.append({
                "question": ex.get("question", ""),
                "sql":      sql_hf,
                "table":    table,
            })
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Dataset merger
# ──────────────────────────────────────────────────────────────────────────────

def merge_datasets(
    spider_name: str = "xlangai/spider",
    wikisql_name: str = "wikisql",   # kept for API compat; data is fetched directly
    include_wikisql_validation: bool = True,
    max_train_samples: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load Spider and WikiSQL, apply normalisation, and return three lists.

    Returns
    -------
    train_examples : List[Dict]
        Spider train + WikiSQL train (+ WikiSQL validation if flag set).
        Combined total ≈ 87 K examples.
    spider_dev : List[Dict]
        Spider validation split (~1 034 examples). Used for EM / EX eval.
    spider_test : List[Dict]
        Spider test split (gold queries may be absent on some HF versions).
    """
    def _map(raw_split, fn) -> List[Dict]:
        out = [fn(ex) for ex in raw_split]
        return [r for r in out if r is not None]

    logger.info("Loading Spider from HuggingFace …")
    spider_ds = load_dataset(spider_name)

    logger.info("Loading WikiSQL directly from GitHub (datasets 4.x compat) …")
    wikisql_train_raw = _load_wikisql_split("train")
    wikisql_val_raw   = _load_wikisql_split("validation") if include_wikisql_validation else []

    spider_train = _map(spider_ds["train"],      _normalize_spider)
    spider_dev   = _map(spider_ds["validation"], _normalize_spider)
    spider_test  = _map(spider_ds.get("test", []), _normalize_spider)

    wikisql_train = _map(wikisql_train_raw, _normalize_wikisql)
    wikisql_train += _map(wikisql_val_raw,  _normalize_wikisql)

    train = spider_train + wikisql_train

    if max_train_samples is not None:
        train = train[:max_train_samples]

    logger.info(
        "Dataset summary — Spider train: %d | WikiSQL train: %d | "
        "Combined train: %d | Dev: %d | Test: %d",
        len(spider_train), len(wikisql_train),
        len(train), len(spider_dev), len(spider_test),
    )
    return train, spider_dev, spider_test


# ──────────────────────────────────────────────────────────────────────────────
# Custom collator for eval batches (tensors + string metadata)
# ──────────────────────────────────────────────────────────────────────────────

def eval_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    DataLoader collator for evaluation mode.
    Stacks tensor fields and keeps string fields as plain Python lists.
    """
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "gold_query":     [b["gold_query"] for b in batch],
        "difficulty":     [b["difficulty"] for b in batch],
        "db_id":          [b["db_id"] for b in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SQLDataset(Dataset):
    """
    PyTorch Dataset for SQL generation fine-tuning and evaluation.

    Training mode
    -------------
    Returns ``{input_ids, attention_mask, labels}``.
    Labels are a copy of input_ids with the prompt tokens set to -100
    so cross-entropy loss is computed only on SQL tokens.

    Eval mode
    ---------
    Returns ``{input_ids, attention_mask, gold_query, difficulty, db_id}``.
    Use ``eval_collate_fn`` as the DataLoader ``collate_fn``.

    SFTTrainer compatibility
    ------------------------
    Call ``dataset.as_hf_dataset()`` to get a HuggingFace Dataset with a
    ``text`` column containing the full prompt + SQL string.

    Parameters
    ----------
    examples:
        List of normalised example dicts (from ``merge_datasets``).
    tokenizer:
        HuggingFace tokenizer (must have pad_token set).
    schema_linker:
        ``HeuristicSchemaLinker`` instance.
    prompt_template:
        Format string with ``{schema}`` and ``{question}`` placeholders.
    max_length:
        Maximum sequence length (prompt + SQL combined).
    is_training:
        If True, build label tensors for training; else return raw metadata.
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
        self.tokenizer = tokenizer
        self.schema_linker = schema_linker
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.is_training = is_training

        logger.info("Preprocessing %d examples …", len(examples))
        processed = [self._preprocess(ex) for ex in examples]
        self.data: List[Dict] = [p for p in processed if p is not None]
        skipped = len(examples) - len(self.data)
        if skipped:
            logger.warning("Skipped %d examples during preprocessing.", skipped)
        logger.info("SQLDataset ready — %d examples.", len(self.data))

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocessing (applied once at init time, not per __getitem__ call)
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, example: Dict) -> Optional[Dict]:
        try:
            filtered_schema = self.schema_linker.filter_schema(
                example["schema"], example["question"]
            )
            prompt = self.prompt_template.format(
                schema=filtered_schema,
                question=example["question"],
            )
            # Normalise SQL: strip trailing semicolons (we append one explicitly)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        prompt: str = item["prompt"]

        if self.is_training:
            return self._training_item(prompt, item["query"])
        else:
            return self._eval_item(prompt, item)

    def _training_item(self, prompt: str, query: str) -> Dict[str, torch.Tensor]:
        """
        Tokenise (prompt + SQL) and build label tensor with prompt masked to -100.
        """
        full_text = f"{prompt}{query} ;"

        enc_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Tokenise prompt alone to find the boundary index
        enc_prompt = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = enc_full["input_ids"].squeeze(0)
        attention_mask: torch.Tensor = enc_full["attention_mask"].squeeze(0)
        labels: torch.Tensor = input_ids.clone()

        prompt_len = min(enc_prompt["input_ids"].shape[1], self.max_length)
        labels[:prompt_len] = -100   # ignore prompt in loss

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def _eval_item(self, prompt: str, item: Dict) -> Dict[str, Any]:
        """
        Tokenise prompt (no padding — left-padding is applied per-batch in eval).
        """
        enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # String metadata — collated by eval_collate_fn
            "gold_query": item["query"],
            "difficulty": item["difficulty"],
            "db_id":      item["db_id"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SFTTrainer helper
    # ──────────────────────────────────────────────────────────────────────────

    def as_hf_dataset(self) -> HFDataset:
        """
        Return a HuggingFace Dataset with a single ``text`` column
        (prompt + SQL + semicolon).  Pass this directly to ``SFTTrainer``.
        """
        texts = [
            f"{d['prompt']}{d['query']} ;"
            for d in self.data
        ]
        return HFDataset.from_dict({"text": texts})

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def difficulty_distribution(self) -> Dict[str, int]:
        """Return a count of examples per difficulty bucket."""
        from collections import Counter
        return dict(Counter(d["difficulty"] for d in self.data))
