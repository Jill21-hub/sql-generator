"""
Heuristic Schema Linker
=======================
Filters a database schema down to only the tables and columns that are
relevant to the natural-language question.  This keeps prompts compact and
prevents the model from being distracted by irrelevant schema nodes.

Strategy
--------
1. Tokenise the question (lower-case, strip punctuation, remove stop-words).
2. For every table/column, split its identifier into sub-tokens
   (snake_case → ["student", "id"], CamelCase → ["student", "name"]).
3. Keep a table if ANY of its identifier tokens overlap with the question
   tokens, OR if any of its columns match.
4. When nothing matches, fall back to the first two tables (minimal context).

No external embeddings or DB catalogues are required — pure heuristics.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional NLTK integration (graceful fallback to str.split)
# ──────────────────────────────────────────────────────────────────────────────

try:
    import nltk

    def _ensure_nltk(resource_path: str, download_name: str) -> None:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)

    _ensure_nltk("tokenizers/punkt_tab", "punkt_tab")
    _ensure_nltk("corpora/stopwords", "stopwords")

    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.tokenize import word_tokenize as _nltk_tokenize

    _NLTK_AVAILABLE = True
    _NLTK_STOP_WORDS: Set[str] = set(_nltk_stopwords.words("english"))

except Exception:  # pragma: no cover
    _NLTK_AVAILABLE = False
    _NLTK_STOP_WORDS = set()
    logger.debug("NLTK not available — using fallback tokeniser.")


# ──────────────────────────────────────────────────────────────────────────────
# Schema type alias
# ──────────────────────────────────────────────────────────────────────────────

# Normalised schema format used throughout the project:
#
#   {
#     "db_id": "concert_singer",
#     "tables": [
#       {
#         "name": "stadium",
#         "columns": [
#           {"name": "stadium_id", "type": "number"},
#           {"name": "location",   "type": "text"},
#           ...
#         ]
#       },
#       ...
#     ]
#   }

SchemaDict = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific stop-words to drop from question tokens before matching
# ──────────────────────────────────────────────────────────────────────────────

_DOMAIN_STOP_WORDS: Set[str] = {
    # Question starters / verbs that never appear as column names
    "find", "show", "list", "give", "get", "return", "display",
    "what", "which", "who", "where", "when", "how", "why",
    "many", "much", "number", "count",
    # Common English function words not caught by NLTK
    "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "with", "that", "this", "these", "those",
    "and", "or", "but", "not",
    "all", "any", "each", "every", "some",
    "between", "across", "by", "from", "into",
}


class HeuristicSchemaLinker:
    """
    Keyword-overlap schema linker.

    Parameters
    ----------
    min_token_length:
        Identifier sub-tokens shorter than this are ignored.
    max_columns_per_table:
        Hard cap on columns included per matched table (prevents prompt bloat
        when a wide table partially matches).
    """

    def __init__(
        self,
        min_token_length: int = 2,
        max_columns_per_table: int = 12,
    ) -> None:
        self.min_token_length = min_token_length
        self.max_columns_per_table = max_columns_per_table
        self._punct_re = re.compile(f"[{re.escape(string.punctuation)}]")

        # Combined stop-word set
        self._stop_words: Set[str] = _DOMAIN_STOP_WORDS | _NLTK_STOP_WORDS

    # ──────────────────────────────────────────────────────────────────────────
    # Tokenisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _tokenize_question(self, question: str) -> Set[str]:
        """Return a cleaned set of content tokens from the question."""
        text = question.lower()
        if _NLTK_AVAILABLE:
            try:
                raw_tokens = _nltk_tokenize(text)
            except Exception:
                raw_tokens = self._punct_re.sub(" ", text).split()
        else:
            raw_tokens = self._punct_re.sub(" ", text).split()

        return {
            t for t in raw_tokens
            if len(t) >= self.min_token_length and t not in self._stop_words
        }

    def _split_identifier(self, identifier: str) -> Set[str]:
        """
        Decompose a DB identifier into matching-ready sub-tokens.

        Examples
        --------
        "student_id"   → {"student", "id"}
        "StadiumName"  → {"stadium", "name"}
        "first_name"   → {"first", "name"}
        """
        # Normalise separators
        normed = identifier.lower().replace("-", "_").replace(" ", "_")
        # Handle CamelCase: insert separator before uppercase runs
        normed = re.sub(r"([a-z])([A-Z])", r"\1_\2", normed)
        parts = normed.split("_")
        return {p for p in parts if len(p) >= self.min_token_length}

    def _overlaps(self, identifier_tokens: Set[str], question_tokens: Set[str]) -> bool:
        return bool(identifier_tokens & question_tokens)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def filter_schema(
        self,
        schema_json: SchemaDict,
        question: str,
    ) -> str:
        """
        Return a compact schema string containing only relevant tables/columns.

        Output format (one line per table):
            DB: <db_id>
            <table_name>: <col1>(<type>), <col2>(<type>), ...
            <table_name>: <col1>(<type>), ...

        Parameters
        ----------
        schema_json:
            Normalised schema dict (see module docstring for exact shape).
        question:
            The natural-language question to answer with SQL.

        Returns
        -------
        str
            Compact schema string ready to inject into the prompt.
        """
        if not schema_json:
            return ""

        q_tokens = self._tokenize_question(question)
        tables: List[Dict] = schema_json.get("tables", [])

        matched_lines: List[str] = []

        for table in tables:
            table_name: str = table.get("name", "")
            if not table_name:
                continue

            table_id_tokens = self._split_identifier(table_name)
            table_matched = self._overlaps(table_id_tokens, q_tokens)

            matched_cols: List[str] = []
            all_col_reprs: List[str] = []

            for col in table.get("columns", []):
                col_name: str = col.get("name", "")
                if not col_name or col_name == "*":
                    continue

                col_type: str = col.get("type", "")
                col_repr = f"{col_name}({col_type})" if col_type else col_name
                all_col_reprs.append(col_repr)

                col_id_tokens = self._split_identifier(col_name)
                if table_matched or self._overlaps(col_id_tokens, q_tokens):
                    matched_cols.append(col_repr)

            if matched_cols:
                # Some columns explicitly matched — include only those
                cols_str = ", ".join(matched_cols[: self.max_columns_per_table])
                matched_lines.append(f"{table_name}: {cols_str}")

            elif table_matched:
                # Table name matched but no column did — include all cols (capped)
                cols_str = ", ".join(all_col_reprs[: self.max_columns_per_table])
                matched_lines.append(f"{table_name}: {cols_str}")

        # Fallback: nothing matched → return the first two tables so the model
        # at least has some schema context rather than an empty prompt.
        if not matched_lines:
            logger.debug(
                "SchemaLinker: no overlap found for question '%s'. "
                "Falling back to first 2 tables.",
                question[:80],
            )
            for table in tables[:2]:
                tname = table.get("name", "")
                cols = [
                    f"{c.get('name', '')}({c.get('type', '')})"
                    for c in table.get("columns", [])
                    if c.get("name") and c.get("name") != "*"
                ]
                matched_lines.append(f"{tname}: {', '.join(cols[:6])}")

        db_id: str = schema_json.get("db_id", "")
        header = f"DB: {db_id}\n" if db_id else ""
        return header + "\n".join(matched_lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def explain(self, schema_json: SchemaDict, question: str) -> Dict[str, Any]:
        """
        Debug helper — returns question tokens and per-table match details.
        Not used in the training/inference pipeline.
        """
        q_tokens = self._tokenize_question(question)
        breakdown = []
        for table in schema_json.get("tables", []):
            tname = table.get("name", "")
            t_tokens = self._split_identifier(tname)
            col_matches = {}
            for col in table.get("columns", []):
                cname = col.get("name", "")
                c_tokens = self._split_identifier(cname)
                col_matches[cname] = bool(c_tokens & q_tokens)
            breakdown.append({
                "table": tname,
                "table_tokens": t_tokens,
                "table_matched": bool(t_tokens & q_tokens),
                "column_matches": col_matches,
            })
        return {"question_tokens": q_tokens, "tables": breakdown}
