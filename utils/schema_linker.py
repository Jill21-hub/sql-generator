"""
utils/schema_linker.py — Production-Ready Heuristic Schema Linker
==================================================================
# PROPOSAL_REF: Enhancement 2 — Schema Linking Preprocessing

Filters a database schema down to only the tables and columns relevant
to the natural-language question, keeping prompts compact.

Strategy
--------
1. Tokenise the question (lower-case, strip punctuation, remove stop-words).
2. Split each table/column identifier into sub-tokens
   (snake_case → ["student", "id"], CamelCase → ["student", "name"]).
3. Keep a table if ANY identifier tokens overlap with question tokens,
   OR if any of its columns match.
4. When nothing matches, fall back to the first two tables.

Output: Table1(col1, col2) | Table2(col3, col4)

No external embeddings or DB catalogues required — pure heuristics.

Usage:
    python utils/schema_linker.py          # run built-in test suite
"""

from __future__ import annotations

import logging
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional NLTK (graceful fallback to str.split)
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

except Exception:
    _NLTK_AVAILABLE = False
    _NLTK_STOP_WORDS: Set[str] = set()
    logger.debug("NLTK not available — using fallback tokeniser.")


# ──────────────────────────────────────────────────────────────────────────────
# Domain stop-words (SQL / NL query words that add no schema signal)
# ──────────────────────────────────────────────────────────────────────────────

_DOMAIN_STOP_WORDS: Set[str] = {
    "find", "show", "list", "give", "get", "return", "display",
    "what", "which", "who", "where", "when", "how", "why",
    "many", "much", "number", "count",
    "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "with", "that", "this", "these", "those",
    "and", "or", "but", "not",
    "all", "any", "each", "every", "some",
    "between", "across", "by", "from", "into",
    "me", "us", "their", "its",
}

SchemaDict = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# HeuristicSchemaLinker
# ──────────────────────────────────────────────────────────────────────────────

class HeuristicSchemaLinker:
    """
    Keyword-overlap schema linker.

    Parameters
    ----------
    min_token_length      : Ignore identifier sub-tokens shorter than this.
    max_columns_per_table : Hard cap on columns included per matched table.
    """

    def __init__(
        self,
        min_token_length: int = 2,
        max_columns_per_table: int = 12,
    ) -> None:
        self.min_token_length      = min_token_length
        self.max_columns_per_table = max_columns_per_table
        self._punct_re   = re.compile(f"[{re.escape(string.punctuation)}]")
        self._stop_words: Set[str] = _DOMAIN_STOP_WORDS | _NLTK_STOP_WORDS

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize_question(self, question: str) -> Set[str]:
        text = question.lower()
        if _NLTK_AVAILABLE:
            try:
                raw = _nltk_tokenize(text)
            except Exception:
                raw = self._punct_re.sub(" ", text).split()
        else:
            raw = self._punct_re.sub(" ", text).split()

        return {
            t for t in raw
            if len(t) >= self.min_token_length and t not in self._stop_words
        }

    def _split_identifier(self, identifier: str) -> Set[str]:
        normed = identifier.lower().replace("-", "_").replace(" ", "_")
        normed = re.sub(r"([a-z])([A-Z])", r"\1_\2", normed)
        parts  = normed.split("_")
        return {p for p in parts if len(p) >= self.min_token_length}

    def _overlaps(self, id_tokens: Set[str], q_tokens: Set[str]) -> bool:
        if id_tokens & q_tokens:
            return True
        # Handle plural/singular and prefix: "singers" ↔ "singer", "countries" ↔ "country"
        for id_tok in id_tokens:
            for q_tok in q_tokens:
                min_len = min(len(id_tok), len(q_tok))
                if min_len >= 4 and (q_tok.startswith(id_tok) or id_tok.startswith(q_tok)):
                    return True
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def filter_schema(self, schema_json: SchemaDict, question: str) -> str:
        """
        Return a compact schema string with only relevant tables/columns.

        Output format:
            Table1(col1, col2) | Table2(col3, col4)

        Parameters
        ----------
        schema_json : Normalised schema dict with 'tables' key.
        question    : Natural-language question.
        """
        if not schema_json:
            return ""

        q_tokens: Set[str] = self._tokenize_question(question)
        tables: List[Dict] = schema_json.get("tables", [])
        matched_parts: List[str] = []

        for table in tables:
            table_name: str = table.get("name", "")
            if not table_name:
                continue

            t_tokens      = self._split_identifier(table_name)
            table_matched = self._overlaps(t_tokens, q_tokens)

            matched_cols: List[str] = []
            all_cols:     List[str] = []

            for col in table.get("columns", []):
                col_name = col.get("name", "")
                if not col_name or col_name == "*":
                    continue
                all_cols.append(col_name)

                c_tokens = self._split_identifier(col_name)
                if table_matched or self._overlaps(c_tokens, q_tokens):
                    matched_cols.append(col_name)

            if matched_cols:
                cols_str = ", ".join(matched_cols[: self.max_columns_per_table])
                matched_parts.append(f"{table_name}({cols_str})")
            elif table_matched:
                cols_str = ", ".join(all_cols[: self.max_columns_per_table])
                matched_parts.append(f"{table_name}({cols_str})")

        # Fallback: return first two tables so the model has at least some context
        if not matched_parts:
            logger.debug(
                "SchemaLinker: no overlap for '%s'. Falling back to first 2 tables.",
                question[:80],
            )
            for table in tables[:2]:
                tname = table.get("name", "")
                cols  = [
                    c.get("name", "")
                    for c in table.get("columns", [])
                    if c.get("name") and c.get("name") != "*"
                ]
                matched_parts.append(f"{tname}({', '.join(cols[:6])})")

        return " | ".join(matched_parts)

    def explain(self, schema_json: SchemaDict, question: str) -> Dict[str, Any]:
        """Debug helper — returns question tokens and per-table match details."""
        q_tokens = self._tokenize_question(question)
        breakdown = []
        for table in schema_json.get("tables", []):
            tname    = table.get("name", "")
            t_tokens = self._split_identifier(tname)
            col_matches = {}
            for col in table.get("columns", []):
                cname    = col.get("name", "")
                c_tokens = self._split_identifier(cname)
                col_matches[cname] = bool(c_tokens & q_tokens)
            breakdown.append({
                "table":          tname,
                "table_tokens":   t_tokens,
                "table_matched":  bool(t_tokens & q_tokens),
                "column_matches": col_matches,
            })
        return {"question_tokens": q_tokens, "tables": breakdown}

    def test(self) -> bool:
        """Run the built-in test suite. Returns True if all cases pass."""
        return test_linker()


# ──────────────────────────────────────────────────────────────────────────────
# Built-in test suite — presentation-ready console output
# PROPOSAL_REF: Enhancement 2
# ──────────────────────────────────────────────────────────────────────────────

_TEST_CASES = [
    {
        "name": "Single-table match (easy)",
        "question": "How many singers are there?",
        "schema": {
            "db_id": "concert_singer",
            "tables": [
                {"name": "singer", "columns": [
                    {"name": "singer_id"}, {"name": "name"},
                    {"name": "country"},  {"name": "age"},
                ]},
                {"name": "concert", "columns": [
                    {"name": "concert_id"}, {"name": "concert_name"},
                    {"name": "theme"},      {"name": "stadium_id"},
                ]},
                {"name": "stadium", "columns": [
                    {"name": "stadium_id"}, {"name": "location"},
                    {"name": "name"},       {"name": "capacity"},
                ]},
            ],
        },
        "expected_tables": ["singer"],
    },
    {
        "name": "Multi-table JOIN (medium)",
        "question": "What are the names and countries of singers who performed at a concert?",
        "schema": {
            "db_id": "concert_singer",
            "tables": [
                {"name": "singer", "columns": [
                    {"name": "singer_id"}, {"name": "name"}, {"name": "country"},
                ]},
                {"name": "singer_in_concert", "columns": [
                    {"name": "concert_id"}, {"name": "singer_id"},
                ]},
                {"name": "concert", "columns": [
                    {"name": "concert_id"}, {"name": "concert_name"}, {"name": "theme"},
                ]},
            ],
        },
        "expected_tables": ["singer", "singer_in_concert", "concert"],
    },
    {
        "name": "No overlap → fallback (hard)",
        "question": "Which department has the highest budget allocation?",
        "schema": {
            "db_id": "university",
            "tables": [
                {"name": "faculty", "columns": [
                    {"name": "faculty_id"}, {"name": "rank"}, {"name": "salary"},
                ]},
                {"name": "enrollment", "columns": [
                    {"name": "student_id"}, {"name": "course_id"}, {"name": "grade"},
                ]},
            ],
        },
        "expected_tables": ["faculty", "enrollment"],  # fallback returns both
    },
    {
        "name": "Column-level match (medium)",
        "question": "What is the average age and country of all singers?",
        "schema": {
            "db_id": "concert_singer",
            "tables": [
                {"name": "singer", "columns": [
                    {"name": "singer_id"}, {"name": "name"},
                    {"name": "country"},  {"name": "age"},
                ]},
                {"name": "concert", "columns": [
                    {"name": "concert_id"}, {"name": "concert_name"},
                ]},
            ],
        },
        "expected_tables": ["singer"],
    },
]


def test_linker() -> bool:
    """
    Run built-in schema-linking test cases.
    # PROPOSAL_REF: Enhancement 2
    Returns True if all tests pass.
    """
    linker = HeuristicSchemaLinker()
    sep    = "─" * 66

    print()
    print("=" * 66)
    print("  Schema Linker — Built-in Test Suite")
    print("  PROPOSAL_REF: Enhancement 2")
    print("=" * 66)

    all_passed = True
    for i, tc in enumerate(_TEST_CASES, 1):
        schema   = tc["schema"]
        question = tc["question"]
        expected = set(tc.get("expected_tables", []))

        all_tables = schema.get("tables", [])
        full_repr  = " | ".join(
            f"{t['name']}({', '.join(c['name'] for c in t.get('columns', []))})"
            for t in all_tables
        )

        filtered = linker.filter_schema(schema, question)
        explain  = linker.explain(schema, question)

        # Check that all expected tables appear in output
        got_tables = set(re.findall(r"(\w+)\(", filtered))
        passed     = expected.issubset(got_tables) if expected else bool(filtered)
        all_passed = all_passed and passed

        n_before = sum(len(t.get("columns", [])) for t in all_tables)
        n_after  = len(re.findall(r"\w+", filtered.replace("|", ""))) - filtered.count("(")

        print()
        print(f"  Test {i}: {tc['name']}")
        print(sep)
        print(f"  Question  : {question}")
        print(f"  Q-tokens  : {sorted(explain['question_tokens'])}")
        print()
        print(f"  BEFORE ({len(all_tables)} tables, ~{n_before} cols):")
        print(f"    {full_repr}")
        print()
        print(f"  AFTER  (filtered):")
        print(f"    {filtered}")
        print()
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
        print(sep)

    print()
    print(f"  Overall: {'✅ All tests passed' if all_passed else '❌ Some tests failed'}")
    print("=" * 66)
    print()
    return all_passed


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_linker()
