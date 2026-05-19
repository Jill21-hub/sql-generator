"""
data/loader_local.py — Robust Local Data Pipeline
===================================================
Loads Spider (train + dev) from HuggingFace datasets with disk caching.
Falls back to a local JSON cache if HF is slow or unavailable.

Usage:
    python data/loader_local.py                # full Spider train
    python data/loader_local.py --subset 10    # 10 examples, fast local test
    python data/loader_local.py --split dev    # dev split only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import requests

# ── Project root on sys.path so sibling imports work ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows terminals default to cp1252 — force UTF-8 so emoji/box chars render
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Local JSON fallback cache (written once, reused on HF outages)
_LOCAL_CACHE_DIR = Path(os.path.expanduser("~/.cache/sql_generator_local"))

# Spider tables.json — canonical schema source (HF rows lack schema fields)
_SPIDER_TABLES_URL   = "https://raw.githubusercontent.com/taoyds/spider/master/tables.json"
_SPIDER_TABLES_CACHE = _LOCAL_CACHE_DIR / "spider_tables.json"

# ──────────────────────────────────────────────────────────────────────────────
# Built-in schema registry for common Spider dev databases
# Covers the databases that appear most frequently in the dev split.
# Avoids any external download — works fully offline after first HF load.
# ──────────────────────────────────────────────────────────────────────────────

def _t(name: str, cols: List) -> Dict:
    return {"name": name, "columns": [{"name": c, "type": ""} for c in cols]}


_BUILTIN_SCHEMAS: Dict[str, Dict] = {
    "concert_singer": {"db_id": "concert_singer", "tables": [
        _t("stadium",          ["stadium_id", "location", "name", "capacity", "highest", "lowest", "average"]),
        _t("singer",           ["singer_id", "name", "country", "song_name", "song_release_year", "age", "is_male"]),
        _t("concert",          ["concert_id", "concert_name", "theme", "stadium_id", "year"]),
        _t("singer_in_concert",["concert_id", "singer_id"]),
    ]},
    "pets_1": {"db_id": "pets_1", "tables": [
        _t("Student", ["StuID", "LName", "Fname", "Age", "Sex", "Major", "Advisor", "city_code"]),
        _t("Has_Pet", ["StuID", "PetID"]),
        _t("Pets",    ["PetID", "PetType", "pet_age", "weight"]),
    ]},
    "car_1": {"db_id": "car_1", "tables": [
        _t("continents",   ["ContId", "Continent"]),
        _t("countries",    ["CountryId", "CountryName", "Continent"]),
        _t("car_makers",   ["Id", "Maker", "FullName", "Country"]),
        _t("model_list",   ["ModelId", "Maker", "Model"]),
        _t("car_names",    ["MakeId", "Model", "Make"]),
        _t("cars_data",    ["Id", "MPG", "Cylinders", "Edispl", "Horsepower", "Weight", "Accelerate", "Year"]),
    ]},
    "employee_hire_evaluation": {"db_id": "employee_hire_evaluation", "tables": [
        _t("employee",   ["Employee_ID", "Name", "Age", "City"]),
        _t("shop",       ["Shop_ID", "Name", "Location", "District", "Number_products", "Manager_name"]),
        _t("hiring",     ["Shop_ID", "Employee_ID", "Start_from", "Is_full_time"]),
        _t("evaluation", ["Employee_ID", "Year_awarded", "Bonus"]),
    ]},
    "flight_2": {"db_id": "flight_2", "tables": [
        _t("airlines",  ["uid", "Airline", "Abbreviation", "Country"]),
        _t("airports",  ["City", "AirportCode", "AirportName", "Country", "CountryAbbrev"]),
        _t("flights",   ["Airline", "FlightNo", "SourceAirport", "DestAirport"]),
    ]},
    "world_1": {"db_id": "world_1", "tables": [
        _t("city",         ["ID", "Name", "CountryCode", "District", "Population"]),
        _t("sqlite_sequence",["name", "seq"]),
        _t("country",      ["Code", "Name", "Continent", "Region", "SurfaceArea",
                             "IndepYear", "Population", "LifeExpectancy", "GNP",
                             "GNPOld", "LocalName", "GovernmentForm", "HeadOfState",
                             "Capital", "Code2"]),
        _t("countrylanguage",["CountryCode", "Language", "IsOfficial", "Percentage"]),
    ]},
    "network_1": {"db_id": "network_1", "tables": [
        _t("Highschooler", ["ID", "name", "grade"]),
        _t("Friend",       ["student_id", "friend_id"]),
        _t("Likes",        ["student_id", "liked_id"]),
    ]},
    "dog_kennels": {"db_id": "dog_kennels", "tables": [
        _t("Breeds",            ["breed_code", "breed_name"]),
        _t("Charges",           ["charge_id", "charge_type", "charge_amount"]),
        _t("Sizes",             ["size_code", "size_description"]),
        _t("Treatment_Types",   ["treatment_type_code", "treatment_type_description"]),
        _t("Owners",            ["owner_id", "first_name", "last_name", "street", "city", "state", "zip_code", "email_address", "home_phone", "cell_number"]),
        _t("Dogs",              ["dog_id", "owner_id", "abandoned_yn", "breed_code", "size_code", "name", "age", "date_of_birth", "gender", "weight", "date_arrived", "date_adopted", "date_departed"]),
        _t("Professionals",     ["professional_id", "role_code", "first_name", "street", "city", "state", "zip_code", "last_name", "email_address", "home_phone", "cell_number"]),
        _t("Treatments",        ["treatment_id", "dog_id", "professional_id", "treatment_type_code", "date_of_treatment", "cost_of_treatment"]),
    ]},
    "student_transcripts_tracking": {"db_id": "student_transcripts_tracking", "tables": [
        _t("Addresses",              ["address_id", "line_1", "line_2", "line_3", "city", "zip_postcode", "state_province_county", "country", "other_address_details"]),
        _t("Courses",                ["course_id", "course_name", "course_description", "other_details"]),
        _t("Departments",            ["department_id", "department_name", "department_description", "other_details"]),
        _t("Degree_Programs",        ["degree_program_id", "department_id", "degree_summary_name", "degree_summary_description", "other_details"]),
        _t("Sections",               ["section_id", "course_id", "section_name", "section_description", "other_details"]),
        _t("Semesters",              ["semester_id", "semester_name", "semester_description", "other_details"]),
        _t("Students",               ["student_id", "current_address_id", "permanent_address_id", "first_name", "middle_name", "last_name", "cell_mobile_number", "email_address", "ssn", "date_first_registered", "date_left", "other_student_details"]),
        _t("Student_Enrolment",      ["student_enrolment_id", "degree_program_id", "semester_id", "student_id", "other_details"]),
        _t("Student_Enrolment_Courses",["student_course_id", "course_id", "student_enrolment_id"]),
        _t("Transcripts",            ["transcript_id", "transcript_date", "other_details"]),
        _t("Transcript_Contents",    ["student_course_id", "transcript_id"]),
    ]},
    "poker_player": {"db_id": "poker_player", "tables": [
        _t("people",       ["People_ID", "Nationality", "Name", "Birth_Date", "Height"]),
        _t("poker_player", ["Poker_Player_ID", "People_ID", "Final_Table_Made", "Best_Finish", "Money_Rank", "Earnings"]),
    ]},
}

# Module-level schema registry: db_id → normalised schema dict
_SCHEMA_REGISTRY: Dict[str, Dict] = dict(_BUILTIN_SCHEMAS)


def _ensure_schema_registry() -> None:
    """
    Try to enrich the registry from a cached tables.json download.
    If unavailable, the built-in stubs (covering the most common Spider dev
    databases) are used silently.
    """
    global _SCHEMA_REGISTRY
    if _SPIDER_TABLES_CACHE.exists():
        try:
            tables_list = json.loads(_SPIDER_TABLES_CACHE.read_text(encoding="utf-8"))
            for db in tables_list:
                db_id       = db.get("db_id", "")
                table_names = db.get("table_names_original", db.get("table_names", []))
                col_names   = db.get("column_names_original", db.get("column_names", []))
                col_types   = db.get("column_types", [])
                tbl_map: Dict[int, Dict] = {
                    i: {"name": n, "columns": []}
                    for i, n in enumerate(table_names)
                }
                for idx, (tid, cname) in enumerate(col_names):
                    if tid < 0:
                        continue
                    ctype = col_types[idx] if idx < len(col_types) else ""
                    if tid in tbl_map:
                        tbl_map[tid]["columns"].append({"name": cname, "type": ctype})
                _SCHEMA_REGISTRY[db_id] = {"db_id": db_id, "tables": list(tbl_map.values())}
            logger.info("Schema registry enriched to %d databases from cache", len(_SCHEMA_REGISTRY))
        except Exception as exc:
            logger.debug("Could not parse cached tables.json: %s", exc)
_SPIDER_CACHE = {
    "train": _LOCAL_CACHE_DIR / "spider_train.json",
    "dev":   _LOCAL_CACHE_DIR / "spider_dev.json",
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard", "extra hard"]


# ──────────────────────────────────────────────────────────────────────────────
# Schema parsing  (mirrors data/dataset.py — kept self-contained for portability)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_spider_schema(example: Dict) -> Dict:
    """Convert Spider's flat parallel arrays into the normalised schema dict."""
    table_names = example.get("db_table_names", [])
    col_info    = example.get("db_column_names", {})
    col_types   = example.get("db_column_types", [])

    table_id_list = col_info.get("table_id", [])
    col_name_list = col_info.get("column_name", [])

    tables = {i: {"name": n, "columns": []} for i, n in enumerate(table_names)}
    for idx, (tid, cname) in enumerate(zip(table_id_list, col_name_list)):
        if tid < 0:
            continue
        ctype = col_types[idx] if idx < len(col_types) else ""
        if tid in tables:
            tables[tid]["columns"].append({"name": cname, "type": ctype})

    return {"db_id": example.get("db_id", ""), "tables": list(tables.values())}


def _estimate_difficulty(sql: str) -> str:
    up = sql.upper()
    score = (
        up.count("JOIN")
        + (up.count("SELECT") - 1) * 2
        + ("GROUP BY" in up)
        + ("HAVING"   in up)
        + any(k in up for k in ("INTERSECT", "UNION", "EXCEPT")) * 2
    )
    if score == 0:   return "easy"
    if score <= 2:   return "medium"
    if score <= 4:   return "hard"
    return "extra hard"


def _normalize(example: Dict) -> Optional[Dict]:
    try:
        query = (example.get("query") or "").strip()
        if not query:
            return None

        db_id = example.get("db_id", "")

        # HF xlangai/spider rows don't include schema fields — look up registry.
        # Fall back to inline parsing for datasets that do embed schema info.
        if db_id in _SCHEMA_REGISTRY:
            schema = _SCHEMA_REGISTRY[db_id]
        elif example.get("db_table_names"):
            schema = _parse_spider_schema(example)
        else:
            schema = {"db_id": db_id, "tables": []}

        return {
            "db_id":      db_id,
            "schema":     schema,
            "question":   (example.get("question") or "").strip(),
            "query":      query,
            "difficulty": example.get("difficulty") or _estimate_difficulty(query),
            "source":     "spider",
        }
    except Exception as exc:
        logger.debug("Skipped example: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace loader with local JSON fallback
# ──────────────────────────────────────────────────────────────────────────────

def _load_from_hf(split: str) -> List[Dict]:
    from datasets import load_dataset
    _ensure_schema_registry()   # populate before normalising
    logger.info("Loading Spider '%s' split from HuggingFace …", split)
    hf_split = "validation" if split == "dev" else split
    ds = load_dataset("xlangai/spider", split=hf_split)
    examples = [_normalize(ex) for ex in ds]
    return [e for e in examples if e is not None]


def _load_from_cache(split: str) -> List[Dict]:
    path = _SPIDER_CACHE[split]
    logger.info("Loading Spider '%s' split from local cache: %s", split, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_to_cache(split: str, examples: List[Dict]) -> None:
    _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _SPIDER_CACHE[split]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    logger.info("Cached %d examples → %s", len(examples), path)


def load_spider(
    split: str = "train",
    subset: Optional[int] = None,
) -> List[Dict]:
    """
    Load Spider split with HF → local cache fallback.

    Parameters
    ----------
    split   : "train" or "dev"
    subset  : if set, return only the first N examples (fast local testing)
    """
    examples: List[Dict] = []

    try:
        examples = _load_from_hf(split)
        _save_to_cache(split, examples)          # refresh cache on success
    except Exception as exc:
        logger.warning("HuggingFace load failed (%s). Trying local cache …", exc)
        if _SPIDER_CACHE[split].exists():
            examples = _load_from_cache(split)
        else:
            raise RuntimeError(
                f"HF unavailable and no local cache found at {_SPIDER_CACHE[split]}. "
                "Run once with internet access to populate the cache."
            ) from exc

    if subset is not None:
        examples = examples[:subset]

    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer  (presentation-ready console output)
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(examples: List[Dict], split: str) -> None:
    counts = Counter(e["difficulty"] for e in examples)
    total  = len(examples)

    print()
    print("=" * 56)
    print(f"  Spider — '{split}' split")
    print("=" * 56)
    print(f"  ✅ Loaded {total} examples")
    print()
    for level in DIFFICULTY_LEVELS:
        n   = counts.get(level, 0)
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 5)
        print(f"  {level:<12} {n:>5}  ({pct:5.1f}%)  {bar}")
    print("=" * 56)
    print()

    # Show one sample per difficulty for a quick visual check
    seen = set()
    print("  Sample questions per difficulty:")
    print()
    for ex in examples:
        d = ex["difficulty"]
        if d not in seen:
            seen.add(d)
            print(f"  [{d}]")
            print(f"    Q: {ex['question']}")
            print(f"    SQL: {ex['query'][:80]}{'…' if len(ex['query'])>80 else ''}")
            print()
        if len(seen) == len(DIFFICULTY_LEVELS):
            break


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Spider data loader")
    parser.add_argument("--split",  default="train", choices=["train", "dev"],
                        help="Which Spider split to load")
    parser.add_argument("--subset", type=int, default=None,
                        help="Only return the first N examples (quick local test)")
    args = parser.parse_args()

    examples = load_spider(split=args.split, subset=args.subset)
    print_summary(examples, split=args.split)


if __name__ == "__main__":
    main()
