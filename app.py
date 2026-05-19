"""
app.py — FastAPI Backend + Streamlit Frontend (Local Demo)
===========================================================
Two modes in one file:
  - `python app.py api`          → starts FastAPI on port 8000
  - `streamlit run app.py`       → starts Streamlit UI on port 8501

FastAPI endpoint:
  POST /query  {"question": str, "schema": dict}  → {"sql": str, ...}

Streamlit UI:
  - Text input for natural-language question
  - "Demo Mode" toggle (fast mock) vs "Real Model" (loads CodeLlama)
  - Shows: filtered schema, predicted SQL, complexity, execution status
  - Degrades gracefully when no model is loaded

Usage:
    # Backend only
    python app.py api --port 8000

    # Frontend (auto-starts backend in same process)
    streamlit run app.py -- --port 8501
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Shared — schema linker + inference helpers (used by both API and UI)
# ──────────────────────────────────────────────────────────────────────────────

from utils.schema_linker import HeuristicSchemaLinker

_linker = HeuristicSchemaLinker()

_MOCK_SQL_BY_KEYWORD: Dict[str, str] = {
    "how many":  "SELECT COUNT(*) FROM {table} ;",
    "list":      "SELECT * FROM {table} ;",
    "average":   "SELECT AVG({col}) FROM {table} ;",
    "maximum":   "SELECT MAX({col}) FROM {table} ;",
    "minimum":   "SELECT MIN({col}) FROM {table} ;",
    "who":       "SELECT name FROM {table} ;",
    "where":     "SELECT * FROM {table} WHERE {col} = 'value' ;",
}
_MOCK_SQL_DEFAULT = "SELECT * FROM {table} LIMIT 10 ;"


def _mock_sql(question: str, schema_str: str) -> str:
    """Generate a plausible-looking mock SQL from the question keywords."""
    q_low  = question.lower()
    table  = re.search(r"(\w+)\(", schema_str)
    table  = table.group(1) if table else "table"
    col    = re.search(r"\((\w+)", schema_str)
    col    = col.group(1) if col else "id"

    for kw, tmpl in _MOCK_SQL_BY_KEYWORD.items():
        if kw in q_low:
            return tmpl.format(table=table, col=col)
    return _MOCK_SQL_DEFAULT.format(table=table)


def _estimate_complexity(sql: str) -> str:
    up = sql.upper()
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


def _parse_plain_schema(text: str) -> Dict:
    """Parse plain-text or JSON schema into the linker's normalised dict format."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    tables = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("DB:"):
            continue
        if ":" in line:
            tname, _, col_str = line.partition(":")
            columns = []
            for part in col_str.split(","):
                part = part.strip()
                m = re.match(r"(\w+)\((\w+)\)", part)
                if m:
                    columns.append({"name": m.group(1), "type": m.group(2)})
                elif part:
                    columns.append({"name": part, "type": ""})
            tables.append({"name": tname.strip(), "columns": columns})
    return {"db_id": "", "tables": tables}


# ──────────────────────────────────────────────────────────────────────────────
# Global model state (loaded lazily on first real-model request)
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_STATE: Dict[str, Any] = {"model": None, "tokenizer": None, "loaded": False}


def _ensure_model_loaded(model_id: str = "codellama/CodeLlama-7b-hf") -> bool:
    if _MODEL_STATE["loaded"]:
        return True
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)

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
        _MODEL_STATE.update({"model": model, "tokenizer": tok, "loaded": True})
        return True
    except Exception as exc:
        logger.warning("Model load failed: %s", exc)
        return False


def _real_inference(prompt: str) -> str:
    import torch
    model, tok = _MODEL_STATE["model"], _MODEL_STATE["tokenizer"]
    inputs = tok(prompt, return_tensors="pt",
                 truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new = out[0, inputs["input_ids"].shape[1]:]
    sql = tok.decode(new, skip_special_tokens=True).strip()
    if ";" in sql:
        sql = sql[: sql.index(";") + 1]
    return sql


def generate_sql(
    question:  str,
    schema:    Any,
    demo_mode: bool = True,
) -> Dict[str, Any]:
    """
    Core inference function shared by both FastAPI and Streamlit.

    Parameters
    ----------
    question  : natural-language question
    schema    : dict (normalised) or str (plain-text / JSON)
    demo_mode : True → fast mock, False → real model

    Returns
    -------
    dict with keys: sql, filtered_schema, complexity, mode, error
    """
    if isinstance(schema, str):
        schema_dict = _parse_plain_schema(schema)
    else:
        schema_dict = schema or {}

    filtered = _linker.filter_schema(schema_dict, question)

    if demo_mode:
        sql   = _mock_sql(question, filtered)
        mode  = "demo"
        error = None
    else:
        loaded = _ensure_model_loaded()
        if not loaded:
            sql   = _mock_sql(question, filtered)
            mode  = "demo_fallback"
            error = "Model failed to load — showing mock response"
        else:
            from config import Config
            cfg    = Config()
            prompt = cfg.prompt_template.format(schema=filtered, question=question)
            sql    = _real_inference(prompt)
            mode   = "real_model"
            error  = None

    return {
        "sql":             sql,
        "filtered_schema": filtered,
        "complexity":      _estimate_complexity(sql),
        "mode":            mode,
        "error":           error,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI backend
# ──────────────────────────────────────────────────────────────────────────────

def build_api():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        question:  str
        schema:    Any
        demo_mode: bool = True

    class QueryResponse(BaseModel):
        sql:             str
        filtered_schema: str
        complexity:      str
        mode:            str
        error:           Optional[str] = None

    api = FastAPI(
        title="SQL Query Generator",
        description="Text-to-SQL with CodeLlama-7B + Schema Linking",
        version="1.0.0",
    )

    @api.get("/health")
    def health():
        return {"status": "ok", "model_loaded": _MODEL_STATE["loaded"]}

    @api.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        try:
            result = generate_sql(req.question, req.schema, demo_mode=req.demo_mode)
            return QueryResponse(**result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return api


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit frontend
# Streamlit re-imports this module — guard heavy code behind __name__ checks
# ──────────────────────────────────────────────────────────────────────────────

def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="SQL Query Generator",
        page_icon="🔍",
        layout="wide",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔍 LLM-Powered SQL Query Generator")
    st.caption("CodeLlama-7B + QLoRA · Spider + WikiSQL · Schema Linking")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        demo_mode = st.toggle("Demo Mode (fast mock)", value=True)
        if not demo_mode:
            st.warning(
                "Real model mode downloads CodeLlama-7B (~13 GB). "
                "May be slow without a GPU."
            )
        st.markdown("---")
        st.markdown("**Enhancements**")
        st.markdown("✅ Multi-dataset training")
        st.markdown("✅ Schema linking")
        st.markdown("✅ Zero-shot baseline")
        st.markdown("✅ Error analysis")

    # ── Main inputs ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        question = st.text_input(
            "Natural Language Question",
            placeholder="How many singers are there?",
        )

        schema_input = st.text_area(
            "Database Schema (plain text or JSON)",
            height=180,
            placeholder=(
                "singer: singer_id, name, country, age\n"
                "concert: concert_id, concert_name, theme, stadium_id\n"
                "stadium: stadium_id, location, name, capacity"
            ),
        )

        run_btn = st.button("🚀 Generate SQL", type="primary", use_container_width=True)

    # ── Output ────────────────────────────────────────────────────────────────
    with col2:
        st.subheader("Output")

        if run_btn and question and schema_input:
            with st.spinner("Running pipeline …"):
                result = generate_sql(question, schema_input, demo_mode=demo_mode)

            if result.get("error"):
                st.warning(f"⚠️ {result['error']}")

            mode_label = {
                "demo":          "🟡 Demo Mode",
                "demo_fallback": "🟠 Demo Fallback",
                "real_model":    "🟢 Real Model",
            }.get(result["mode"], result["mode"])

            st.markdown(f"**Mode:** {mode_label}")

            st.markdown("**Filtered Schema** *(after schema linking)*")
            st.code(result["filtered_schema"], language="sql")

            st.markdown("**Generated SQL**")
            st.code(result["sql"], language="sql")

            complexity = result["complexity"]
            colour = {"easy": "green", "medium": "blue",
                      "hard": "orange", "extra hard": "red"}.get(complexity, "gray")
            st.markdown(
                f"**Complexity:** :{colour}[{complexity.upper()}]"
            )

            # Execution status (mock — no live DB)
            st.markdown("**Execution Status**")
            st.info("ℹ️ Live execution requires Spider SQLite databases. "
                    "Download from the Spider project page and set "
                    "`spider_db_path` in config.py.")

        elif run_btn:
            st.error("Please enter both a question and a schema.")
        else:
            st.info("Enter a question and schema, then click **Generate SQL**.")

    # ── Example gallery ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Example Queries")

    examples = [
        ("How many singers are there?",
         "singer: singer_id, name, country, age", "easy"),
        ("List all concerts with theme 'Pop'",
         "concert: concert_id, concert_name, theme, stadium_id", "medium"),
        ("Which singers performed in more than one concert?",
         "singer: singer_id, name\nsinger_in_concert: concert_id, singer_id", "hard"),
    ]

    cols = st.columns(len(examples))
    for col, (q, s, diff) in zip(cols, examples):
        with col:
            colour = {"easy": "green", "medium": "blue", "hard": "orange"}.get(diff, "gray")
            st.markdown(f"**:{colour}[{diff.upper()}]**")
            st.markdown(f"*{q}*")
            if st.button(f"Try →", key=q):
                result = generate_sql(q, s, demo_mode=True)
                st.code(result["sql"], language="sql")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — detect execution context
# ──────────────────────────────────────────────────────────────────────────────

def _is_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    # Streamlit called this module — run the UI
    run_streamlit()

elif __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SQL Generator API / UI")
    sub    = parser.add_subparsers(dest="command")

    api_p  = sub.add_parser("api",  help="Start FastAPI backend")
    api_p.add_argument("--port", type=int, default=8000)
    api_p.add_argument("--host", default="0.0.0.0")

    args = parser.parse_args()

    if args.command == "api" or args.command is None:
        port = getattr(args, "port", 8000)
        host = getattr(args, "host", "0.0.0.0")
        print(f"\n🚀 Starting FastAPI server on http://{host}:{port}")
        print("   POST /query  →  {question, schema, demo_mode}")
        print("   GET  /health →  status check\n")
        app = build_api()
        uvicorn.run(app, host=host, port=port, reload=False)
