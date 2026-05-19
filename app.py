"""
app.py — FastAPI Backend + Streamlit Frontend
==============================================
Two modes in one file:
  python app.py api          → FastAPI on port 8000
  streamlit run app.py       → Streamlit UI on port 8501

FastAPI endpoints:
  POST /query   {"question": str, "schema": any, "demo_mode": bool}
  GET  /health  → {"status": "ok", "model_loaded": bool, "model_mode": str}

Streamlit UI:
  - Schema input (plain text or JSON)
  - Demo Mode toggle (instant mock) vs Real Model (CodeLlama + LoRA)
  - Displays filtered schema, predicted SQL, complexity badge
  - Graceful VRAM OOM fallback to demo mode

Model loading order:
  1. If ./results/lora_weights exists → load fine-tuned (LoRA) model
  2. Else → load base CodeLlama in 4-bit NF4
  3. If model load fails (OOM / no GPU) → fall back to mock mode silently

Usage:
    python app.py api --port 8000
    streamlit run app.py
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

# Lazy singleton — imported and instantiated on first query, not at startup.
# This avoids blocking NLTK corpus load during Streamlit's module import phase.
_LORA_WEIGHTS_DIR = "./results/lora_weights"
_linker_instance = None


def _get_linker():
    global _linker_instance
    if _linker_instance is None:
        from utils.schema_linker import HeuristicSchemaLinker
        _linker_instance = HeuristicSchemaLinker()
    return _linker_instance

# ──────────────────────────────────────────────────────────────────────────────
# Mock SQL helpers
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_SQL_BY_KEYWORD: Dict[str, str] = {
    "how many":  "SELECT COUNT(*) FROM {table} ;",
    "count":     "SELECT COUNT(*) FROM {table} ;",
    "list":      "SELECT * FROM {table} ;",
    "all":       "SELECT * FROM {table} LIMIT 10 ;",
    "average":   "SELECT AVG({col}) FROM {table} ;",
    "maximum":   "SELECT MAX({col}) FROM {table} ;",
    "minimum":   "SELECT MIN({col}) FROM {table} ;",
    "who":       "SELECT name FROM {table} ;",
    "where":     "SELECT * FROM {table} WHERE {col} = 'value' ;",
    "group":     "SELECT {col}, COUNT(*) FROM {table} GROUP BY {col} ;",
    "order":     "SELECT * FROM {table} ORDER BY {col} DESC ;",
}
_MOCK_SQL_DEFAULT = "SELECT * FROM {table} LIMIT 10 ;"


def _mock_sql(question: str, schema_str: str) -> str:
    q_low = question.lower()
    table = re.search(r"(\w+)\(", schema_str)
    table = table.group(1) if table else "table"
    col   = re.search(r"\((\w+)", schema_str)
    col   = col.group(1) if col else "id"
    for kw, tmpl in _MOCK_SQL_BY_KEYWORD.items():
        if kw in q_low:
            return tmpl.format(table=table, col=col)
    return _MOCK_SQL_DEFAULT.format(table=table)


def _estimate_complexity(sql: str) -> str:
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


def _parse_plain_schema(text: str) -> Dict:
    """Parse plain-text or JSON schema into the linker's normalised dict."""
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
        elif "(" in line:
            # format: table(col1, col2)
            m = re.match(r"(\w+)\(([^)]*)\)", line)
            if m:
                cols = [{"name": c.strip(), "type": ""} for c in m.group(2).split(",") if c.strip()]
                tables.append({"name": m.group(1), "columns": cols})

    return {"db_id": "", "tables": tables}


# ──────────────────────────────────────────────────────────────────────────────
# Global model state (loaded lazily on first real-model request)
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_STATE: Dict[str, Any] = {
    "model":      None,
    "tokenizer":  None,
    "loaded":     False,
    "model_mode": "not_loaded",  # "base" | "finetuned" | "not_loaded"
}


def _ensure_model_loaded(
    base_model_id: str = "codellama/CodeLlama-7b-hf",
    lora_path: str = _LORA_WEIGHTS_DIR,
) -> bool:
    """
    Lazy-load the best available model:
    1. LoRA fine-tuned if ./results/lora_weights exists
    2. Base model in 4-bit NF4
    3. CPU fp32 fallback
    Returns True on success, False on failure.
    """
    if _MODEL_STATE["loaded"]:
        return True
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)

        tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"

        use_gpu    = torch.cuda.is_available()
        lora_exists = Path(lora_path).exists()

        if use_gpu:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, quantization_config=bnb, device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch.float32,
                device_map="cpu", low_cpu_mem_usage=True,
            )

        model_mode = "base"

        if lora_exists:
            try:
                from peft import PeftModel
                model      = PeftModel.from_pretrained(model, lora_path)
                model_mode = "finetuned"
                logger.info("Loaded fine-tuned LoRA model from %s", lora_path)
            except Exception as exc:
                logger.warning("LoRA load failed (%s) — using base model.", exc)

        model.eval()
        _MODEL_STATE.update({
            "model":      model,
            "tokenizer":  tok,
            "loaded":     True,
            "model_mode": model_mode,
        })
        return True

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning(
                "OOM loading model on 6 GB GPU — falling back to demo mode. "
                "Try closing other GPU applications."
            )
        else:
            logger.warning("Model load failed: %s", exc)
        return False

    except Exception as exc:
        logger.warning("Model load failed: %s", exc)
        return False


def _real_inference(prompt: str) -> str:
    import torch
    model, tok = _MODEL_STATE["model"], _MODEL_STATE["tokenizer"]
    inputs = tok(
        prompt, return_tensors="pt",
        truncation=True, max_length=512,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
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
    Core inference function shared by FastAPI and Streamlit.

    Parameters
    ----------
    question  : Natural-language question.
    schema    : Dict (normalised) or str (plain-text / JSON).
    demo_mode : True → instant mock; False → real model (loads on first call).

    Returns
    -------
    dict: {sql, filtered_schema, complexity, mode, error}
    """
    if isinstance(schema, str):
        schema_dict = _parse_plain_schema(schema)
    else:
        schema_dict = schema or {}

    filtered = _get_linker().filter_schema(schema_dict, question)

    if demo_mode:
        sql   = _mock_sql(question, filtered)
        mode  = "demo"
        error = None
    else:
        loaded = _ensure_model_loaded()
        if not loaded:
            sql   = _mock_sql(question, filtered)
            mode  = "demo_fallback"
            error = "Model failed to load (OOM or no GPU) — showing mock response"
        else:
            from config import Config
            cfg    = Config()
            prompt = cfg.prompt_template.format(schema=filtered, question=question)
            try:
                sql   = _real_inference(prompt)
                mode  = _MODEL_STATE["model_mode"]
                error = None
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    sql   = _mock_sql(question, filtered)
                    mode  = "demo_fallback"
                    error = "VRAM OOM during inference — showing mock response. Restart app."
                else:
                    raise

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
        title="LLM-Powered SQL Query Generator",
        description=(
            "Text-to-SQL with CodeLlama-7B + QLoRA + Schema Linking.\n\n"
            "Datasets: Spider + WikiSQL (~87K examples)\n"
            "Enhancement 2: Heuristic schema linking\n"
            "Enhancement 3: Zero-shot vs fine-tuned comparison"
        ),
        version="2.0.0",
    )

    @api.get("/health")
    def health():
        return {
            "status":       "ok",
            "model_loaded": _MODEL_STATE["loaded"],
            "model_mode":   _MODEL_STATE["model_mode"],
        }

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
    st.caption(
        "CodeLlama-7B · QLoRA (NF4) · Spider + WikiSQL (~87K examples) · "
        "Schema Linking · RTX 3050 Optimised"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        demo_mode = st.toggle("Demo Mode (fast mock)", value=True)
        if not demo_mode:
            lora_exists = Path(_LORA_WEIGHTS_DIR).exists()
            if lora_exists:
                st.success("✅ Fine-tuned LoRA weights found")
            else:
                st.warning(
                    "LoRA weights not found at `./results/lora_weights`. "
                    "Base model will be loaded (zero-shot)."
                )
            st.info(
                "First inference will download CodeLlama-7B (~13 GB). "
                "On 6 GB GPU, 4-bit NF4 quantization is used (~3.5 GB)."
            )

        st.markdown("---")
        st.markdown("**Enhancement Coverage**")
        st.markdown("✅ Enhancement 1 — Spider + WikiSQL training")
        st.markdown("✅ Enhancement 2 — Schema linking")
        st.markdown("✅ Enhancement 3 — Zero-shot baseline")
        st.markdown("✅ Enhancement 4 — Error analysis by complexity")
        st.markdown("---")
        st.markdown("**Model Info**")
        if _MODEL_STATE["loaded"]:
            mode_icons = {"base": "🟡", "finetuned": "🟢"}
            icon = mode_icons.get(_MODEL_STATE["model_mode"], "⚪")
            st.markdown(f"{icon} Model: `{_MODEL_STATE['model_mode']}`")
        else:
            st.markdown("⚪ Model: not loaded")

    # ── Main layout ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        question = st.text_input(
            "Natural Language Question",
            placeholder="How many singers are there?",
        )

        schema_input = st.text_area(
            "Database Schema  (plain text or JSON)",
            height=160,
            placeholder=(
                "singer: singer_id, name, country, age\n"
                "concert: concert_id, concert_name, theme, stadium_id\n"
                "stadium: stadium_id, location, name, capacity"
            ),
        )

        run_btn = st.button("🚀 Generate SQL", type="primary", use_container_width=True)

    with col2:
        st.subheader("Output")

        if run_btn and question and schema_input:
            with st.spinner("Running pipeline …"):
                result = generate_sql(question, schema_input, demo_mode=demo_mode)

            if result.get("error"):
                st.warning(f"⚠️ {result['error']}")

            mode_labels = {
                "demo":          "🟡 Demo (mock)",
                "demo_fallback": "🟠 Demo Fallback (model failed)",
                "base":          "🔵 Zero-Shot (base model)",
                "finetuned":     "🟢 Fine-Tuned (LoRA)",
                "zero_shot":     "🔵 Zero-Shot (base model)",
            }
            mode_label = mode_labels.get(result["mode"], result["mode"])
            st.markdown(f"**Mode:** {mode_label}")
            st.markdown("---")

            st.markdown("**Filtered Schema** *(after schema linking — Enhancement 2)*")
            st.code(result["filtered_schema"], language="sql")

            st.markdown("**Generated SQL**")
            st.code(result["sql"], language="sql")

            complexity = result["complexity"]
            colour = {
                "easy":       "green",
                "medium":     "blue",
                "hard":       "orange",
                "extra hard": "red",
            }.get(complexity, "gray")
            st.markdown(f"**SQL Complexity:** :{colour}[{complexity.upper()}]")

            st.markdown("**Execution Status**")
            st.info(
                "Live execution requires Spider SQLite databases. "
                "Download and set `spider_db_path` in config.py to enable."
            )

        elif run_btn:
            st.error("Please enter both a question and a schema.")
        else:
            st.info("Enter a question and schema, then click **Generate SQL**.")

    # ── Example gallery ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Example Queries")

    examples = [
        (
            "How many singers are there?",
            "singer: singer_id, name, country, age",
            "easy",
        ),
        (
            "List all concerts with theme 'Pop'",
            "concert: concert_id, concert_name, theme, stadium_id",
            "medium",
        ),
        (
            "Which singers performed in more than one concert?",
            "singer: singer_id, name\nsinger_in_concert: concert_id, singer_id",
            "hard",
        ),
        (
            "Show names for all stadiums except those with a concert in 2014.",
            "stadium: stadium_id, name\nconcert: concert_id, stadium_id, year",
            "extra hard",
        ),
    ]

    cols = st.columns(len(examples))
    for col, (q, s, diff) in zip(cols, examples):
        with col:
            colour = {
                "easy": "green", "medium": "blue",
                "hard": "orange", "extra hard": "red",
            }.get(diff, "gray")
            st.markdown(f"**:{colour}[{diff.upper()}]**")
            st.markdown(f"*{q}*")
            if st.button("Try →", key=q):
                r = generate_sql(q, s, demo_mode=True)
                st.code(r["sql"], language="sql")
                st.caption(f"Filtered schema: `{r['filtered_schema']}`")

    # ── Eval results gallery ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Latest Eval Results")

    demo_json = Path("demo_output.json")
    if demo_json.exists():
        with open(demo_json, encoding="utf-8") as f:
            records = json.load(f)

        total   = len(records)
        correct = sum(1 for r in records if r.get("match"))
        em_pct  = correct / total * 100 if total else 0.0

        st.metric("Exact Match", f"{correct}/{total}", f"{em_pct:.0f}%")

        rows = []
        for r in records:
            rows.append({
                "Complexity": r.get("complexity", ""),
                "Match":      "✅" if r.get("match") else "❌",
                "Question":   r.get("question", "")[:60],
                "Gold SQL":   r.get("gold_sql", "")[:60],
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info(
            "No eval results yet. Run `python eval.py --mock` to generate "
            "`demo_output.json` and see results here."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _is_streamlit() -> bool:
    # Check sys.modules — Streamlit imports itself before running the user script,
    # so this is always True in Streamlit context and never requires importing Streamlit.
    return "streamlit" in sys.modules


if _is_streamlit():
    run_streamlit()

elif __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SQL Generator API / UI")
    sub    = parser.add_subparsers(dest="command")
    api_p  = sub.add_parser("api", help="Start FastAPI backend")
    api_p.add_argument("--port", type=int, default=8000)
    api_p.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    port = getattr(args, "port", 8000)
    host = getattr(args, "host", "0.0.0.0")
    print(f"\n🚀 Starting FastAPI server on http://{host}:{port}")
    print("   POST /query  →  {question, schema, demo_mode}")
    print("   GET  /health →  status + model info\n")
    app = build_api()
    uvicorn.run(app, host=host, port=port, reload=False)
