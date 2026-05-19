"""
app.py — FastAPI Backend + HTML/JS Frontend
============================================
Single command starts everything:
    python app.py

Then open: http://localhost:8000

Endpoints:
  GET  /           → Web UI (HTML page)
  GET  /health     → {"status": "ok", "model_loaded": bool, "model_mode": str}
  GET  /eval       → eval results from demo_output.json
  POST /query      → {question, schema, demo_mode} → {sql, filtered_schema, complexity, mode, error}
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_LORA_WEIGHTS_DIR = "./results/lora_weights"

# ──────────────────────────────────────────────────────────────────────────────
# Lazy schema linker (no NLTK at import time)
# ──────────────────────────────────────────────────────────────────────────────

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
            columns = [{"name": p.strip(), "type": ""} for p in col_str.split(",") if p.strip()]
            tables.append({"name": tname.strip(), "columns": columns})
        elif "(" in line:
            m = re.match(r"(\w+)\(([^)]*)\)", line)
            if m:
                cols = [{"name": c.strip(), "type": ""} for c in m.group(2).split(",") if c.strip()]
                tables.append({"name": m.group(1), "columns": cols})
    return {"db_id": "", "tables": tables}


# ──────────────────────────────────────────────────────────────────────────────
# Lazy model state
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_STATE: Dict[str, Any] = {
    "model": None, "tokenizer": None,
    "loaded": False, "model_mode": "not_loaded",
}


def _ensure_model_loaded(base_model_id: str = "codellama/CodeLlama-7b-hf") -> bool:
    if _MODEL_STATE["loaded"]:
        return True
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"

        if torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
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
        if Path(_LORA_WEIGHTS_DIR).exists():
            try:
                from peft import PeftModel
                model      = PeftModel.from_pretrained(model, _LORA_WEIGHTS_DIR)
                model_mode = "finetuned"
                logger.info("Loaded fine-tuned LoRA from %s", _LORA_WEIGHTS_DIR)
            except Exception as exc:
                logger.warning("LoRA load failed (%s) — using base model.", exc)

        model.eval()
        _MODEL_STATE.update({"model": model, "tokenizer": tok,
                              "loaded": True, "model_mode": model_mode})
        return True
    except Exception as exc:
        logger.warning("Model load failed: %s", exc)
        return False


def _real_inference(prompt: str) -> str:
    import torch
    model, tok = _MODEL_STATE["model"], _MODEL_STATE["tokenizer"]
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
    new = out[0, inputs["input_ids"].shape[1]:]
    sql = tok.decode(new, skip_special_tokens=True).strip()
    if ";" in sql:
        sql = sql[: sql.index(";") + 1]
    return sql


def generate_sql(question: str, schema: Any, demo_mode: bool = True) -> Dict[str, Any]:
    schema_dict = _parse_plain_schema(schema) if isinstance(schema, str) else (schema or {})
    filtered    = _get_linker().filter_schema(schema_dict, question)

    if demo_mode:
        sql, mode, error = _mock_sql(question, filtered), "demo", None
    else:
        if not _ensure_model_loaded():
            sql   = _mock_sql(question, filtered)
            mode  = "demo_fallback"
            error = "Model failed to load — showing mock response"
        else:
            from config import Config
            prompt = Config().prompt_template.format(schema=filtered, question=question)
            try:
                sql   = _real_inference(prompt)
                mode  = _MODEL_STATE["model_mode"]
                error = None
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    sql, mode, error = (_mock_sql(question, filtered),
                                        "demo_fallback", "VRAM OOM — showing mock")
                else:
                    raise

    return {
        "sql": sql, "filtered_schema": filtered,
        "complexity": _estimate_complexity(sql), "mode": mode, "error": error,
    }


# ──────────────────────────────────────────────────────────────────────────────
# HTML page (embedded — no template folder needed)
# ──────────────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SQL Query Generator</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" rel="stylesheet">
  <style>
    body { background: #f8f9fa; }
    .hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
            color: #fff; padding: 2rem 0 1.5rem; margin-bottom: 2rem; }
    .hero h1 { font-size: 1.8rem; font-weight: 700; }
    .hero p  { color: #a0aec0; margin: 0; font-size: 0.9rem; }
    .badge-easy       { background: #198754; }
    .badge-medium     { background: #0d6efd; }
    .badge-hard       { background: #fd7e14; }
    .badge-extra-hard { background: #dc3545; }
    .badge-demo       { background: #6c757d; }
    .badge-base       { background: #0dcaf0; color: #000; }
    .badge-finetuned  { background: #198754; }
    .badge-fallback   { background: #fd7e14; }
    .card { border: none; box-shadow: 0 1px 4px rgba(0,0,0,.08); border-radius: .75rem; }
    .card-header { background: #fff; border-bottom: 1px solid #eee;
                   font-weight: 600; border-radius: .75rem .75rem 0 0 !important; }
    pre code.hljs { border-radius: .5rem; font-size: 0.85rem; padding: 1rem; }
    .example-card { cursor: pointer; transition: box-shadow .15s; }
    .example-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.12); }
    #spinner { display: none; }
    .table th { font-size: .8rem; color: #6c757d; border-top: none; }
    .match-yes { color: #198754; font-weight: 600; }
    .match-no  { color: #dc3545; }
  </style>
</head>
<body>

<div class="hero">
  <div class="container">
    <h1>&#128269; LLM-Powered SQL Query Generator</h1>
    <p>CodeLlama-7B &middot; QLoRA NF4 &middot; Spider + WikiSQL (~87K examples) &middot; Schema Linking &middot; RTX 3050</p>
  </div>
</div>

<div class="container pb-5">

  <div class="mb-4 d-flex gap-2 flex-wrap">
    <span class="badge bg-success">Enhancement 1 &mdash; Spider + WikiSQL</span>
    <span class="badge bg-primary">Enhancement 2 &mdash; Schema Linking</span>
    <span class="badge bg-warning text-dark">Enhancement 3 &mdash; Zero-Shot Baseline</span>
    <span class="badge bg-danger">Enhancement 4 &mdash; Error Analysis</span>
  </div>

  <div class="row g-4">

    <div class="col-lg-5">
      <div class="card h-100">
        <div class="card-header">&#9997;&#65039; Input</div>
        <div class="card-body d-flex flex-column gap-3">

          <div>
            <label class="form-label fw-semibold">Natural Language Question</label>
            <input id="question" type="text" class="form-control"
                   placeholder="How many singers are there?" />
          </div>

          <div>
            <label class="form-label fw-semibold">Database Schema
              <small class="text-muted fw-normal">(plain text or JSON)</small>
            </label>
            <textarea id="schema" class="form-control font-monospace" rows="6"
                      placeholder="singer: singer_id, name, country, age&#10;concert: concert_id, concert_name, theme, stadium_id&#10;stadium: stadium_id, location, name, capacity"></textarea>
          </div>

          <div class="form-check form-switch">
            <input class="form-check-input" type="checkbox" id="demoMode" checked>
            <label class="form-check-label" for="demoMode">
              Demo Mode <small class="text-muted">(instant mock &mdash; no GPU needed)</small>
            </label>
          </div>

          <button onclick="generateSQL()" class="btn btn-primary mt-auto">
            &#128640; Generate SQL
          </button>
          <div id="spinner" class="text-center text-muted small">Generating&hellip;</div>
        </div>
      </div>
    </div>

    <div class="col-lg-7">
      <div class="card h-100" style="min-height:300px">
        <div class="card-header d-flex justify-content-between align-items-center">
          <span>&#128200; Output</span>
          <span id="modeBadge" class="badge bg-secondary">waiting</span>
        </div>
        <div class="card-body" id="outputBody">
          <p class="text-muted mt-3">Enter a question and schema, then click <strong>Generate SQL</strong>.</p>
        </div>
      </div>
    </div>
  </div>

  <h5 class="mt-5 mb-3">&#128218; Example Queries</h5>
  <div class="row g-3" id="examples"></div>

  <h5 class="mt-5 mb-3">&#128202; Latest Eval Results <small class="text-muted fs-6 fw-normal">from demo_output.json</small></h5>
  <div class="card">
    <div class="card-body" id="evalBody">
      <p class="text-muted">Loading&hellip;</p>
    </div>
  </div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/sql.min.js"></script>
<script>
const EXAMPLES = [
  { q: "How many singers are there?",
    s: "singer: singer_id, name, country, age",
    diff: "easy" },
  { q: "Show all countries and the number of singers in each country.",
    s: "singer: singer_id, name, country, age\\nconcert: concert_id, concert_name, theme",
    diff: "medium" },
  { q: "Which singers performed in more than one concert?",
    s: "singer: singer_id, name\\nsinger_in_concert: concert_id, singer_id",
    diff: "hard" },
  { q: "Show names for all stadiums except those with a concert in 2014.",
    s: "stadium: stadium_id, name\\nconcert: concert_id, stadium_id, year",
    diff: "extra hard" },
];

const DIFF_COLOUR = { easy:"success", medium:"primary", hard:"warning", "extra hard":"danger" };
const MODE_COLOUR = { demo:"secondary", demo_fallback:"warning", base:"info", finetuned:"success", zero_shot:"info", mock:"secondary" };
const MODE_LABEL  = { demo:"Demo (mock)", demo_fallback:"Demo Fallback", base:"Zero-Shot", finetuned:"Fine-Tuned (LoRA)", mock:"Mock" };

(function buildExamples() {
  const el = document.getElementById("examples");
  EXAMPLES.forEach(ex => {
    const col = document.createElement("div");
    col.className = "col-sm-6 col-xl-3";
    col.innerHTML =
      "<div class='card example-card h-100' onclick='useExample(" + JSON.stringify(ex.q) + "," + JSON.stringify(ex.s) + ")'>" +
      "<div class='card-body'>" +
      "<span class='badge badge-" + ex.diff.replace(" ","-") + " mb-2'>" + ex.diff.toUpperCase() + "</span>" +
      "<p class='small mb-0'>" + ex.q + "</p>" +
      "<p class='text-muted' style='font-size:.75rem;margin-top:.4rem'>" + ex.s.split("\\n")[0] + "&hellip;</p>" +
      "</div></div>";
    el.appendChild(col);
  });
})();

function useExample(q, s) {
  document.getElementById("question").value = q;
  document.getElementById("schema").value   = s;
  generateSQL();
}

async function generateSQL() {
  const question = document.getElementById("question").value.trim();
  const schema   = document.getElementById("schema").value.trim();
  const demoMode = document.getElementById("demoMode").checked;

  if (!question || !schema) {
    alert("Please fill in both the question and schema fields.");
    return;
  }

  document.getElementById("spinner").style.display = "block";
  document.getElementById("outputBody").innerHTML = "";

  try {
    const resp = await fetch("/query", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({question, schema, demo_mode: demoMode}),
    });
    const data = await resp.json();
    renderOutput(data);
  } catch(e) {
    document.getElementById("outputBody").innerHTML =
      "<div class='alert alert-danger'>Request failed: " + e.message + "</div>";
  } finally {
    document.getElementById("spinner").style.display = "none";
  }
}

function renderOutput(data) {
  const modeKey   = data.mode || "demo";
  const diffKey   = (data.complexity || "easy").replace(" ", "-");
  const modeCol   = MODE_COLOUR[modeKey] || "secondary";
  const modeLabel = MODE_LABEL[modeKey]  || modeKey;

  document.getElementById("modeBadge").className   = "badge bg-" + modeCol;
  document.getElementById("modeBadge").textContent = modeLabel;

  const errorHtml = data.error
    ? "<div class='alert alert-warning py-2 small'>&#9888;&#65039; " + data.error + "</div>" : "";

  document.getElementById("outputBody").innerHTML =
    errorHtml +
    "<div class='mb-3'>" +
    "<div class='d-flex align-items-center gap-2 mb-1'>" +
    "<strong>Filtered Schema</strong>" +
    "<small class='text-muted'>(Enhancement 2 &mdash; schema linking)</small>" +
    "</div>" +
    "<code class='d-block p-2 bg-light rounded small text-break'>" + escHtml(data.filtered_schema) + "</code>" +
    "</div>" +
    "<div class='mb-3'>" +
    "<strong>Generated SQL</strong>" +
    "<pre class='mt-1 mb-0'><code class='language-sql'>" + escHtml(data.sql) + "</code></pre>" +
    "</div>" +
    "<div class='d-flex gap-2 align-items-center'>" +
    "<strong>Complexity:</strong>" +
    "<span class='badge badge-" + diffKey + "'>" + (data.complexity||"easy").toUpperCase() + "</span>" +
    "</div>";

  document.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
}

function escHtml(str) {
  return (str || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

async function loadEval() {
  try {
    const resp = await fetch("/eval");
    if (!resp.ok) {
      document.getElementById("evalBody").innerHTML = "<p class='text-muted'>No eval results yet. Run <code>python eval.py --mock</code> to generate them.</p>";
      return;
    }
    const data    = await resp.json();
    const records = data.results || [];
    if (!records.length) { document.getElementById("evalBody").innerHTML = "<p class='text-muted'>No records.</p>"; return; }

    const total   = records.length;
    const correct = records.filter(r => r.match).length;
    const pct     = total ? Math.round(correct / total * 100) : 0;

    const rows = records.map((r,i) =>
      "<tr><td>" + (i+1) + "</td>" +
      "<td><span class='badge badge-" + (r.complexity||"easy").replace(" ","-") + "'>" + (r.complexity||"") + "</span></td>" +
      "<td class='" + (r.match ? "match-yes" : "match-no") + "'>" + (r.match ? "&#10003; Match" : "&#10007; No match") + "</td>" +
      "<td class='small'>" + escHtml((r.question||"").substring(0,55)) + "</td>" +
      "<td class='small font-monospace'>" + escHtml((r.gold_sql||"").substring(0,50)) + "</td></tr>"
    ).join("");

    document.getElementById("evalBody").innerHTML =
      "<div class='mb-3 d-flex gap-3 align-items-center'>" +
      "<span class='fs-5 fw-bold'>" + correct + "/" + total + "</span>" +
      "<span class='text-muted'>exact match</span>" +
      "<span class='badge bg-" + (pct>=50?"success":pct>=25?"warning":"danger") + " fs-6'>" + pct + "% EM</span>" +
      "</div>" +
      "<div class='table-responsive'>" +
      "<table class='table table-hover table-sm align-middle'>" +
      "<thead><tr><th>#</th><th>Complexity</th><th>Result</th><th>Question</th><th>Gold SQL</th></tr></thead>" +
      "<tbody>" + rows + "</tbody></table></div>";
  } catch(e) {
    document.getElementById("evalBody").innerHTML = "<p class='text-muted'>Run <code>python eval.py --mock</code> to generate eval results.</p>";
  }
}

loadEval();

document.getElementById("question").addEventListener("keydown", e => {
  if (e.key === "Enter") generateSQL();
});
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

def build_api():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    class QueryResponse(BaseModel):
        sql:             str
        filtered_schema: str
        complexity:      str
        mode:            str
        error:           Optional[str] = None

    api = FastAPI(
        title="SQL Query Generator",
        description="Text-to-SQL · CodeLlama-7B · QLoRA · Spider + WikiSQL",
        version="2.0.0",
    )

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return _HTML

    @api.get("/health")
    def health():
        return {
            "status":       "ok",
            "model_loaded": _MODEL_STATE["loaded"],
            "model_mode":   _MODEL_STATE["model_mode"],
        }

    @api.get("/eval")
    def eval_results():
        path = Path("demo_output.json")
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo_output.json not found")
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        total   = len(records)
        correct = sum(1 for r in records if r.get("match"))
        return {
            "total":   total,
            "correct": correct,
            "em_pct":  round(correct / total * 100, 1) if total else 0.0,
            "results": records,
        }

    @api.post("/query", response_model=QueryResponse)
    def query(body: Dict[str, Any] = Body(...)):
        try:
            result = generate_sql(
                question  = body.get("question", ""),
                schema    = body.get("schema"),
                demo_mode = body.get("demo_mode", True),
            )
            return QueryResponse(**result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return api


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SQL Generator — FastAPI web app")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  SQL Query Generator")
    print(f"  Open → http://{args.host}:{args.port}")
    print(f"  API  → http://{args.host}:{args.port}/docs")
    print(f"  Press Ctrl+C to stop\n")

    app = build_api()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
