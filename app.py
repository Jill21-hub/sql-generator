"""
app.py — FastAPI Inference Server
===================================
Serves the fine-tuned SQL generation model over HTTP.

Startup:
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /generate_sql   — main inference endpoint
    GET  /health         — liveness probe
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import Config
from utils.schema_linker import HeuristicSchemaLinker

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class SQLRequest(BaseModel):
    schema: str = Field(..., description="Database schema as plain text or JSON string")
    question: str = Field(..., description="Natural language question to convert to SQL")
    use_schema_linking: bool = Field(
        True,
        description="Apply heuristic schema linking before inference (recommended)",
    )


class SQLResponse(BaseModel):
    sql: str
    filtered_schema: str
    model_mode: str      # "fine_tuned" or "zero_shot"


# ──────────────────────────────────────────────────────────────────────────────
# Global model state (populated at startup)
# ──────────────────────────────────────────────────────────────────────────────

_state: dict = {}


def _load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def _load_model(cfg: Config) -> tuple:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=cfg.quantization.bnb_4bit_compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = True

    lora_dir = cfg.training.lora_weights_dir
    if os.path.isdir(lora_dir):
        logger.info("Loading LoRA adapters from %s", lora_dir)
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
        mode = "fine_tuned"
    else:
        logger.warning(
            "LoRA weights not found at '%s'. Running in zero-shot mode.", lora_dir
        )
        mode = "zero_shot"

    model.eval()
    return model, mode


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan — load model once at startup, free on shutdown
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config()
    logger.info("Loading tokenizer and model …")
    tokenizer = _load_tokenizer(cfg.model_id)
    model, mode = _load_model(cfg)
    torch.cuda.empty_cache()

    _state["cfg"] = cfg
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["mode"] = mode
    _state["schema_linker"] = HeuristicSchemaLinker()

    logger.info("Model ready — mode=%s", mode)
    yield

    # Cleanup on shutdown
    del _state["model"]
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SQL Query Generator",
    description="Fine-tuned CodeLlama-7B that converts natural language to SQL.",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

def _infer(prompt: str) -> str:
    cfg: Config = _state["cfg"]
    model = _state["model"]
    tokenizer: AutoTokenizer = _state["tokenizer"]
    ic = cfg.inference

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.training.max_seq_length,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=ic.max_new_tokens,
            do_sample=ic.do_sample,
            temperature=ic.temperature,
            repetition_penalty=ic.repetition_penalty,
            num_beams=ic.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clip at the first semicolon
    if ";" in decoded:
        decoded = decoded[: decoded.index(";") + 1]

    return decoded


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "mode": _state.get("mode", "not_loaded")}


@app.post("/generate_sql", response_model=SQLResponse)
def generate_sql(request: SQLRequest):
    if not _state.get("model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    cfg: Config = _state["cfg"]
    schema_linker: HeuristicSchemaLinker = _state["schema_linker"]

    # Schema linking: if the schema is plain text we pass it through a minimal
    # wrapper so HeuristicSchemaLinker can tokenise against it.  If it's a
    # structured JSON-like dict the caller can send it via a richer endpoint.
    raw_schema = request.schema

    if request.use_schema_linking:
        # Wrap plain-text schema into the minimal dict format for the linker.
        # Lines like "table: col1, col2" are parsed into tables/columns.
        schema_dict = _parse_plain_schema(raw_schema)
        filtered_schema = schema_linker.filter_schema(schema_dict, request.question)
    else:
        filtered_schema = raw_schema

    prompt = cfg.prompt_template.format(
        schema=filtered_schema,
        question=request.question,
    )

    try:
        sql = _infer(prompt)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))

    return SQLResponse(
        sql=sql,
        filtered_schema=filtered_schema,
        model_mode=_state["mode"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Utility: parse plain-text schema into linker format
# ──────────────────────────────────────────────────────────────────────────────

def _parse_plain_schema(text: str) -> dict:
    """
    Convert a plain-text schema like:
        stadium: stadium_id(number), location(text), name(text)
        singer:  singer_id(number), name(text), country(text)
    into the normalised dict that HeuristicSchemaLinker expects.

    Also handles JSON strings by importing json.
    """
    import json, re

    # Try JSON first
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Plain-text line-by-line parse
    tables = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("DB:"):
            continue
        if ":" in line:
            tname, _, col_str = line.partition(":")
            columns = []
            for col_part in col_str.split(","):
                col_part = col_part.strip()
                m = re.match(r"(\w+)\((\w+)\)", col_part)
                if m:
                    columns.append({"name": m.group(1), "type": m.group(2)})
                elif col_part:
                    columns.append({"name": col_part, "type": ""})
            tables.append({"name": tname.strip(), "columns": columns})

    return {"db_id": "", "tables": tables}


# ──────────────────────────────────────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
