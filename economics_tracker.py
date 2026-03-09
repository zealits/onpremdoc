"""
Economics / Token Usage Tracker for Document Processing Pipeline

Records token usage and estimated cost at each pipeline step (upload, vectorization,
retrieval, LLM, page summary) for stakeholder reporting and cost visibility.
All data is saved under the economics/ folder for easy export and presentation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from config.inference_config import (
        get_provider_name,
        get_embedding_model_id,
        get_llm_model_id,
    )
except Exception:  # best-effort import
    def get_provider_name() -> str:  # type: ignore[no-redef]
        return "ollama"

    def get_embedding_model_id() -> str:  # type: ignore[no-redef]
        return ""

    def get_llm_model_id() -> str:  # type: ignore[no-redef]
        return ""


def _model_label(model: str) -> str:
    """Default model label from inference provider when not specified."""
    if model:
        return model
    try:
        return get_provider_name()
    except Exception:
        return "ollama"


logger = logging.getLogger(__name__)

# Folder where all economics data is stored (for stakeholder reports)
ECONOMICS_DIR = Path("economics")
# Optional: approximate cost per 1K tokens (USD) for cloud LLM comparison; Ollama local = $0
DEFAULT_INPUT_COST_PER_1K = 0.0
DEFAULT_OUTPUT_COST_PER_1K = 0.0
DEFAULT_EMBEDDING_COST_PER_1K = 0.0
# Chars per token for estimation when API doesn't return usage (~4 for English)
CHARS_PER_TOKEN_ESTIMATE = 4


def _active_models() -> Tuple[str, str, str]:
    """Return (provider, embedding_model, llm_model) from inference_config."""
    try:
        provider = get_provider_name()
        embed = get_embedding_model_id()
        llm = get_llm_model_id()
    except Exception:
        provider, embed, llm = "ollama", "", ""
    return provider, embed, llm


def estimate_cost_for_usage(
    input_tokens: int,
    output_tokens: int,
    embedding_tokens: int,
) -> Dict[str, Any]:
    """
    Estimate USD cost for a given token usage snapshot.

    Rules based on current project configuration:
    - Ollama + nomic-embed-text:v1.5 + llama3.1:8b => local => cost_display = "N.A"
    - OpenAI text-embedding-3-large: $0.13 per 1M tokens (embeddings)
    - OpenAI gpt-4o-mini: input $0.15, output $0.60 per 1M tokens
    - Other providers/models: cost_display = "N.A"
    """
    provider, embed_model, llm_model = _active_models()

    cost = 0.0
    components: Dict[str, float] = {}

    # Local Ollama – not billed
    if provider == "ollama":
        return {
            "provider": provider,
            "embedding_model": embed_model,
            "llm_model": llm_model,
            "cost_estimate_usd": None,
            "cost_display": "N.A",
            "components": {},
        }

    # OpenAI pricing for specific models
    if provider == "openai":
        # Embeddings: text-embedding-3-large
        if "text-embedding-3-large" in (embed_model or ""):
            emb_price_per_tok = 0.13 / 1_000_000.0
            emb_cost = embedding_tokens * emb_price_per_tok
            components["embeddings_usd"] = emb_cost
            cost += emb_cost

        # LLM: gpt-4o-mini
        if "gpt-4o-mini" in (llm_model or ""):
            in_price_per_tok = 0.15 / 1_000_000.0
            out_price_per_tok = 0.60 / 1_000_000.0
            in_cost = input_tokens * in_price_per_tok
            out_cost = output_tokens * out_price_per_tok
            components["llm_input_usd"] = in_cost
            components["llm_output_usd"] = out_cost
            cost += in_cost + out_cost

    # HuggingFace or unsupported OpenAI models → treat as N.A.
    if not components:
        return {
            "provider": provider,
            "embedding_model": embed_model,
            "llm_model": llm_model,
            "cost_estimate_usd": None,
            "cost_display": "N.A",
            "components": {},
        }

    return {
        "provider": provider,
        "embedding_model": embed_model,
        "llm_model": llm_model,
        "cost_estimate_usd": round(cost, 8),
        "cost_display": f"{cost:.6f}",
        "components": components,
    }


def _ensure_economics_dir() -> Path:
    """Ensure economics directory exists; return path."""
    ECONOMICS_DIR.mkdir(parents=True, exist_ok=True)
    readme = ECONOMICS_DIR / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Token usage and pipeline economics for stakeholder reporting.\n"
            "Files: usage_YYYY-MM-DD.jsonl (one JSON object per line, one line per pipeline step).\n"
            "Phases: upload, pdf_processing, vectorization, retrieval, page_summary.\n"
            "Fields per line: timestamp, step, phase, document_id, input_tokens, output_tokens, "
            "embedding_tokens, total_tokens, model.\n"
            "API: GET /economics/summary?date=YYYY-MM-DD for aggregated totals by phase and step.\n",
            encoding="utf-8",
        )
    return ECONOMICS_DIR


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length (when real usage not available)."""
    if not text:
        return 0
    return max(1, len(str(text).strip()) // CHARS_PER_TOKEN_ESTIMATE)


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def log_step(
    step_name: str,
    phase: str,
    document_id: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    model: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Log a single pipeline step to the economics folder.
    phase: one of "upload" | "pdf_processing" | "vectorization" | "retrieval" | "llm" | "page_summary"
    """
    root = _ensure_economics_dir()
    record = {
        "timestamp": _timestamp(),
        "step": step_name,
        "phase": phase,
        "document_id": document_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "embedding_tokens": embedding_tokens,
        "total_tokens": input_tokens + output_tokens + embedding_tokens,
        "model": _model_label(model),
        "cost_estimate_usd": None,
    }
    # Attach pricing based on active provider/models
    pricing = estimate_cost_for_usage(input_tokens, output_tokens, embedding_tokens)
    record["cost_estimate_usd"] = pricing.get("cost_estimate_usd")
    record["pricing"] = pricing
    if extra:
        record["extra"] = extra

    # Append to daily JSONL file for easy aggregation
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    jsonl_path = root / f"usage_{date_str}.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.debug("Economics log: %s -> %s", step_name, jsonl_path)
    return jsonl_path


def log_upload(document_id: str, file_size_bytes: int = 0, filename: str = "") -> Path:
    """Log PDF upload step (no tokens; for pipeline visibility)."""
    return log_step(
        step_name="pdf_upload",
        phase="upload",
        document_id=document_id,
        extra={"file_size_bytes": file_size_bytes, "filename": filename},
    )


def log_pdf_processing(document_id: str, total_pages: Optional[int] = None) -> Path:
    """Log PDF processing (Docling) step."""
    return log_step(
        step_name="pdf_processing",
        phase="pdf_processing",
        document_id=document_id,
        extra={"total_pages": total_pages} if total_pages is not None else None,
    )


def log_vectorization(
    document_id: str,
    embedding_tokens: int,
    llm_tokens: int,
    total_chunks: int,
    truncated_chunks: int = 0,
    model_embed: str = "nomic-embed-text",
    model_llm: str = "llama3.1:8b",
    total_pages: Optional[int] = None,
    total_words: Optional[int] = None,
    total_tokens: Optional[int] = None,
    duration_seconds: Optional[float] = None,
) -> Path:
    """Log vectorization step (chunk summaries + embeddings)."""
    extra: Dict[str, Any] = {
        "total_chunks": total_chunks,
        "truncated_chunks": truncated_chunks,
    }
    if total_pages is not None:
        extra["total_pages"] = total_pages
    if total_words is not None:
        extra["total_words"] = total_words
    if total_tokens is not None:
        extra["total_tokens"] = total_tokens
    if duration_seconds is not None:
        extra["duration_seconds"] = duration_seconds

    return log_step(
        step_name="vectorization",
        phase="vectorization",
        document_id=document_id,
        input_tokens=llm_tokens,  # LLM input for summaries
        output_tokens=0,          # Could split if we had output count
        embedding_tokens=embedding_tokens,
        model=f"{model_embed}+{model_llm}",
        extra=extra,
    )


def log_retrieval_step(
    document_id: str,
    step_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Log a single retrieval sub-step (e.g. query_classification, analyze_chunks, generate_answer)."""
    return log_step(
        step_name=step_name,
        phase="retrieval",
        document_id=document_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
        extra=extra,
    )


def log_query_usage(
    document_id: str,
    steps: List[Dict[str, Any]],
) -> Path:
    """
    Log all token usage for one query (classification, retrieval embeddings,
    analysis, second retrieval if any, answer generation).
    steps: list of {"step": str, "input_tokens": int, "output_tokens": int, "embedding_tokens": int, "extra": dict}
    """
    root = _ensure_economics_dir()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    jsonl_path = root / f"usage_{date_str}.jsonl"

    total_in = total_out = total_emb = 0
    for s in steps:
        total_in += s.get("input_tokens", 0)
        total_out += s.get("output_tokens", 0)
        total_emb += s.get("embedding_tokens", 0)
        record = {
            "timestamp": _timestamp(),
            "step": s.get("step", "unknown"),
            "phase": "retrieval",
            "document_id": document_id,
            "input_tokens": s.get("input_tokens", 0),
            "output_tokens": s.get("output_tokens", 0),
            "embedding_tokens": s.get("embedding_tokens", 0),
            "total_tokens": s.get("input_tokens", 0) + s.get("output_tokens", 0) + s.get("embedding_tokens", 0),
            "model": _model_label(""),
        }
        pricing = estimate_cost_for_usage(
            record["input_tokens"],
            record["output_tokens"],
            record["embedding_tokens"],
        )
        record["cost_estimate_usd"] = pricing.get("cost_estimate_usd")
        record["pricing"] = pricing
        if s.get("extra"):
            record["extra"] = s["extra"]
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # One summary line for this query
    summary_record = {
        "timestamp": _timestamp(),
        "step": "query_total",
        "phase": "retrieval",
        "document_id": document_id,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "embedding_tokens": total_emb,
        "total_tokens": total_in + total_out + total_emb,
        "model": _model_label(""),
        "extra": {"sub_steps": len(steps)},
    }
    summary_pricing = estimate_cost_for_usage(total_in, total_out, total_emb)
    summary_record["cost_estimate_usd"] = summary_pricing.get("cost_estimate_usd")
    summary_record["pricing"] = summary_pricing
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    return jsonl_path


def log_page_summary(
    document_id: str,
    page_number: int,
    input_tokens_estimate: int,
    output_tokens_estimate: int,
    chunks_used: int = 0,
) -> Path:
    """Log page summarization LLM usage."""
    return log_step(
        step_name="page_summary",
        phase="page_summary",
        document_id=document_id,
        input_tokens=input_tokens_estimate,
        output_tokens=output_tokens_estimate,
        extra={"page_number": page_number, "chunks_used": chunks_used},
    )


def get_usage_summary(date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Read usage from economics/usage_<date>.jsonl and return aggregated summary.
    If date_str is None, use today (UTC).
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    root = _ensure_economics_dir()
    jsonl_path = root / f"usage_{date_str}.jsonl"
    if not jsonl_path.exists():
        return {
            "date": date_str,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_embedding_tokens": 0,
            "total_tokens": 0,
            "by_phase": {},
            "by_step": {},
            "events": 0,
        }

    total_in = total_out = total_emb = 0
    by_phase: Dict[str, Dict[str, int]] = {}
    by_step: Dict[str, Dict[str, int]] = {}
    events = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            events += 1
            ti = rec.get("input_tokens", 0)
            to = rec.get("output_tokens", 0)
            te = rec.get("embedding_tokens", 0)
            total_in += ti
            total_out += to
            total_emb += te
            phase = rec.get("phase", "unknown")
            step = rec.get("step", "unknown")
            if phase not in by_phase:
                by_phase[phase] = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
            by_phase[phase]["input_tokens"] += ti
            by_phase[phase]["output_tokens"] += to
            by_phase[phase]["embedding_tokens"] += te
            if step not in by_step:
                by_step[step] = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
            by_step[step]["input_tokens"] += ti
            by_step[step]["output_tokens"] += to
            by_step[step]["embedding_tokens"] += te

    return {
        "date": date_str,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_embedding_tokens": total_emb,
        "total_tokens": total_in + total_out + total_emb,
        "by_phase": by_phase,
        "by_step": by_step,
        "events": events,
    }


def get_latest_vectorization_for_document(
    document_id: str,
    date_str: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return the most recent vectorization economics record for a specific document_id
    from economics/usage_<date>.jsonl.

    This is used by the API so the frontend can fetch vectorization stats for
    exactly one document, even when multiple documents exist for the same day.
    """
    # Kept for backwards compatibility; now implemented via get_document_pipeline_economics.
    data = get_document_pipeline_economics(document_id)
    # Find the last vectorization event for this document, if any.
    vector_events = [e for e in data.get("events", []) if e.get("step") == "vectorization"]
    return vector_events[-1] if vector_events else None


def get_document_pipeline_economics(document_id: str) -> Dict[str, Any]:
    """
    Aggregate pipeline economics for a single document across ALL days.

    Includes only:
    - pdf_upload
    - pdf_processing
    - vectorization
    and ignores retrieval / query steps entirely.
    """
    root = _ensure_economics_dir()
    if not root.exists():
        return {
            "document_id": document_id,
            "events": [],
            "totals": {
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_tokens": 0,
                "total_tokens": 0,
                "cost_estimate_usd": 0.0,
            },
        }

    wanted_steps = {"pdf_upload", "pdf_processing", "vectorization"}
    events: List[Dict[str, Any]] = []

    for jsonl_path in sorted(root.glob("usage_*.jsonl")):
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("document_id") != document_id:
                        continue
                    if rec.get("step") not in wanted_steps:
                        continue
                    events.append(rec)
        except Exception:
            continue

    # Sort events by timestamp so frontend can show a clean timeline
    def _ts(rec: Dict[str, Any]) -> str:
        return rec.get("timestamp", "")

    events.sort(key=_ts)

    # Aggregate totals across these three steps
    total_in = total_out = total_emb = 0
    total_cost = 0.0
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    for rec in events:
        ts = rec.get("timestamp")
        if isinstance(ts, str):
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts
        total_in += int(rec.get("input_tokens", 0) or 0)
        total_out += int(rec.get("output_tokens", 0) or 0)
        total_emb += int(rec.get("embedding_tokens", 0) or 0)
        c = rec.get("cost_estimate_usd")
        if isinstance(c, (int, float)):
            total_cost += float(c)

    # Approximate end-to-end pipeline duration from first to last event timestamp (ISO strings).
    pipeline_seconds: Optional[float] = None
    if first_ts and last_ts:
        try:
            start_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            pipeline_seconds = max(0.0, (end_dt - start_dt).total_seconds())
        except Exception:
            pipeline_seconds = None

    totals = {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "embedding_tokens": total_emb,
        "total_tokens": total_in + total_out + total_emb,
        "cost_estimate_usd": round(total_cost, 8),
        "pipeline_seconds": pipeline_seconds,
        "pipeline_start": first_ts,
        "pipeline_end": last_ts,
    }

    return {
        "document_id": document_id,
        "events": events,
        "totals": totals,
    }
