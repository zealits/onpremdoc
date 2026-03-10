"""
Shared document pipeline service layer.
Used by both the FastAPI app (main.py) and the MCP server (mcp_server.py).
Raises normal Python exceptions (ValueError, FileNotFoundError) so callers can
map them to HTTP responses or MCP tool error messages.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import networkx as nx

from config.inference_config import get_llm, get_embedding_model_id, get_llm_model_id
from detection import process_single_pdf
from economics_tracker import (
    log_pdf_processing,
    log_query_usage,
    log_upload,
    log_vectorization,
    log_page_summary,
    estimate_cost_for_usage,
)
from page_summarization import load_page_agent
from retrivalAgentE import (
    create_retrieval_agent,
    set_agent_resources,
    load_chunks_from_mapping,
    load_vector_store,
    find_vector_mapping_file,
    find_graph_file,
    find_vector_db_path,
    DocumentGraph as RetrievalDocumentGraph,
    AgentState,
)
from vectorizerE import (
    create_vectorization_workflow,
    VectorizerState,
    DocumentGraph,
)

from db import SessionLocal, DocumentRecord, ChatSession

logger = logging.getLogger(__name__)

# Configuration - must match main.py layout
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")

_loaded_agents: Dict[str, Dict[str, Any]] = {}


def _get_document_record(document_id: str) -> Optional[DocumentRecord]:
    db = SessionLocal()
    try:
        return (
            db.query(DocumentRecord)
            .filter(DocumentRecord.document_id == document_id)
            .first()
        )
    finally:
        db.close()


def register_document_for_user(user_id: int, document_id: str) -> None:
    """Create ownership record for a document."""
    db = SessionLocal()
    try:
        rec = DocumentRecord(user_id=user_id, document_id=document_id)
        db.add(rec)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ensure_document_belongs_to_user(document_id: str, user_id: int) -> None:
    """Raise ValueError if the document is not owned by this user."""
    rec = _get_document_record(document_id)
    if not rec or rec.user_id != user_id:
        raise ValueError(f"Document {document_id} not found for this user")


def get_document_path_for_user(user_id: int, document_id: str) -> Path:
    """Path for a document owned by a specific user (for new uploads)."""
    base = OUTPUT_DIR / f"user_{user_id}"
    return base / document_id


def get_document_path(document_id: str) -> Path:
    """Get document output directory (resolves user-specific folder if known)."""
    rec = _get_document_record(document_id)
    if rec:
        base = OUTPUT_DIR / f"user_{rec.user_id}"
    else:
        base = OUTPUT_DIR
    return base / document_id


def get_document_info(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document information as a dict (same keys as main.DocumentInfo).
    Returns None if the document does not exist.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        return None

    # Detect primary artifacts
    pdf_files = list(doc_path.glob("*.pdf"))
    md_files = list(doc_path.glob("*.md"))
    md_path = md_files[0] if md_files else None

    page_mapping_path = None
    if md_path:
        page_mapping_path = doc_path / f"{md_path.stem}_page_mapping.json"
        if not page_mapping_path.exists():
            page_mapping_path = None

    confidence_path = None
    layout_score = ocr_score = parse_score = table_score = None
    mean_grade = low_grade = None
    if md_path:
        potential_conf_path = doc_path / f"{md_path.stem}_confidence.json"
        if potential_conf_path.exists():
            confidence_path = potential_conf_path
            try:
                with open(confidence_path, "r", encoding="utf-8") as f:
                    confidence_data = json.load(f)
                layout_score = confidence_data.get("layout_score")
                ocr_score = confidence_data.get("ocr_score")
                parse_score = confidence_data.get("parse_score")
                table_score = confidence_data.get("table_score")
                mean_grade = confidence_data.get("mean_grade")
                low_grade = confidence_data.get("low_grade")
            except Exception:
                confidence_path = None

    plan_e_dir = doc_path / "E"
    vector_mapping_path = None
    graph_path = None
    vector_db_path = None
    total_chunks = None

    if plan_e_dir.exists():
        mapping_file = find_vector_mapping_file(plan_e_dir)
        if mapping_file:
            vector_mapping_path = str(mapping_file)
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    total_chunks = len(json.load(f))
            except Exception:
                pass
        graph_file = find_graph_file(plan_e_dir)
        if graph_file:
            graph_path = str(graph_file)
        vector_db = find_vector_db_path(plan_e_dir)
        if vector_db:
            vector_db_path = str(vector_db)

    status = "uploaded"
    if md_path:
        status = "processing"
    if vector_mapping_path and graph_path:
        status = "vectorized"
    if vector_mapping_path and graph_path and vector_db_path:
        status = "ready"

    total_pages = None
    doc_summary = None
    suggested_queries = None
    if page_mapping_path:
        try:
            with open(page_mapping_path, "r", encoding="utf-8") as f:
                total_pages = json.load(f).get("total_pages")
        except Exception:
            pass

    # Try to load document overview (summary + suggested queries) if present
    if md_path:
        try:
            stem = md_path.stem
            overview_path = doc_path / "E" / f"{stem}_doc_overview.json"
            if overview_path.exists():
                with open(overview_path, "r", encoding="utf-8") as f:
                    overview_data = json.load(f)
                doc_summary = overview_data.get("doc_summary")
                suggested_queries = overview_data.get("suggested_queries")
        except Exception:
            doc_summary = None
            suggested_queries = None

    # Prefer a human-friendly name derived from the original PDF filename
    # (e.g., "HDFC-Life-Cancer-Care-101N106V04-Policy-Document (7)").
    # If the PDF is missing (older documents), fall back to the markdown stem.
    # If that is also missing, use the folder name / document ID.
    if pdf_files:
        display_name = pdf_files[0].stem
    elif md_path:
        display_name = md_path.stem
    else:
        display_name = doc_path.name

    return {
        "document_id": document_id,
        "name": display_name,
        "status": status,
        "markdown_path": str(md_path) if md_path else None,
        "page_mapping_path": str(page_mapping_path) if page_mapping_path else None,
        "vector_mapping_path": vector_mapping_path,
        "graph_path": graph_path,
        "vector_db_path": vector_db_path,
        "confidence_path": str(confidence_path) if confidence_path else None,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "layout_score": layout_score,
        "ocr_score": ocr_score,
        "parse_score": parse_score,
        "table_score": table_score,
        "mean_grade": mean_grade,
        "low_grade": low_grade,
        "doc_summary": doc_summary,
        "suggested_queries": suggested_queries,
    }


def list_documents_for_user(user_id: int) -> List[Dict[str, Any]]:
    """List all documents owned by a specific user."""
    db = SessionLocal()
    try:
        records = (
            db.query(DocumentRecord)
            .filter(DocumentRecord.user_id == user_id)
            .order_by(DocumentRecord.created_at.desc())
            .all()
        )
    finally:
        db.close()

    docs: List[Dict[str, Any]] = []
    for rec in records:
        info = get_document_info(rec.document_id)
        if info:
            docs.append(info)
    return docs


def delete_document_for_user(user_id: int, document_id: str) -> None:
    """
    Delete a document and all related data (sessions, messages, ownership)
    for the given user. Raises ValueError if the document does not belong
    to the user.
    """
    # Ensure ownership
    ensure_document_belongs_to_user(document_id, user_id)

    # Delete files on disk
    doc_path = get_document_path(document_id)
    if doc_path.exists():
        try:
            shutil.rmtree(doc_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to remove document folder {doc_path}: {e}")

    # Delete DB rows (chat sessions/messages via cascade, then ownership record)
    db = SessionLocal()
    try:
        sessions = (
            db.query(ChatSession)
            .filter(
                ChatSession.user_id == user_id,
                ChatSession.document_id == document_id,
            )
            .all()
        )
        for s in sessions:
            db.delete(s)

        rec = (
            db.query(DocumentRecord)
            .filter(
                DocumentRecord.user_id == user_id,
                DocumentRecord.document_id == document_id,
            )
            .first()
        )
        if rec:
            db.delete(rec)

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def generate_quick_summary(document_id: str) -> str:
    """
    Generate a very fast document summary using only the
    beginning and end of the markdown file.
    """

    doc_path = get_document_path(document_id)

    md_files = list(doc_path.glob("*.md"))
    if not md_files:
        return "Document content not available."

    md_path = md_files[0]

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) < 4000:
        snippet = text
    else:
        start = text[:2000]
        end = text[-2000:]
        snippet = start + "\n...\n" + end

    llm = get_llm(temperature=0)

    prompt = f"""
You are generating a high-level overview of a document.

The text below contains only fragments from different parts of the document 
(beginning and ending sections). Use it only to infer the general topic 
and purpose of the document.

Write a short neutral summary (6–7 sentences) describing what the entire 
document is about, not just the provided text.

Your task is to just give a very highlevel summary of the document and not the specific details.

Text fragments:
{snippet}
"""

    response = llm.invoke(prompt)
    # Some providers (e.g. OpenAI) return an object with a .content attribute,
    # others (e.g. Ollama) may return a plain string. Handle both.
    text = getattr(response, "content", None) or str(response)
    return text.strip()

def load_agent_for_document(document_id: str) -> Dict[str, Any]:
    """Load agent resources for a document. Raises ValueError if not vectorized or files missing."""
    if document_id in _loaded_agents:
        return _loaded_agents[document_id]

    doc_path = get_document_path(document_id)
    plan_e_dir = doc_path / "E"

    if not plan_e_dir.exists():
        raise ValueError(f"Document {document_id} not vectorized. Run vectorization first.")

    vector_mapping_file = find_vector_mapping_file(plan_e_dir)
    graph_file = find_graph_file(plan_e_dir)
    vdb_path = find_vector_db_path(plan_e_dir)

    if not vector_mapping_file or not vector_mapping_file.exists():
        raise ValueError("Vector mapping file not found")
    if not graph_file or not graph_file.exists():
        raise ValueError("Graph file not found")
    if not vdb_path or not vdb_path.exists():
        raise ValueError("Vector database not found")

    chunks = load_chunks_from_mapping(vector_mapping_file)
    document_graph = RetrievalDocumentGraph()
    document_graph.load(graph_file)
    vector_store = load_vector_store(vdb_path)
    llm = get_llm(temperature=0.3)
    agent = create_retrieval_agent(vector_store, document_graph, chunks, llm, doc_path)
    set_agent_resources(vector_store, document_graph, chunks, llm, doc_path)

    _loaded_agents[document_id] = {
        "agent": agent,
        "vector_store": vector_store,
        "document_graph": document_graph,
        "chunks": chunks,
        "llm": llm,
    }
    return _loaded_agents[document_id]


def clear_loaded_agents() -> None:
  """Clear in-memory agent cache (used on application shutdown)."""
  _loaded_agents.clear()


def run_detection_for_document(document_id: str) -> None:
    """
    Run PDF detection (steps 3–6) for a document whose PDF is already in the document folder.
    Used by main.py after saving an uploaded file. Raises ValueError if no PDF found in folder.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")
    pdfs = list(doc_path.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF file found in document folder {doc_path}")
    pdf_path = pdfs[0]
    process_single_pdf(pdf_path, OUTPUT_DIR, target_dir=doc_path)
    total_pages = None
    md_files = list(doc_path.glob("*.md"))
    if md_files:
        pm_path = doc_path / f"{md_files[0].stem}_page_mapping.json"
        if pm_path.exists():
            try:
                with open(pm_path, "r", encoding="utf-8") as f:
                    total_pages = json.load(f).get("total_pages")
            except Exception:
                pass
    log_pdf_processing(document_id, total_pages=total_pages)


def upload_pdf_from_path(file_path: Path) -> str:
    """
    Copy PDF to a new document folder and run detection (steps 3–6). Returns document_id.
    Raises FileNotFoundError if file_path does not exist; ValueError if not a PDF.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    document_id = str(uuid4())
    doc_path = get_document_path(document_id)
    doc_path.mkdir(parents=True, exist_ok=True)
    dest_pdf = doc_path / file_path.name
    shutil.copy2(file_path, dest_pdf)
    log_upload(document_id, file_size_bytes=dest_pdf.stat().st_size, filename=file_path.name)

    process_single_pdf(dest_pdf, OUTPUT_DIR, target_dir=doc_path)

    total_pages = None
    md_files = list(doc_path.glob("*.md"))
    if md_files:
        pm_path = doc_path / f"{md_files[0].stem}_page_mapping.json"
        if pm_path.exists():
            try:
                with open(pm_path, "r", encoding="utf-8") as f:
                    total_pages = json.load(f).get("total_pages")
            except Exception:
                pass
    log_pdf_processing(document_id, total_pages=total_pages)
    return document_id


def trigger_vectorize(document_id: str) -> None:
    """
    Run vectorization for the document (synchronous). Raises ValueError if document
    not found or markdown missing.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")
    md_files = list(doc_path.glob("*.md"))
    if not md_files:
        raise ValueError("Document must be processed first (markdown not found)")

    logger.info("Starting vectorization for document: %s", document_id)
    initial_state: VectorizerState = {
        "markdown_file": str(doc_path),
        "chunks": [],
        "structure": {},
        "processed_chunks": [],
        "vector_store": None,
        "document_graph": DocumentGraph(),
        "json_mapping": [],
        "page_mapping": None,
        "page_classifications": None,
        "output_folder": str(doc_path),
        "token_usage": None,
    }
    workflow = create_vectorization_workflow()
    start_time = time.perf_counter()
    final_state = workflow.invoke(initial_state)
    duration = time.perf_counter() - start_time

    usage = final_state.get("token_usage") or {}
    embedding_tokens = int(usage.get("embedding_tokens", 0) or 0)
    llm_input_tokens = int(usage.get("llm_tokens", 0) or 0)
    llm_output_tokens = int(usage.get("llm_output_tokens", 0) or 0)
    total_chunks = int(usage.get("total_chunks", 0) or 0)
    truncated_chunks = int(usage.get("truncated_chunks", 0) or 0)
    total_tokens = embedding_tokens + llm_input_tokens + llm_output_tokens

    total_pages = None
    total_words = None
    md_files = list(doc_path.glob("*.md"))
    if md_files:
        md_path = md_files[0]
        try:
            text = md_path.read_text(encoding="utf-8")
            total_words = len(text.split())
        except Exception:
            total_words = None

        pm_path = doc_path / f"{md_path.stem}_page_mapping.json"
        if pm_path.exists():
            try:
                with open(pm_path, "r", encoding="utf-8") as f:
                    total_pages = json.load(f).get("total_pages")
            except Exception:
                total_pages = None

    log_vectorization(
        document_id,
        embedding_tokens=embedding_tokens,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        total_chunks=total_chunks,
        truncated_chunks=truncated_chunks,
        model_embed=get_embedding_model_id(),
        model_llm=get_llm_model_id(),
        total_pages=total_pages,
        total_words=total_words,
        total_tokens=total_tokens,
        duration_seconds=duration,
    )
    logger.info("Vectorization complete for document: %s", document_id)


def query_document(
    document_id: str,
    query: str,
    include_chunks: bool = True,
    streaming: bool = False,
    past_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run RAG query over a vectorized document. Returns dict with answer, retrieval_stats,
    chunks (if include_chunks), chunk_analysis, debug_info. When streaming=True, the
    agent only builds the answer prompt (no LLM call); result includes "answer_prompt"
    for the API to stream via llm.stream().
    past_messages: optional list of {role, content, chunks} for conversation context.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty")

    agent_resources = load_agent_for_document(document_id)
    agent = agent_resources["agent"]

    initial_state: AgentState = {
        "query": query,
        "is_page_summary": False,
        "page_number": None,
        "streaming_mode": streaming,
        "past_messages": past_messages or [],
        "is_use_history": False,
        "seed_chunk_ids": [],
        "seed_chunk_scores": {},
        "graph_expanded_ids": [],
        "retrieved_chunks": [],
        "reranked_chunks": None,
        "rerank_scores": {},
        "chunk_analysis": "",
        "needs_more_info": False,
        "new_query": None,
        "second_seed_ids": [],
        "second_seed_scores": {},
        "second_expanded_ids": [],
        "second_retrieval_chunks": [],
        "final_answer": "",
        "iteration_count": 0,
        "document_folder": None,
        "debug_info": {},
        "token_usage": [],
    }

    final_state = agent.invoke(initial_state)

    # Per-query economics (tokens + cost) for API and logging
    steps_usage = final_state.get("token_usage") or []
    if steps_usage:
        log_query_usage(document_id, steps_usage)

    q_total_in = sum(int(s.get("input_tokens", 0) or 0) for s in steps_usage)
    q_total_out = sum(int(s.get("output_tokens", 0) or 0) for s in steps_usage)
    q_total_emb = sum(int(s.get("embedding_tokens", 0) or 0) for s in steps_usage)
    q_total_tokens = q_total_in + q_total_out + q_total_emb
    q_pricing = estimate_cost_for_usage(q_total_in, q_total_out, q_total_emb)

    seed_chunk_ids = set(final_state.get("seed_chunk_ids", []))
    graph_expanded_ids = set(final_state.get("graph_expanded_ids", []))
    second_seed_ids = set(final_state.get("second_seed_ids", []))
    second_expanded_ids = set(final_state.get("second_expanded_ids", []))
    seed_chunk_scores = final_state.get("seed_chunk_scores", {})
    second_seed_scores = final_state.get("second_seed_scores", {})
    rerank_scores = final_state.get("rerank_scores", {})
    reranked_chunks = final_state.get("reranked_chunks")
    total_initial = len(reranked_chunks) if reranked_chunks else len(final_state.get("retrieved_chunks", []))

    retrieval_stats = {
        "seed_chunks": len(seed_chunk_ids),
        "graph_expanded_chunks": len(graph_expanded_ids),
        "total_initial_chunks": total_initial,
        "second_seed_chunks": len(second_seed_ids),
        "second_expanded_chunks": len(second_expanded_ids),
        "total_second_chunks": len(final_state.get("second_retrieval_chunks", [])),
        "total_chunks_used": total_initial + len(final_state.get("second_retrieval_chunks", [])),
        "iterations": final_state.get("iteration_count", 0),
        "reranking_applied": reranked_chunks is not None,
        "reranked_chunks_count": len(reranked_chunks) if reranked_chunks else 0,
    }
    if final_state.get("new_query"):
        retrieval_stats["second_query"] = final_state["new_query"]

    chunks_out: List[Dict[str, Any]] = []
    if include_chunks:
        # Page summary: build chunks from loaded agent's chunks using page_summary_chunk_indices
        if final_state.get("is_page_summary") and final_state.get("page_summary_chunk_indices") is not None:
            id_set = set(final_state["page_summary_chunk_indices"])
            resources = _loaded_agents.get(document_id)
            all_chunks = list(resources.get("chunks", [])) if resources else []
            for chunk in all_chunks:
                cid = chunk.metadata.get("chunk_index")
                if cid is None or cid not in id_set:
                    continue
                raw_lines = chunk.metadata.get("raw_content_lines")
                content_for_api = raw_lines if isinstance(raw_lines, list) else chunk.page_content
                chunks_out.append({
                    "chunk_index": cid,
                    "content": content_for_api,
                    "heading": chunk.metadata.get("heading", "No heading"),
                    "section_path": chunk.metadata.get("section_path", ""),
                    "section_title": chunk.metadata.get("section_title", ""),
                    "page_number": chunk.metadata.get("page_number"),
                    "page_classification": chunk.metadata.get("page_classification"),
                    "summary": chunk.metadata.get("summary", ""),
                    "chunk_type": chunk.metadata.get("chunk_type", "text"),
                    "has_table": chunk.metadata.get("has_table", False),
                    "table_context": chunk.metadata.get("table_context"),
                    "start_line": chunk.metadata.get("start_line"),
                    "content_length": len(chunk.page_content),
                    "retrieval_source": "page_summary",
                    "similarity_score": None,
                    "rerank_score": None,
                })
            chunks_out.sort(key=lambda c: c["chunk_index"])
        else:
            for chunk in final_state.get("retrieved_chunks", []):
                cid = chunk.metadata.get("chunk_index")
                if cid is None:
                    continue
                raw_lines = chunk.metadata.get("raw_content_lines")
                content_for_api = raw_lines if isinstance(raw_lines, list) else chunk.page_content
                source = "seed" if cid in seed_chunk_ids else ("graph_expanded" if cid in graph_expanded_ids else "initial")
                chunks_out.append({
                    "chunk_index": cid,
                    "content": content_for_api,
                    "heading": chunk.metadata.get("heading", "No heading"),
                    "section_path": chunk.metadata.get("section_path", ""),
                    "section_title": chunk.metadata.get("section_title", ""),
                    "page_number": chunk.metadata.get("page_number"),
                    "page_classification": chunk.metadata.get("page_classification"),
                    "summary": chunk.metadata.get("summary", ""),
                    "chunk_type": chunk.metadata.get("chunk_type", "text"),
                    "has_table": chunk.metadata.get("has_table", False),
                    "table_context": chunk.metadata.get("table_context"),
                    "start_line": chunk.metadata.get("start_line"),
                    "content_length": len(chunk.page_content),
                    "retrieval_source": source,
                    "similarity_score": seed_chunk_scores.get(cid),
                    "rerank_score": rerank_scores.get(cid),
                })
            for chunk in final_state.get("second_retrieval_chunks", []):
                cid = chunk.metadata.get("chunk_index")
                if cid is None or any(c["chunk_index"] == cid for c in chunks_out):
                    continue
                raw_lines = chunk.metadata.get("raw_content_lines")
                content_for_api = raw_lines if isinstance(raw_lines, list) else chunk.page_content
                source = "second_seed" if cid in second_seed_ids else ("second_expanded" if cid in second_expanded_ids else "second_retrieval")
                chunks_out.append({
                    "chunk_index": cid,
                    "content": content_for_api,
                    "heading": chunk.metadata.get("heading", "No heading"),
                    "section_path": chunk.metadata.get("section_path", ""),
                    "section_title": chunk.metadata.get("section_title", ""),
                    "page_number": chunk.metadata.get("page_number"),
                    "page_classification": chunk.metadata.get("page_classification"),
                    "summary": chunk.metadata.get("summary", ""),
                    "chunk_type": chunk.metadata.get("chunk_type", "text"),
                    "has_table": chunk.metadata.get("has_table", False),
                    "table_context": chunk.metadata.get("table_context"),
                    "start_line": chunk.metadata.get("start_line"),
                    "content_length": len(chunk.page_content),
                    "retrieval_source": source,
                    "similarity_score": second_seed_scores.get(cid),
                    "rerank_score": rerank_scores.get(cid),
                })
            # Return only chunks that were actually sent to the LLM for the answer (if tracked)
            used_ids = final_state.get("chunk_indices_used_for_answer")
            if used_ids is not None and isinstance(used_ids, (set, list)):
                used_set = set(used_ids)
                chunks_out = [c for c in chunks_out if c["chunk_index"] in used_set]
            chunks_out.sort(key=lambda c: c["chunk_index"])

    debug_info = dict(final_state.get("debug_info", {}))
    if rerank_scores:
        debug_info["rerank_scores_summary"] = {
            "total_reranked": len(rerank_scores),
            "avg_score": sum(rerank_scores.values()) / len(rerank_scores),
            "max_score": max(rerank_scores.values()),
            "min_score": min(rerank_scores.values()),
        }

    economics: Dict[str, Any] = {
        "steps": steps_usage,
        "total_input_tokens": q_total_in,
        "total_output_tokens": q_total_out,
        "total_embedding_tokens": q_total_emb,
        "total_tokens": q_total_tokens,
        "pricing": q_pricing,
    }

    out: Dict[str, Any] = {
        "answer": final_state.get("final_answer", "No answer generated"),
        "document_id": document_id,
        "query": query,
        "retrieval_stats": retrieval_stats,
        "chunk_analysis": final_state.get("chunk_analysis"),
        "chunks": chunks_out,
        "debug_info": debug_info,
        "economics": economics,
        "is_page_summary": final_state.get("is_page_summary", False),
        "is_use_history": final_state.get("is_use_history", False),
        "page_number": final_state.get("page_number"),
        "next_questions": final_state.get("next_questions") or [],
    }
    if streaming and final_state.get("answer_prompt"):
        out["answer_prompt"] = final_state["answer_prompt"]
    return out


def summarize_page(document_id: str, page_number: int) -> Dict[str, Any]:
    """
    Generate page-level summary. Returns dict with page_number, summary, key_points,
    sections, has_content, used_adjacent_pages, adjacent_pages_used, page_classification,
    chunks_used, total_chunks. Raises ValueError if document not found or not vectorized.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")
    plan_e_dir = doc_path / "E"
    if not plan_e_dir.exists():
        raise ValueError("Document not vectorized. Run vectorization first.")

    agent = load_page_agent(doc_path)
    if not agent:
        raise ValueError("Failed to load page summarization agent")

    summary_result = agent.summarize_page(page_number, use_adjacent_if_empty=True)
    chunks_used = len(summary_result.chunks_used) if summary_result.chunks_used else 0
    input_est = (chunks_used * 500 + 500) // 4
    output_est = max(1, len(summary_result.summary) // 4)
    log_page_summary(document_id, page_number, input_tokens_estimate=input_est, output_tokens_estimate=output_est, chunks_used=chunks_used)

    return {
        "page_number": summary_result.page_number,
        "summary": summary_result.summary,
        "key_points": summary_result.key_points,
        "sections": summary_result.sections,
        "has_content": summary_result.has_content,
        "used_adjacent_pages": summary_result.used_adjacent_pages,
        "adjacent_pages_used": summary_result.adjacent_pages_used,
        "page_classification": summary_result.page_classification,
        "chunks_used": summary_result.chunks_used or [],
        "total_chunks": chunks_used,
    }


def get_graph_stats(document_id: str) -> Dict[str, Any]:
    """
    Return graph node/edge counts and optional similarity stats. Returns empty-style
    stats if document or graph not found (no exception).
    """
    from visualizeGraphE import load_graph, find_graph_file as find_viz_graph_file

    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")

    plan_e_dir = doc_path / "E"
    empty = {
        "total_nodes": 0,
        "total_edges": 0,
        "node_types": {},
        "edge_relations": {},
        "density": 0.0,
    }
    if not plan_e_dir.exists():
        return empty

    graph_file = find_viz_graph_file(plan_e_dir)
    if not graph_file or not graph_file.exists():
        return empty

    try:
        G = load_graph(graph_file)
        node_types: Dict[str, int] = {}
        for nid in G.nodes():
            t = G.nodes[nid].get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
        edge_relations: Dict[str, int] = {}
        for _u, _v, d in G.edges(data=True):
            r = d.get("relation", "unknown")
            edge_relations[r] = edge_relations.get(r, 0) + 1
        stats = {
            "total_nodes": len(G.nodes),
            "total_edges": len(G.edges),
            "node_types": node_types,
            "edge_relations": edge_relations,
            "density": float(nx.density(G)) if len(G.nodes) > 0 else 0.0,
        }
        if "similar_to" in edge_relations:
            sim_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") == "similar_to"]
            if sim_edges:
                sims = [d.get("similarity", 0.0) for u, v, d in sim_edges if d.get("similarity")]
                if sims:
                    stats["similarity_stats"] = {
                        "count": len(sims),
                        "average": sum(sims) / len(sims),
                        "min": min(sims),
                        "max": max(sims),
                    }
        return stats
    except Exception as e:
        logger.error("Error getting graph stats: %s", e, exc_info=True)
        raise ValueError(f"Failed to get stats: {e}") from e


def get_document_markdown(document_id: str) -> str:
    """Return full markdown content for the document. Raises ValueError if not found."""
    info = get_document_info(document_id)
    if not info or not info.get("markdown_path"):
        raise ValueError("Markdown file not found for document")
    md_path = Path(info["markdown_path"])
    if not md_path.exists():
        raise ValueError("Markdown file not found")
    return md_path.read_text(encoding="utf-8")


def get_page_brief_summaries(document_id: str) -> List[Dict[str, Any]]:
    """
    Return section-level page summaries for a document.
    Data source: Plan E file *_page_brief_summaries.json produced by VectorizerE,
    which currently has the shape:
        [
          {"start_page": int, "end_page": int, "summary": str},
          ...
        ]
    Raises ValueError if document or summaries are not found.
    """
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise ValueError(f"Document {document_id} not found")

    plan_e_dir = doc_path / "E"
    if not plan_e_dir.exists():
        raise ValueError("Document not vectorized. Run vectorization first.")

    # Find the most recent *_page_brief_summaries.json in Plan E folder
    candidates = list(plan_e_dir.glob("*_page_brief_summaries.json"))
    if not candidates:
        raise ValueError("Page brief summaries not found for this document.")

    summaries_path = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        with summaries_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read page brief summaries: {e}") from e

    # Normalise to list of dicts with start_page, end_page, summary
    out: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            start_page = item.get("start_page")
            end_page = item.get("end_page")
            summary = (item.get("summary") or "").strip()
            if start_page is None or end_page is None or not summary:
                continue
            try:
                sp = int(start_page)
                ep = int(end_page)
            except (TypeError, ValueError):
                continue
            if sp <= 0 or ep < sp:
                continue
            out.append({"start_page": sp, "end_page": ep, "summary": summary})
    else:
        # Backwards-compatibility: if an older dict format is ever passed through,
        # just return empty rather than raising.
        out = []

    if not out:
        raise ValueError("No valid page brief summaries found for this document.")

    out.sort(key=lambda x: x["start_page"])
    return out


def _distance_to_similarity(distance: float, scale_factor: float = 150.0) -> float:
    """Convert L2 distance to similarity score [0, 1] for search results."""
    return 1.0 / (1.0 + (distance / scale_factor))


def search_document(document_id: str, query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Search a vectorized document by semantic similarity (retrieval only, no LLM).
    Returns a list of chunk dicts with content, page_number, section_title, etc.
    """
    query = (query or "").strip()
    if not query:
        return []

    agent_resources = load_agent_for_document(document_id)
    vector_store = agent_resources["vector_store"]
    document_graph = agent_resources["document_graph"]
    chunks_list = agent_resources["chunks"]

    # Vector search
    seed_chunk_ids = []
    seed_chunk_scores = {}
    try:
        vector_results = vector_store.similarity_search_with_score(query, k=20)
        for doc, distance in vector_results:
            chunk_id = doc.metadata.get("chunk_index")
            if chunk_id is not None:
                sim = _distance_to_similarity(distance)
                seed_chunk_ids.append(chunk_id)
                seed_chunk_scores[chunk_id] = sim
    except Exception as e:
        logger.warning("Search vector error for %s: %s", document_id, e)
        return []

    # Graph expansion from top seeds
    top_seeds = seed_chunk_ids[:5]
    expanded_ids = document_graph.expand_from_chunks(top_seeds, max_expansion=15) if top_seeds else []
    all_chunk_ids = list(dict.fromkeys(seed_chunk_ids + expanded_ids))  # preserve order, no dupes

    # Build chunk dict by id
    chunk_by_id = {}
    for ch in chunks_list:
        cid = ch.metadata.get("chunk_index")
        if cid is not None:
            chunk_by_id[cid] = ch

    out: List[Dict[str, Any]] = []
    for cid in all_chunk_ids[:limit]:
        ch = chunk_by_id.get(cid)
        if not ch:
            continue
        raw_lines = ch.metadata.get("raw_content_lines")
        content_for_api = raw_lines if isinstance(raw_lines, list) else ch.page_content
        out.append({
            "chunk_index": cid,
            "content": content_for_api,
            "heading": ch.metadata.get("heading", ""),
            "section_path": ch.metadata.get("section_path", ""),
            "section_title": ch.metadata.get("section_title", ""),
            "page_number": ch.metadata.get("page_number"),
            "summary": ch.metadata.get("summary", ""),
            "similarity_score": seed_chunk_scores.get(cid),
        })
    return out


def extract_information(document_id: str, extract_type: str = "key_facts") -> Dict[str, Any]:
    """
    Extract structured information from a document using the LLM.
    extract_type: 'key_facts' | 'entities' | 'dates' | 'obligations'
    Returns a dict with 'extract_type', 'items' (list), and optional 'summary'.
    """
    agent_resources = load_agent_for_document(document_id)
    chunks_list = agent_resources["chunks"]
    llm = get_llm(temperature=0)

    # Build context from first 30 chunks (or all if fewer)
    context_parts = []
    for ch in chunks_list[:30]:
        context_parts.append(ch.page_content[:800])
    context = "\n\n---\n\n".join(context_parts)
    if len(context) > 12000:
        context = context[:12000] + "\n...[truncated]"

    type_prompts = {
        "key_facts": "List the key facts, main points, and important conclusions from the document. Return a JSON array of strings, one per fact.",
        "entities": "List the main entities mentioned: people, organizations, products, places. Return a JSON array of strings.",
        "dates": "List all explicit dates, deadlines, or time periods mentioned. Return a JSON array of strings.",
        "obligations": "List obligations, requirements, or commitments stated in the document. Return a JSON array of strings.",
    }
    prompt_text = type_prompts.get(extract_type, type_prompts["key_facts"])

    prompt = f"""Based on the following document excerpts, {prompt_text}

Document excerpts:
{context}

Respond with only a valid JSON array of strings, no other text. Example: ["item1", "item2"]"""

    try:
        import re
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        text = (text or "").strip()
        # Try to parse JSON array
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            items = json.loads(match.group())
            if isinstance(items, list):
                items = [str(x) for x in items if x]
            else:
                items = []
        else:
            items = [line.strip() for line in text.split("\n") if line.strip()][:50]
        return {"extract_type": extract_type, "items": items}
    except Exception as e:
        logger.warning("extract_information failed for %s: %s", document_id, e)
        return {"extract_type": extract_type, "items": [], "error": str(e)}


def send_document_summary_email(document_id: str, to_email: str, subject: Optional[str] = None) -> Dict[str, Any]:
    """
    Send the document summary to the given email address.
    Uses SMTP if configured (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM);
    otherwise logs and returns success for testing.
    """
    import os
    to_email = (to_email or "").strip()
    if not to_email or "@" not in to_email:
        raise ValueError("Valid recipient email is required")

    summary_text = ""
    try:
        doc_info = get_document_info(document_id)
        if doc_info and (doc_info.get("doc_summary") or "").strip():
            summary_text = doc_info["doc_summary"].strip()
        else:
            summary_text = generate_quick_summary(document_id)
    except Exception as e:
        logger.warning("Could not get summary for email: %s", e)
        summary_text = "Summary not available."

    doc_name = ""
    info = get_document_info(document_id)
    if info:
        doc_name = info.get("name") or document_id

    email_subject = subject or f"Document summary: {doc_name}"
    body = f"Document: {doc_name}\n\nSummary:\n{summary_text}"

    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    # Support both SMTP_MAIL and SMTP_USER for sender/login
    smtp_user = os.environ.get("SMTP_USER") or os.environ.get("SMTP_MAIL")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    email_from = os.environ.get("EMAIL_FROM") or smtp_user

    if smtp_host and smtp_user and smtp_pass:
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = email_subject
            msg["From"] = email_from
            msg["To"] = to_email
            # Port 465 uses SSL (SMTPS); 587 uses STARTTLS
            if smtp_port == 465:
                with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(email_from, [to_email], msg.as_string())
            else:
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(email_from, [to_email], msg.as_string())
            return {"sent": True, "to": to_email, "message": "Email sent successfully."}
        except Exception as e:
            logger.exception("SMTP send failed")
            raise ValueError(f"Failed to send email: {e}") from e
    else:
        logger.info("Email (no SMTP): would send to %s subject %s. Configure SMTP_* and EMAIL_FROM to send.", to_email, email_subject)
        return {"sent": True, "to": to_email, "message": "Summary prepared; email not sent (SMTP not configured)."}
