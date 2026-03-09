"""
FastAPI Application for Document Processing Pipeline
Integrates PDF detection, vectorization, retrieval, and visualization
"""

import asyncio
import json
import logging
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

import networkx as nx
import uvicorn
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Query,
    Path as PathParam,
    status,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Import modules
from config.inference_config import check_inference_ready
from visualizeGraphE import (
    load_graph,
    visualize_interactive,
    visualize_static,
    visualize_simplified,
    find_graph_file as find_viz_graph_file,
)
from economics_tracker import (
    log_upload,
    get_usage_summary,
    get_latest_vectorization_for_document,
    get_document_pipeline_economics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - use service layer for paths and document ops
from services import document_service
from db import init_db, ChatSession, ChatMessage
from auth import router as auth_router, get_current_user, get_db
from chat_api import router as chat_router

OUTPUT_DIR = document_service.OUTPUT_DIR
UPLOAD_DIR = document_service.UPLOAD_DIR
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- PYDANTIC MODELS ----------------


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ollama_available: bool
    ollama_models: List[str]


class DocumentInfo(BaseModel):
    """Document information"""
    document_id: str
    name: str
    status: str  # "uploaded", "processing", "vectorized", "ready"
    # Paths to artifacts
    markdown_path: Optional[str] = None
    page_mapping_path: Optional[str] = None
    vector_mapping_path: Optional[str] = None
    graph_path: Optional[str] = None
    vector_db_path: Optional[str] = None
    confidence_path: Optional[str] = None
    # Summary stats
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = None
    # Docling confidence (document-level)
    layout_score: Optional[float] = None
    ocr_score: Optional[float] = None
    parse_score: Optional[float] = None
    table_score: Optional[float] = None
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None
    # High-level document overview (generated after vectorization)
    doc_summary: Optional[str] = None
    suggested_queries: Optional[List[str]] = None


class ConfidencePageScores(BaseModel):
    """Per-page confidence scores"""
    parse_score: Optional[float] = None
    layout_score: Optional[float] = None
    table_score: Optional[float] = None
    ocr_score: Optional[float] = None


class ConfidenceResponse(BaseModel):
    """Confidence / accuracy information for a document"""
    document_id: str
    has_confidence: bool
    layout_score: Optional[float] = None
    ocr_score: Optional[float] = None
    parse_score: Optional[float] = None
    table_score: Optional[float] = None
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None
    pages: Dict[str, ConfidencePageScores] = Field(default_factory=dict)


class ProcessPDFRequest(BaseModel):
    """Request to process a PDF"""
    document_id: Optional[str] = None
    preserve_page_breaks: bool = True


class ProcessPDFResponse(BaseModel):
    """Response from PDF processing"""
    document_id: str
    status: str
    message: str
    markdown_path: Optional[str] = None
    page_mapping_path: Optional[str] = None
    total_pages: Optional[int] = None


class VectorizeRequest(BaseModel):
    """Request to vectorize a document"""
    document_id: str


class VectorizeResponse(BaseModel):
    """Response from vectorization"""
    document_id: str
    status: str
    message: str
    vector_mapping_path: Optional[str] = None
    graph_path: Optional[str] = None
    vector_db_path: Optional[str] = None
    total_chunks: Optional[int] = None
    graph_nodes: Optional[int] = None
    graph_edges: Optional[int] = None


class ChunkDetail(BaseModel):
    """Detailed information about a retrieved chunk"""
    chunk_index: int
    # Content can be either:
    # - List[str]: exact lines as stored in vector_mapping JSON (preferred for highlighting in markdown)
    # - str: fallback for older documents
    content: Any
    heading: str
    section_path: str
    section_title: str
    page_number: Optional[int] = None
    page_classification: Optional[str] = None
    summary: str
    chunk_type: str
    has_table: bool
    table_context: Optional[str] = None
    start_line: Optional[int] = None
    content_length: int
    retrieval_source: str = Field(..., description="How chunk was retrieved: 'seed', 'graph_expanded', 'second_seed', 'second_expanded', 'initial', 'second_retrieval'")
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = Field(None, description="Re-ranking relevance score [0-1]")


class QueryRequest(BaseModel):
    """Query request"""
    query: str = Field(..., description="The query to search for")
    document_id: str = Field(..., description="Document ID to query")
    include_chunks: bool = Field(default=True, description="Include detailed chunk information in response")
    session_id: Optional[int] = Field(default=None, description="Chat session ID for preserving conversation context")


class QueryResponse(BaseModel):
    """Query response with detailed chunk information"""
    answer: str
    document_id: str
    query: str
    retrieval_stats: Dict[str, Any]
    chunk_analysis: Optional[str] = None
    chunks: List[ChunkDetail] = Field(default_factory=list, description="Detailed information about chunks used to generate the answer")
    debug_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Debugging information about retrieval process")
    next_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions based on the answer")


class VisualizationRequest(BaseModel):
    """Request for graph visualization"""
    document_id: str
    visualization_type: str = Field(
        default="interactive",
        description="Type: interactive, static, simplified"
    )


class PageSummaryRequest(BaseModel):
    """Request for page summarization"""
    document_id: str = Field(..., description="Document ID")
    page_number: int = Field(..., description="Page number to summarize", gt=0)


class PageSummaryResponse(BaseModel):
    """Page summary response"""
    page_number: int
    summary: str
    key_points: List[str]
    sections: List[str]
    has_content: bool
    used_adjacent_pages: bool
    adjacent_pages_used: Optional[List[int]] = None
    page_classification: Optional[str] = None
    chunks_used: List[int]
    total_chunks: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


class VectorizationEconomicsResponse(BaseModel):
    """Vectorization economics for a single document (for frontend UI)"""
    document_id: str
    date: str
    total_pages: Optional[int] = None
    total_words: Optional[int] = None
    total_tokens: Optional[int] = None
    duration_seconds: Optional[float] = None
    cost_estimate_usd: Optional[float] = None
    cost_display: Optional[str] = None
    pricing: Dict[str, Any] = Field(default_factory=dict)


class PipelineEconomicsResponse(BaseModel):
    """Combined upload + processing + vectorization economics for one document"""
    document_id: str
    events: List[Dict[str, Any]]
    totals: Dict[str, Any]


# ---------------- LIFECYCLE MANAGEMENT ----------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")
    # Ensure database schema exists
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
    
    # Check inference backend on startup
    is_running, models = check_inference_ready()
    if not is_running:
        logger.warning("Inference backend not detected. Some features may not work.")
    else:
        logger.info("Inference backend ready: %s", models[:5] if len(models) > 5 else models)
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")
    try:
        document_service.clear_loaded_agents()
    except Exception as e:
        logger.warning(f"Failed to clear loaded agents on shutdown: {e}")


# ---------------- FASTAPI APP ----------------


app = FastAPI(
    title="DocOnPrem API",
    description="On-premises document pipeline: PDF processing, vectorization, graph-based retrieval, and visualization",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(chat_router)


# ---------------- HELPER FUNCTIONS (delegate to service) ----------------


def get_document_path(document_id: str) -> Path:
    """Get document output directory"""
    return document_service.get_document_path(document_id)


def get_document_info(document_id: str) -> Optional[DocumentInfo]:
    """Get document information"""
    info = document_service.get_document_info(document_id)
    return DocumentInfo(**info) if info else None


def load_agent_for_document(document_id: str) -> Dict[str, Any]:
    """Load agent resources for a document. Raises HTTPException if not vectorized."""
    try:
        return document_service.load_agent_for_document(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _chunk_markdown(text: str, max_chunk_chars: int = 400) -> List[str]:
    """
    Split markdown text into reasonably sized chunks for streaming.
    Prefer splitting on paragraph boundaries to keep formatting clean.
    """
    if not text:
        return []

    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        # Include the paragraph separator when joining
        para_with_sep = para + "\n\n"
        if current_len + len(para_with_sep) > max_chunk_chars and current:
            chunks.append("".join(current))
            current = [para_with_sep]
            current_len = len(para_with_sep)
        else:
            current.append(para_with_sep)
            current_len += len(para_with_sep)

    if current:
        chunks.append("".join(current))

    return chunks


def _extract_stream_delta(chunk: Any) -> str:
    """
    Extract just the text delta from a streaming LLM chunk.
    Handles common LangChain / OpenAI-style chunk shapes and
    avoids dumping reprs with response_metadata, tool_calls, etc.
    """
    # 1) Prefer structured dump if available (LangChain message chunks are pydantic models)
    data = None
    if hasattr(chunk, "model_dump"):
        try:
            data = chunk.model_dump()
        except Exception:
            data = None
    if isinstance(data, dict) and "content" in data:
        content = data.get("content")
    else:
        content = getattr(chunk, "content", None)

    # 2) Handle common shapes for `.content`
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some providers return a list of content parts
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # OpenAI-style: {"type": "text", "text": "..."}
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return "".join(parts)

    # 3) Raw dict response (OpenAI HTTP-like shapes)
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str):
                    return delta["content"]
        return ""

    return ""


# ---------------- API ENDPOINTS ----------------


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "documents": "/documents",
            "upload": "/upload",
            "vectorize": "/vectorize/{document_id}",
            "query": "/query",
            "visualize": "/visualize/{document_id}",
            "page_summarize": "/page/{document_id}/summarize",
            "economics_summary": "/economics/summary",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint (inference backend: Ollama or Hugging Face)"""
    is_running, models = check_inference_ready()
    return HealthResponse(
        status="healthy" if is_running else "degraded",
        ollama_available=is_running,
        ollama_models=models,
    )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents(current_user=Depends(get_current_user)):
    """List all processed documents for the current user"""
    docs = document_service.list_documents_for_user(current_user.id)
    return [DocumentInfo(**d) for d in docs]


@app.get("/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document(
    document_id: str = PathParam(..., description="Document ID"),
    current_user=Depends(get_current_user),
):
    """Get document information"""
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )
    doc_info = get_document_info(document_id)
    if not doc_info:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    return doc_info


@app.get("/documents/{document_id}/markdown", tags=["Documents"])
async def get_markdown(
    document_id: str = PathParam(..., description="Document ID"),
    current_user=Depends(get_current_user),
):
    """Get markdown content for a document"""
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
        content = document_service.get_document_markdown(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    from fastapi.responses import Response
    return Response(content=content, media_type="text/plain")


@app.get("/documents/{document_id}/confidence", response_model=ConfidenceResponse, tags=["Documents"])
async def get_confidence(
    document_id: str = PathParam(..., description="Document ID"),
    current_user=Depends(get_current_user),
):
    """Get Docling confidence / accuracy scores for a document"""
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )
    doc_info = get_document_info(document_id)
    if not doc_info:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )

    if not doc_info.confidence_path:
        # No confidence available (older runs or failure)
        return ConfidenceResponse(
            document_id=document_id,
            has_confidence=False,
        )

    conf_path = Path(doc_info.confidence_path)
    if not conf_path.exists():
        return ConfidenceResponse(
            document_id=document_id,
            has_confidence=False,
        )

    try:
        with open(conf_path, "r", encoding="utf-8") as f:
            conf_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read confidence file for {document_id}: {e}", exc_info=True)
        return ConfidenceResponse(
            document_id=document_id,
            has_confidence=False,
        )

    # Build per-page scores map
    pages_raw = conf_data.get("pages") or {}
    pages: Dict[str, ConfidencePageScores] = {}
    for page_no, scores in pages_raw.items():
        # keys might be ints or strings; normalize to string for frontend
        key = str(page_no)
        pages[key] = ConfidencePageScores(
            parse_score=scores.get("parse_score"),
            layout_score=scores.get("layout_score"),
            table_score=scores.get("table_score"),
            ocr_score=scores.get("ocr_score"),
        )

    return ConfidenceResponse(
        document_id=document_id,
        has_confidence=True,
        layout_score=conf_data.get("layout_score"),
        ocr_score=conf_data.get("ocr_score"),
        parse_score=conf_data.get("parse_score"),
        table_score=conf_data.get("table_score"),
        mean_grade=conf_data.get("mean_grade"),
        low_grade=conf_data.get("low_grade"),
        pages=pages,
    )


@app.get("/documents/{document_id}/pdf", tags=["Documents"])
async def get_document_pdf(
    document_id: str = PathParam(..., description="Document ID"),
    current_user=Depends(get_current_user),
):
    """Serve the original PDF file for the document (for viewer)."""
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    pdf_files = list(doc_path.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(status_code=404, detail=f"No PDF file found for document {document_id}")
    pdf_path = pdf_files[0]
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{pdf_path.name}"'},
    )


@app.post("/upload", response_model=ProcessPDFResponse, tags=["Documents"])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user=Depends(get_current_user),
):
    """Upload and process a PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Generate document ID and user-specific document path
    document_id = str(uuid4())
    doc_path = document_service.get_document_path_for_user(current_user.id, document_id)
    doc_path.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    pdf_path = doc_path / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_size = pdf_path.stat().st_size
    log_upload(document_id, file_size_bytes=file_size, filename=file.filename or "")
    # Register ownership
    document_service.register_document_for_user(current_user.id, document_id)
    
    # Process in background
    background_tasks.add_task(process_pdf_background, str(pdf_path), document_id)
    
    return ProcessPDFResponse(
        document_id=document_id,
        status="processing",
        message="PDF uploaded and processing started",
    )


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Documents"])
async def delete_document(
    document_id: str,
    current_user=Depends(get_current_user),
):
    """Delete a document and all associated chat data for the current user."""
    try:
        document_service.delete_document_for_user(current_user.id, document_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )
    return


def process_pdf_background(pdf_path: str, document_id: str):
    """Background task to process PDF (file already saved in document folder)"""
    try:
        logger.info(f"Processing PDF: {pdf_path} for document {document_id}")
        document_service.run_detection_for_document(document_id)
        logger.info(f"PDF processing complete for document {document_id}")
    except ValueError as e:
        logger.error(f"Error processing PDF {document_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)

@app.post("/vectorize/{document_id}")
async def vectorize_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    doc_info = get_document_info(document_id)

    # Generate quick summary immediately
    summary = document_service.generate_quick_summary(document_id)

    # Start vectorization in background
    background_tasks.add_task(
        document_service.trigger_vectorize,
        document_id
    )

    return {
        "document_id": document_id,
        "status": "vectorization_started",
        "summary": summary
    }


def vectorize_background(document_id: str):
    """Background task to vectorize document"""
    try:
        logger.info(f"{'='*80}")
        logger.info(f"🚀 Starting vectorization for document: {document_id}")
        logger.info(f"{'='*80}")
        document_service.trigger_vectorize(document_id)
        doc_path = get_document_path(document_id)
        logger.info(f"{'='*80}")
        logger.info(f"✅ Vectorization complete for document {document_id}")
        logger.info(f"   📂 Output directory: {doc_path / 'E'}")
        logger.info(f"{'='*80}")
    except ValueError as e:
        logger.error(f"❌ Vectorization failed for document {document_id}: {e}")
    except Exception as e:
        logger.error(f"{'='*80}")
        logger.error(f"❌ Error vectorizing document {document_id}: {e}")
        logger.error(f"{'='*80}")
        logger.error(f"Error details:", exc_info=True)


@app.get("/economics/summary", tags=["Economics"])
async def economics_summary(date: Optional[str] = Query(None, description="Date YYYY-MM-DD (default: today UTC)")):
    """
    Token usage and cost visibility for stakeholders.
    Returns aggregated token counts by phase (upload, vectorization, retrieval, page_summary)
    and by step. Data is read from the economics/ folder (usage_<date>.jsonl).
    """
    try:
        summary = get_usage_summary(date_str=date)
        return summary
    except Exception as e:
        logger.error(f"Economics summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/economics/pipeline/{document_id}",
    response_model=PipelineEconomicsResponse,
    tags=["Economics"],
)
async def economics_pipeline(
    document_id: str = PathParam(..., description="Document ID"),
    current_user=Depends(get_current_user),
):
    """
    Combined economics for upload + PDF processing + vectorization for one document.

    This is what the frontend should call when it only knows the document_id.
    Retrieval/query economics are deliberately excluded.
    """
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )

    try:
        data = get_document_pipeline_economics(document_id)
    except Exception as e:
        logger.error(f"Pipeline economics error for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if not data.get("events"):
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline economics found for document {document_id}",
        )

    return PipelineEconomicsResponse(**data)


@app.get(
    "/economics/vectorization/{document_id}",
    response_model=VectorizationEconomicsResponse,
    tags=["Economics"],
)
async def economics_vectorization(
    document_id: str = PathParam(..., description="Document ID"),
    current_user=Depends(get_current_user),
):
    """
    Return vectorization-level economics for a specific document.

    Data source: economics/usage_*.jsonl, filtered by:
    - phase == "vectorization"
    - document_id == <document_id>
    The most recent matching record is returned.
    """
    # Ensure user actually owns this document
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )

    try:
        rec = get_latest_vectorization_for_document(document_id)
    except Exception as e:
        logger.error(f"Vectorization economics error for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if not rec:
        raise HTTPException(
            status_code=404,
            detail=f"No vectorization economics found for document {document_id}",
        )

    extra = rec.get("extra") or {}
    pricing = rec.get("pricing") or {}

    return VectorizationEconomicsResponse(
        document_id=document_id,
        date=datetime.utcnow().strftime("%Y-%m-%d"),
        total_pages=extra.get("total_pages"),
        total_words=extra.get("total_words"),
        total_tokens=extra.get("total_tokens"),
        duration_seconds=extra.get("duration_seconds"),
        cost_estimate_usd=rec.get("cost_estimate_usd"),
        cost_display=pricing.get("cost_display"),
        pricing=pricing,
    )


# Number of past messages to send as conversation context to the LLM
PAST_N_MESSAGES = 6
# Max chunks per assistant message to include (only those cited in the answer)
MAX_CHUNKS_PER_PAST_MESSAGE = 2

def _chunk_refs_in_content(content: str) -> List[int]:
    """Parse [C0], [C15], etc. from content; return unique chunk indices in order of first appearance."""
    if not content:
        return []
    seen = set()
    order = []
    for m in re.finditer(r"\[C(\d+)\]", content, re.IGNORECASE):
        idx = int(m.group(1))
        if idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order

def _build_past_messages(
    db: Any, session_id: int, before_message_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load last N messages for the session (before the given message id) and build
    past_messages for the agent. For each assistant message, include only the
    chunks that appear in the answer ([C0], [C15], etc.), max MAX_CHUNKS_PER_PAST_MESSAGE.
    """
    q = db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
    if before_message_id is not None:
        q = q.filter(ChatMessage.id < before_message_id)
    q = q.order_by(ChatMessage.id.desc()).limit(PAST_N_MESSAGES + 1)
    rows = q.all()
    rows = list(reversed(rows))  # chronological
    out: List[Dict[str, Any]] = []
    for msg in rows:
        entry: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
        if msg.role == "assistant" and getattr(msg, "message_metadata", None):
            meta = msg.message_metadata or {}
            chunks_meta = meta.get("chunks") or []
            refs = _chunk_refs_in_content(entry["content"])
            if refs and chunks_meta:
                ref_set = set(refs)
                matched = [c for c in chunks_meta if c.get("chunk_index") in ref_set]
                # Keep order of first appearance in content
                matched.sort(key=lambda c: refs.index(c["chunk_index"]) if c["chunk_index"] in refs else 999)
                entry["chunks"] = matched[:MAX_CHUNKS_PER_PAST_MESSAGE]
            else:
                entry["chunks"] = []
        else:
            entry["chunks"] = []
        out.append(entry)
    return out


def _generate_next_questions(llm: Any, query: str, answer: str) -> List[str]:
    """Generate 3–7 suggested follow-up questions from the user query and the generated answer."""
    if not (query and answer):
        return []
    followup_prompt = f"""You are helping a user explore a long insurance policy document.

Original user question:
{query}

Your answer:
{answer}

Now propose 3–7 SHORT, concrete follow-up questions that the user might naturally ask next
to go deeper or clarify details. Focus on practical, answerable questions about this document.

Reply in the following format ONLY:
- question 1
- question 2
- question 3
..."""
    try:
        response = llm.invoke(followup_prompt)
        text = (getattr(response, "content", None) or str(response)).strip()
    except Exception:
        return []
    suggestions: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0] in "-*":
            q = line[1:].strip()
        else:
            q = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if q:
            suggestions.append(q)
    seen = set()
    deduped = []
    for q in suggestions:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(q)
    return deduped[:7]


def _ensure_session_for_request(
    db: Any, current_user: Any, request: QueryRequest
) -> int:
    """
    Ensure there is a ChatSession for this user + document.
    If request.session_id is provided, validate ownership and return it.
    Otherwise, create a new session and return its id.
    """
    if request.session_id is not None:
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.id,
                ChatSession.is_archived.is_(False),
            )
            .first()
        )
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )
        if session.document_id != request.document_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session does not belong to this document",
            )
        return session.id

    # No session_id provided: create a new session for this document
    session = ChatSession(
        user_id=current_user.id,
        document_id=request.document_id,
        title=None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session.id


@app.post("/query", tags=["Retrieval"])
async def query_document(
    request: QueryRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Single query endpoint: runs the LangGraph agent (llm.invoke) once, then
    streams the agent's final answer to the client. No second LLM call;
    /query and the streamed content are the same.
    """
    # Ensure we have a session for this query
    session_id = _ensure_session_for_request(db, current_user, request)

    # Create and persist the user message immediately
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.query,
        message_metadata=None,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Build past N messages for conversation context (content + chunks cited in answer, max 2 per message)
    past_messages = _build_past_messages(db, session_id, before_message_id=user_msg.id)

    try:
        # Run the agent in streaming mode: it does retrieval and builds the answer
        # prompt but does not call the LLM; we stream via llm.stream() below.
        result = document_service.query_document(
            request.document_id,
            request.query,
            include_chunks=request.include_chunks,
            streaming=True,
            past_messages=past_messages,
        )
    except ValueError as e:
        status_code = 400 if "query" in str(e).lower() else 404
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error during query (retrieval phase): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    is_page_summary = bool(result.get("is_page_summary"))
    answer_prompt = result.get("answer_prompt")
    final_answer_text = (result.get("answer") or "").strip()

    async def event_generator():
        meta = {
            "type": "meta",
            "session_id": session_id,
            "document_id": result["document_id"],
            "query": result["query"],
            "retrieval_stats": result["retrieval_stats"],
            "economics": result.get("economics") or {},
            "chunks": result.get("chunks") or [],
            "chunk_analysis": result.get("chunk_analysis"),
            "debug_info": result.get("debug_info") or {},
            "next_questions": result.get("next_questions") or [],
            "is_page_summary": result.get("is_page_summary", False),
            "is_use_history": result.get("is_use_history", False),
            "page_number": result.get("page_number"),
        }
        yield json.dumps(meta, ensure_ascii=False) + "\n"

        streamed_any = False
        full_answer_parts: List[str] = []
        llm = None
        try:
            # True streaming or single response: when we have answer_prompt (normal Q&A or page-summary).
            if answer_prompt:
                try:
                    agent_resources = document_service.load_agent_for_document(request.document_id)
                    llm = agent_resources.get("llm")
                except Exception as e:
                    logger.error(f"Error loading LLM for streaming: {e}", exc_info=True)
                    llm = None
                if llm is not None and (hasattr(llm, "astream") or hasattr(llm, "stream")):
                    # Actual token-level streaming (normal Q&A and page-summary)
                    if hasattr(llm, "astream"):
                        async for chunk in llm.astream(answer_prompt):
                            delta = _extract_stream_delta(chunk)
                            if not delta:
                                continue
                            streamed_any = True
                            full_answer_parts.append(delta)
                            yield json.dumps(
                                {"type": "answer_chunk", "delta": delta},
                                ensure_ascii=False,
                            ) + "\n"
                    else:
                        # Sync llm.stream(): run in thread and feed queue for async yield
                        loop = asyncio.get_event_loop()
                        q: asyncio.Queue = asyncio.Queue()

                        def produce():
                            for chunk in llm.stream(answer_prompt):
                                delta = _extract_stream_delta(chunk)
                                if delta:
                                    loop.call_soon_threadsafe(q.put_nowait, delta)
                            loop.call_soon_threadsafe(q.put_nowait, None)

                        loop.run_in_executor(None, produce)
                        while True:
                            delta = await q.get()
                            if delta is None:
                                break
                            streamed_any = True
                            full_answer_parts.append(delta)
                            yield json.dumps(
                                {"type": "answer_chunk", "delta": delta},
                                ensure_ascii=False,
                            ) + "\n"
                else:
                    # LLM has neither stream nor astream: use .invoke() and send full answer once (no streaming)
                    if llm is not None:
                        try:
                            response = llm.invoke(answer_prompt)
                            full_text = (
                                (getattr(response, "content", None) or str(response)).strip()
                                if response else ""
                            )
                            if full_text:
                                streamed_any = True
                                full_answer_parts.append(full_text)
                                yield json.dumps(
                                    {"type": "answer_chunk", "delta": full_text},
                                    ensure_ascii=False,
                                ) + "\n"
                        except Exception as inv_err:
                            logger.error(f"LLM invoke error: {inv_err}", exc_info=True)
                    if not streamed_any:
                        text_to_send = final_answer_text or "I could not generate an answer for this query."
                        streamed_any = True
                        full_answer_parts.append(text_to_send)
                        yield json.dumps(
                            {"type": "answer_chunk", "delta": text_to_send},
                            ensure_ascii=False,
                        ) + "\n"
            else:
                # No answer_prompt (e.g. page with no content, or error): send fallback once
                text_to_send = final_answer_text or "I could not generate an answer for this query."
                streamed_any = True
                full_answer_parts.append(text_to_send)
                yield json.dumps(
                    {"type": "answer_chunk", "delta": text_to_send},
                    ensure_ascii=False,
                ) + "\n"
        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}", exc_info=True)
            yield json.dumps(
                {"type": "error", "message": "Streaming failed while generating the answer."},
                ensure_ascii=False,
            ) + "\n"

        if not streamed_any:
            # Ensure the client always receives something for the answer.
            yield json.dumps(
                {"type": "answer_chunk", "delta": "I could not generate an answer for this query."},
                ensure_ascii=False,
            ) + "\n"
            # Persist a minimal assistant message
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content="I could not generate an answer for this query.",
                message_metadata=None,
            )
            db.add(assistant_msg)
            db.commit()
        else:
            # Persist the full assistant message with metadata
            full_answer = "".join(full_answer_parts)
            # Generate suggested follow-up questions (streaming path: agent skips this, so we do it here)
            next_questions_list = result.get("next_questions") or []
            if full_answer and not next_questions_list and llm is not None:
                next_questions_list = _generate_next_questions(llm, request.query, full_answer)
            yield json.dumps(
                {"type": "next_questions", "next_questions": next_questions_list},
                ensure_ascii=False,
            ) + "\n"
            assistant_metadata = {
                "retrieval_stats": result.get("retrieval_stats"),
                "chunks": result.get("chunks"),
                "next_questions": next_questions_list,
            }
            assistant_metadata["is_page_summary"] = is_page_summary
            assistant_metadata["page_number"] = result.get("page_number")
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_answer,
                message_metadata=assistant_metadata,
            )
            db.add(assistant_msg)
            db.commit()

        # Signal completion
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/json")


@app.post("/visualize/{document_id}", tags=["Visualization"])
async def visualize_graph(
    document_id: str = PathParam(..., description="Document ID"),
    viz_type: str = Query(default="interactive", description="Type: interactive, static, simplified"),
    current_user=Depends(get_current_user),
):
    """Generate graph visualization"""
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )
    doc_path = get_document_path(document_id)
    plan_e_dir = doc_path / "E"
    
    if not plan_e_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not vectorized. Run vectorization first."
        )
    
    # Find graph file
    graph_file = find_viz_graph_file(plan_e_dir)
    if not graph_file or not graph_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Graph file not found"
        )
    
    # Load graph
    try:
        G = load_graph(graph_file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load graph: {str(e)}"
        )
    
    # Create output directory
    viz_dir = plan_e_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    graph_name = graph_file.stem.replace("_document_graph", "")
    
    try:
        if viz_type == "interactive":
            output_file = viz_dir / f"{graph_name}_interactive.html"
            visualize_interactive(G, output_file)
            return FileResponse(
                output_file,
                media_type="text/html",
                filename=output_file.name,
            )
        elif viz_type == "static":
            output_file = viz_dir / f"{graph_name}_full.png"
            visualize_static(G, output_file, layout="spring")
            return FileResponse(
                output_file,
                media_type="image/png",
                filename=output_file.name,
            )
        elif viz_type == "simplified":
            output_file = viz_dir / f"{graph_name}_simplified.png"
            visualize_simplified(G, output_file, max_nodes=150)
            return FileResponse(
                output_file,
                media_type="image/png",
                filename=output_file.name,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown visualization type: {viz_type}"
            )
    except Exception as e:
        logger.error(f"Error creating visualization: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Visualization failed: {str(e)}"
        )


@app.get("/visualize/{document_id}/stats", tags=["Visualization"])
async def get_graph_stats(
    document_id: str = PathParam(..., description="Document ID"),
    current_user=Depends(get_current_user),
):
    """Get graph statistics"""
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
        return document_service.get_graph_stats(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/page/{document_id}/summarize", response_model=PageSummaryResponse, tags=["Page Analysis"])
async def summarize_page(
    document_id: str = PathParam(..., description="Document ID"),
    page_number: int = Query(..., description="Page number to summarize", gt=0),
    current_user=Depends(get_current_user),
):
    """Generate comprehensive page-level summary and explanation"""
    try:
        document_service.ensure_document_belongs_to_user(document_id, current_user.id)
        result = document_service.summarize_page(document_id, page_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error summarizing page {page_number} for document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return PageSummaryResponse(
        page_number=result["page_number"],
        summary=result["summary"],
        key_points=result["key_points"],
        sections=result["sections"],
        has_content=result["has_content"],
        used_adjacent_pages=result["used_adjacent_pages"],
        adjacent_pages_used=result.get("adjacent_pages_used"),
        page_classification=result.get("page_classification"),
        chunks_used=result["chunks_used"],
        total_chunks=result["total_chunks"],
    )


# ---------------- ERROR HANDLERS ----------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


# ---------------- MAIN ----------------


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
