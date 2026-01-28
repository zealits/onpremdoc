"""
FastAPI Application for Document Processing Pipeline
Integrates PDF detection, vectorization, retrieval, and visualization
"""

import asyncio
import json
import logging
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
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Import modules
from detection import process_single_pdf, extract_page_mapping_from_markdown
from vectorizerE import (
    create_vectorization_workflow,
    VectorizerState,
    DocumentGraph,
    check_ollama_running,
)
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
from visualizeGraphE import (
    load_graph,
    visualize_interactive,
    visualize_static,
    visualize_simplified,
    print_graph_stats,
    find_graph_file as find_viz_graph_file,
)
from page_summarization import (
    load_page_agent,
    PageSummarizationAgent,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global state for loaded agents (document_id -> agent resources)
_loaded_agents: Dict[str, Dict[str, Any]] = {}

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
    content: str
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


class QueryResponse(BaseModel):
    """Query response with detailed chunk information"""
    answer: str
    document_id: str
    query: str
    retrieval_stats: Dict[str, Any]
    chunk_analysis: Optional[str] = None
    chunks: List[ChunkDetail] = Field(default_factory=list, description="Detailed information about chunks used to generate the answer")
    debug_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Debugging information about retrieval process")


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


# ---------------- LIFECYCLE MANAGEMENT ----------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")
    
    # Check Ollama on startup
    is_running, models = check_ollama_running()
    if not is_running:
        logger.warning("Ollama server not detected. Some features may not work.")
    else:
        logger.info(f"Ollama server detected with {len(models)} models")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")
    _loaded_agents.clear()


# ---------------- FASTAPI APP ----------------


app = FastAPI(
    title="Document Processing API",
    description="API for PDF processing, vectorization, graph-based retrieval, and visualization",
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


# ---------------- HELPER FUNCTIONS ----------------


def get_document_path(document_id: str) -> Path:
    """Get document output directory"""
    return OUTPUT_DIR / document_id


def get_document_info(document_id: str) -> Optional[DocumentInfo]:
    """Get document information"""
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        return None
    
    # Find markdown file
    md_files = list(doc_path.glob("*.md"))
    md_path = md_files[0] if md_files else None
    
    # Find page mapping
    page_mapping_path = None
    if md_path:
        page_mapping_path = doc_path / f"{md_path.stem}_page_mapping.json"
        if not page_mapping_path.exists():
            page_mapping_path = None

    # Find confidence report
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
                # If anything goes wrong, just skip confidence summary
                confidence_path = None
    
    # Check Plan E folder
    plan_e_dir = doc_path / "E"
    vector_mapping_path = None
    graph_path = None
    vector_db_path = None
    total_chunks = None
    graph_nodes = None
    graph_edges = None
    
    if plan_e_dir.exists():
        # Find vector mapping
        mapping_file = find_vector_mapping_file(plan_e_dir)
        if mapping_file:
            vector_mapping_path = str(mapping_file)
            # Count chunks
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                    total_chunks = len(mapping_data)
            except:
                pass
        
        # Find graph
        graph_file = find_graph_file(plan_e_dir)
        if graph_file:
            graph_path = str(graph_file)
            # Count nodes/edges
            try:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    graph_nodes = len(graph_data.get("nodes", []))
                    graph_edges = len(graph_data.get("edges", []))
            except:
                pass
        
        # Find vector DB
        vector_db = find_vector_db_path(plan_e_dir)
        if vector_db:
            vector_db_path = str(vector_db)
    
    # Determine status
    status = "uploaded"
    if md_path:
        status = "processing"
    if vector_mapping_path and graph_path:
        status = "vectorized"
    if vector_mapping_path and graph_path and vector_db_path:
        status = "ready"
    
    # Get total pages
    total_pages = None
    if page_mapping_path:
        try:
            with open(page_mapping_path, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                total_pages = page_data.get("total_pages")
        except:
            pass
    
    return DocumentInfo(
        document_id=document_id,
        name=doc_path.name,
        status=status,
        markdown_path=str(md_path) if md_path else None,
        page_mapping_path=str(page_mapping_path) if page_mapping_path else None,
        vector_mapping_path=vector_mapping_path,
        graph_path=graph_path,
        vector_db_path=vector_db_path,
        confidence_path=str(confidence_path) if confidence_path else None,
        total_pages=total_pages,
        total_chunks=total_chunks,
        layout_score=layout_score,
        ocr_score=ocr_score,
        parse_score=parse_score,
        table_score=table_score,
        mean_grade=mean_grade,
        low_grade=low_grade,
    )


def load_agent_for_document(document_id: str) -> Dict[str, Any]:
    """Load agent resources for a document"""
    if document_id in _loaded_agents:
        return _loaded_agents[document_id]
    
    doc_path = get_document_path(document_id)
    plan_e_dir = doc_path / "E"
    
    if not plan_e_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not vectorized. Run vectorization first."
        )
    
    # Find required files
    vector_mapping_file = find_vector_mapping_file(plan_e_dir)
    graph_file = find_graph_file(plan_e_dir)
    vector_db_path = find_vector_db_path(plan_e_dir)
    
    if not vector_mapping_file or not vector_mapping_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Vector mapping file not found"
        )
    
    if not graph_file or not graph_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Graph file not found"
        )
    
    if not vector_db_path or not vector_db_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Vector database not found"
        )
    
    # Load resources
    from retrivalAgentE import (
        load_chunks_from_mapping,
        load_vector_store,
        Ollama,
        EMBEDDING_MODEL,
        LLM_MODEL,
        OLLAMA_BASE_URL,
    )
    
    chunks = load_chunks_from_mapping(vector_mapping_file)
    document_graph = RetrievalDocumentGraph()
    document_graph.load(graph_file)
    vector_store = load_vector_store(vector_db_path)
    
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3
    )
    
    # Create agent (pass document folder for page summarization)
    agent = create_retrieval_agent(vector_store, document_graph, chunks, llm, doc_path)
    
    # Set global resources (pass document folder for page summarization)
    set_agent_resources(vector_store, document_graph, chunks, llm, doc_path)
    
    # Cache
    _loaded_agents[document_id] = {
        "agent": agent,
        "vector_store": vector_store,
        "document_graph": document_graph,
        "chunks": chunks,
        "llm": llm,
    }
    
    return _loaded_agents[document_id]


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
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    is_running, models = check_ollama_running()
    return HealthResponse(
        status="healthy" if is_running else "degraded",
        ollama_available=is_running,
        ollama_models=models,
    )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all processed documents"""
    documents = []
    if OUTPUT_DIR.exists():
        for doc_dir in OUTPUT_DIR.iterdir():
            if doc_dir.is_dir():
                doc_info = get_document_info(doc_dir.name)
                if doc_info:
                    documents.append(doc_info)
    return documents


@app.get("/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document(document_id: str = PathParam(..., description="Document ID")):
    """Get document information"""
    doc_info = get_document_info(document_id)
    if not doc_info:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    return doc_info


@app.get("/documents/{document_id}/markdown", tags=["Documents"])
async def get_markdown(document_id: str = PathParam(..., description="Document ID")):
    """Get markdown content for a document"""
    doc_info = get_document_info(document_id)
    if not doc_info or not doc_info.markdown_path:
        raise HTTPException(
            status_code=404,
            detail="Markdown file not found"
        )
    
    markdown_path = Path(doc_info.markdown_path)
    if not markdown_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Markdown file not found"
        )
    
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    from fastapi.responses import Response
    return Response(content=content, media_type="text/plain")


@app.get("/documents/{document_id}/confidence", response_model=ConfidenceResponse, tags=["Documents"])
async def get_confidence(document_id: str = PathParam(..., description="Document ID")):
    """Get Docling confidence / accuracy scores for a document"""
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


@app.post("/upload", response_model=ProcessPDFResponse, tags=["Documents"])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Upload and process a PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Generate document ID
    document_id = str(uuid4())
    doc_path = get_document_path(document_id)
    doc_path.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    pdf_path = doc_path / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process in background
    background_tasks.add_task(process_pdf_background, str(pdf_path), document_id)
    
    return ProcessPDFResponse(
        document_id=document_id,
        status="processing",
        message="PDF uploaded and processing started",
    )


def process_pdf_background(pdf_path: str, document_id: str):
    """Background task to process PDF"""
    pdf_file = Path(pdf_path)
    doc_path = get_document_path(document_id)
    
    try:
        logger.info(f"Processing PDF: {pdf_path} for document {document_id}")
        
        # Process PDF directly in the document_id folder - no temporary folder needed
        process_single_pdf(pdf_file, OUTPUT_DIR, target_dir=doc_path)
        
        logger.info(f"PDF processing complete for document {document_id}")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)


@app.post("/vectorize/{document_id}", response_model=VectorizeResponse, tags=["Processing"])
async def vectorize_document(
    document_id: str = PathParam(..., description="Document ID"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Vectorize a processed document"""
    doc_path = get_document_path(document_id)
    if not doc_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    
    # Check if markdown exists
    md_files = list(doc_path.glob("*.md"))
    if not md_files:
        raise HTTPException(
            status_code=400,
            detail="Document must be processed first (markdown not found)"
        )
    
    # Process in background
    background_tasks.add_task(vectorize_background, document_id)
    
    return VectorizeResponse(
        document_id=document_id,
        status="processing",
        message="Vectorization started",
    )


def vectorize_background(document_id: str):
    """Background task to vectorize document"""
    try:
        logger.info(f"{'='*80}")
        logger.info(f"🚀 Starting vectorization for document: {document_id}")
        logger.info(f"{'='*80}")
        
        doc_path = get_document_path(document_id)
        logger.info(f"📁 Document path: {doc_path}")
        
        # Check if markdown exists
        md_files = list(doc_path.glob("*.md"))
        if not md_files:
            logger.error(f"❌ No markdown file found in {doc_path}")
            logger.error(f"   Please ensure PDF processing completed successfully")
            return
        
        logger.info(f"✅ Found markdown file: {md_files[0].name}")
        
        # Create workflow state
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
        }
        
        logger.info(f"📊 Creating vectorization workflow...")
        
        # Run workflow
        workflow = create_vectorization_workflow()
        logger.info(f"⚙️  Running vectorization workflow...")
        logger.info(f"   This may take several minutes depending on document size...")
        
        final_state = workflow.invoke(initial_state)
        
        # Log results
        total_chunks = len(final_state.get("json_mapping", []))
        graph_nodes = len(final_state.get("document_graph", DocumentGraph()).graph.nodes)
        graph_edges = len(final_state.get("document_graph", DocumentGraph()).graph.edges)
        
        logger.info(f"{'='*80}")
        logger.info(f"✅ Vectorization complete for document {document_id}")
        logger.info(f"   📦 Total chunks: {total_chunks}")
        logger.info(f"   🔗 Graph nodes: {graph_nodes}")
        logger.info(f"   🔗 Graph edges: {graph_edges}")
        logger.info(f"   📂 Output directory: {doc_path / 'E'}")
        logger.info(f"{'='*80}")
    except Exception as e:
        logger.error(f"{'='*80}")
        logger.error(f"❌ Error vectorizing document {document_id}: {e}")
        logger.error(f"{'='*80}")
        logger.error(f"Error details:", exc_info=True)


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_document(request: QueryRequest):
    """Query a vectorized document using graph-enhanced retrieval"""
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    # Load agent for document
    try:
        agent_resources = load_agent_for_document(request.document_id)
        agent = agent_resources["agent"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load agent: {str(e)}"
        )
    
    # Initialize state
    initial_state: AgentState = {
        "query": request.query,
        "is_page_summary": False,
        "page_number": None,
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
    }
    
    # Run agent
    try:
        final_state = agent.invoke(initial_state)
        
        # Build retrieval stats - use scores from state if available
        seed_chunk_ids = set(final_state.get("seed_chunk_ids", []))
        graph_expanded_ids = set(final_state.get("graph_expanded_ids", []))
        second_seed_ids = set(final_state.get("second_seed_ids", []))
        second_expanded_ids = set(final_state.get("second_expanded_ids", []))
        
        # Get similarity scores from state (already calculated during retrieval)
        seed_chunk_scores = final_state.get("seed_chunk_scores", {})
        second_seed_scores = final_state.get("second_seed_scores", {})
        rerank_scores = final_state.get("rerank_scores", {})
        
        # Calculate stats
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
        
        # Use similarity scores from state (already calculated)
        seed_score_map = seed_chunk_scores
        second_seed_score_map = second_seed_scores

        # Extract detailed chunk information
        chunks_detail = []
        if request.include_chunks:
            # Process initial retrieved chunks
            initial_chunks = final_state.get("retrieved_chunks", [])
            for chunk in initial_chunks:
                chunk_id = chunk.metadata.get("chunk_index")
                if chunk_id is None:
                    continue
                
                # Determine retrieval source
                if chunk_id in seed_chunk_ids:
                    source = "seed"
                elif chunk_id in graph_expanded_ids:
                    source = "graph_expanded"
                else:
                    source = "initial"
                
                chunk_detail = ChunkDetail(
                    chunk_index=chunk_id,
                    content=chunk.page_content,
                    heading=chunk.metadata.get("heading", "No heading"),
                    section_path=chunk.metadata.get("section_path", ""),
                    section_title=chunk.metadata.get("section_title", ""),
                    page_number=chunk.metadata.get("page_number") if chunk.metadata.get("page_number") else None,
                    page_classification=chunk.metadata.get("page_classification") if chunk.metadata.get("page_classification") else None,
                    summary=chunk.metadata.get("summary", ""),
                    chunk_type=chunk.metadata.get("chunk_type", "text"),
                    has_table=chunk.metadata.get("has_table", False),
                    table_context=chunk.metadata.get("table_context") if chunk.metadata.get("table_context") else None,
                    start_line=chunk.metadata.get("start_line") if chunk.metadata.get("start_line") else None,
                    content_length=len(chunk.page_content),
                    retrieval_source=source,
                    similarity_score=seed_score_map.get(chunk_id),
                    rerank_score=rerank_scores.get(chunk_id),
                )
                chunks_detail.append(chunk_detail)
            
            # Process second retrieval chunks
            second_chunks = final_state.get("second_retrieval_chunks", [])
            for chunk in second_chunks:
                chunk_id = chunk.metadata.get("chunk_index")
                if chunk_id is None:
                    continue
                
                # Skip if already added (from initial retrieval)
                if any(c.chunk_index == chunk_id for c in chunks_detail):
                    continue
                
                # Determine retrieval source
                if chunk_id in second_seed_ids:
                    source = "second_seed"
                elif chunk_id in second_expanded_ids:
                    source = "second_expanded"
                else:
                    source = "second_retrieval"
                
                chunk_detail = ChunkDetail(
                    chunk_index=chunk_id,
                    content=chunk.page_content,
                    heading=chunk.metadata.get("heading", "No heading"),
                    section_path=chunk.metadata.get("section_path", ""),
                    section_title=chunk.metadata.get("section_title", ""),
                    page_number=chunk.metadata.get("page_number") if chunk.metadata.get("page_number") else None,
                    page_classification=chunk.metadata.get("page_classification") if chunk.metadata.get("page_classification") else None,
                    summary=chunk.metadata.get("summary", ""),
                    chunk_type=chunk.metadata.get("chunk_type", "text"),
                    has_table=chunk.metadata.get("has_table", False),
                    table_context=chunk.metadata.get("table_context") if chunk.metadata.get("table_context") else None,
                    start_line=chunk.metadata.get("start_line") if chunk.metadata.get("start_line") else None,
                    content_length=len(chunk.page_content),
                    retrieval_source=source,
                    similarity_score=second_seed_score_map.get(chunk_id),
                    rerank_score=rerank_scores.get(chunk_id),  # Second retrieval chunks may also have rerank scores
                )
                chunks_detail.append(chunk_detail)
            
            # Sort chunks by chunk_index for consistent ordering
            chunks_detail.sort(key=lambda x: x.chunk_index)
        
        # Get debug info from state
        debug_info = final_state.get("debug_info", {})
        
        # Add rerank scores to debug info if available
        if rerank_scores:
            debug_info["rerank_scores_summary"] = {
                "total_reranked": len(rerank_scores),
                "avg_score": sum(rerank_scores.values()) / len(rerank_scores) if rerank_scores else 0,
                "max_score": max(rerank_scores.values()) if rerank_scores else 0,
                "min_score": min(rerank_scores.values()) if rerank_scores else 0,
            }
        
        return QueryResponse(
            answer=final_state.get("final_answer", "No answer generated"),
            document_id=request.document_id,
            query=request.query,
            retrieval_stats=retrieval_stats,
            chunk_analysis=final_state.get("chunk_analysis"),
            chunks=chunks_detail,
            debug_info=debug_info,
        )
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/visualize/{document_id}", tags=["Visualization"])
async def visualize_graph(
    document_id: str = PathParam(..., description="Document ID"),
    viz_type: str = Query(default="interactive", description="Type: interactive, static, simplified"),
):
    """Generate graph visualization"""
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
async def get_graph_stats(document_id: str = PathParam(..., description="Document ID")):
    """Get graph statistics"""
    doc_path = get_document_path(document_id)
    
    if not doc_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    
    plan_e_dir = doc_path / "E"
    
    if not plan_e_dir.exists():
        # Return empty stats instead of 404 - document exists but not vectorized yet
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_relations": {},
            "density": 0.0,
        }
    
    graph_file = find_viz_graph_file(plan_e_dir)
    if not graph_file or not graph_file.exists():
        # Return empty stats instead of 404 - vectorization may be in progress
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_relations": {},
            "density": 0.0,
        }
    
    try:
        G = load_graph(graph_file)
        
        # Collect stats
        node_types = {}
        for node_id in G.nodes():
            node_type = G.nodes[node_id].get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        edge_relations = {}
        for source, target, edge_data in G.edges(data=True):
            relation = edge_data.get("relation", "unknown")
            edge_relations[relation] = edge_relations.get(relation, 0) + 1
        
        stats = {
            "total_nodes": len(G.nodes),
            "total_edges": len(G.edges),
            "node_types": node_types,
            "edge_relations": edge_relations,
            "density": float(nx.density(G)) if len(G.nodes) > 0 else 0.0,
        }
        
        # Similarity edge stats
        if "similar_to" in edge_relations:
            similarity_edges = [
                (u, v, d) for u, v, d in G.edges(data=True)
                if d.get("relation") == "similar_to"
            ]
            if similarity_edges:
                similarities = [
                    d.get("similarity", 0.0)
                    for u, v, d in similarity_edges
                    if d.get("similarity")
                ]
                if similarities:
                    stats["similarity_stats"] = {
                        "count": len(similarities),
                        "average": float(sum(similarities) / len(similarities)),
                        "min": float(min(similarities)),
                        "max": float(max(similarities)),
                    }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.post("/page/{document_id}/summarize", response_model=PageSummaryResponse, tags=["Page Analysis"])
async def summarize_page(
    document_id: str = PathParam(..., description="Document ID"),
    page_number: int = Query(..., description="Page number to summarize", gt=0),
):
    """Generate comprehensive page-level summary and explanation"""
    doc_path = get_document_path(document_id)
    
    if not doc_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    
    plan_e_dir = doc_path / "E"
    if not plan_e_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not vectorized. Run vectorization first."
        )
    
    try:
        # Load page summarization agent
        agent = load_page_agent(doc_path)
        if not agent:
            raise HTTPException(
                status_code=500,
                detail="Failed to load page summarization agent"
            )
        
        # Generate summary
        summary_result = agent.summarize_page(page_number, use_adjacent_if_empty=True)
        
        # Convert to response model
        return PageSummaryResponse(
            page_number=summary_result.page_number,
            summary=summary_result.summary,
            key_points=summary_result.key_points,
            sections=summary_result.sections,
            has_content=summary_result.has_content,
            used_adjacent_pages=summary_result.used_adjacent_pages,
            adjacent_pages_used=summary_result.adjacent_pages_used,
            page_classification=summary_result.page_classification,
            chunks_used=summary_result.chunks_used,
            total_chunks=len(summary_result.chunks_used),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing page {page_number} for document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to summarize page: {str(e)}"
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
