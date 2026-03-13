"""
RetrievalAgentE - Plan E Implementation: LangGraph Agent with Graph-Enhanced Retrieval
Uses LangGraph to create an agent that uses graph traversal for intelligent retrieval.
Expands from seed chunks via graph edges to capture document relationships.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

# Graph library
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.error("networkx is required. Install with: pip install networkx")
    sys.exit(1)

# LangChain imports
try:
    # Preferred new package (avoids deprecation warning)
    from langchain_chroma import Chroma
except ImportError:  # Fallback for environments without langchain-chroma installed yet
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from config.inference_config import get_embeddings, get_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)

# ---------------- RETRIEVAL CONFIG ----------------
# All numeric constants for the retrieval pipeline live here so behavior is easy to tune.
INITIAL_VECTOR_SEARCH_K = 5          # Top-K for initial vector search
NUM_SEEDS_TO_EXPAND = 2              # How many top seeds to expand
EXPANSION_PER_SEED = 10              # Target expanded neighbors per seed (before rerank)
RERANK_TOP_K = 5                     # Top-K per seed after vector rerank
RERANK_SEARCH_K = 64                 # How many neighbors to look at when reranking
FINAL_CHUNKS_FOR_LLM = 15            # Total chunks sent to LLM (initial + expanded)
SECOND_VECTOR_SEARCH_K = 5           # Top-K for second retrieval vector search
SEED_SIMILARITY_THRESHOLD = 0.3      # Optional seed quality threshold for logging/debug

# ---------------- DOCUMENT GRAPH ---------------- 
class DocumentGraph:
    """Knowledge graph for document structure"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.chunk_nodes = {}
        self.section_nodes = {}
        self.page_nodes = {}
    
    def get_parent_section(self, chunk_id: int) -> Optional[str]:
        """Get the parent section node for a chunk"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return None
        
        # Find section node connected via "belongs_to"
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "section":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "belongs_to":
                    return predecessor
        
        return None
    
    def get_section_chunks(self, section_path: str) -> List[int]:
        """Get all chunk IDs in a section"""
        section_node = self.section_nodes.get(section_path)
        if not section_node:
            return []
        
        chunk_ids = []
        for successor in self.graph.successors(section_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(section_node, successor)
                if edge_data and edge_data.get("relation") == "contains":
                    chunk_id = self.graph.nodes[successor].get("chunk_id")
                    if chunk_id is not None:
                        chunk_ids.append(chunk_id)
        
        return sorted(chunk_ids)
    
    def get_adjacent_chunks(self, chunk_id: int, window: int = 1) -> List[int]:
        """Get adjacent chunks via 'follows' edges"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return []
        
        adjacent_ids = []
        
        # Get previous chunks (incoming "follows" edges)
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "follows":
                    prev_id = self.graph.nodes[predecessor].get("chunk_id")
                    if prev_id is not None:
                        adjacent_ids.append(prev_id)
        
        # Get next chunks (outgoing "follows" edges)
        for successor in self.graph.successors(chunk_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(chunk_node, successor)
                if edge_data and edge_data.get("relation") == "follows":
                    next_id = self.graph.nodes[successor].get("chunk_id")
                    if next_id is not None:
                        adjacent_ids.append(next_id)
        
        return adjacent_ids[:window * 2]
    
    def get_similar_chunks(self, chunk_id: int) -> List[int]:
        """Get chunks connected via 'similar_to' edges"""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return []
        
        similar_ids = []
        
        # Get chunks connected via similarity edges (both directions)
        for successor in self.graph.successors(chunk_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(chunk_node, successor)
                if edge_data and edge_data.get("relation") == "similar_to":
                    similar_id = self.graph.nodes[successor].get("chunk_id")
                    if similar_id is not None:
                        similar_ids.append(similar_id)
        
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "similar_to":
                    similar_id = self.graph.nodes[predecessor].get("chunk_id")
                    if similar_id is not None:
                        similar_ids.append(similar_id)
        
        return similar_ids
    
    def get_page_chunks(self, page_number: int) -> List[int]:
        """Get all chunk IDs on a page"""
        page_node = self.page_nodes.get(page_number)
        if not page_node:
            return []
        
        chunk_ids = []
        for successor in self.graph.successors(page_node):
            if self.graph.nodes[successor].get("type") == "chunk":
                edge_data = self.graph.get_edge_data(page_node, successor)
                if edge_data and edge_data.get("relation") == "on_page":
                    chunk_id = self.graph.nodes[successor].get("chunk_id")
                    if chunk_id is not None:
                        chunk_ids.append(chunk_id)
        
        return sorted(chunk_ids)

    def get_page_for_chunk(self, chunk_id: int) -> Optional[int]:
        """Get page number for a chunk via 'on_page' edges."""
        chunk_node = self.chunk_nodes.get(chunk_id)
        if not chunk_node:
            return None

        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "page":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") == "on_page":
                    return self.graph.nodes[predecessor].get("page_number")

        return None

    def expand_from_single_chunk_prioritized(self, chunk_id: int, max_expansion: int = 10) -> List[int]:
        """
        Expand from a single seed chunk with priority:
        1) similar_to neighbors
        2) chunks in the same section
        3) chunks on the same page
        The seed chunk itself is excluded from the result.
        """
        if max_expansion <= 0:
            return []

        result: List[int] = []
        seen: set[int] = set()

        # Helper to add candidates in order without duplicates and without exceeding max_expansion
        def _add_candidates(candidates: List[int]) -> None:
            for cid in candidates:
                if cid == chunk_id:
                    continue
                if cid in seen:
                    continue
                seen.add(cid)
                result.append(cid)
                if len(result) >= max_expansion:
                    break

        # 1) Similar chunks
        similar_ids = self.get_similar_chunks(chunk_id)
        _add_candidates(similar_ids)
        if len(result) >= max_expansion:
            return result

        # 2) Same section chunks
        section_node = self.get_parent_section(chunk_id)
        if section_node:
            section_path = self.graph.nodes[section_node].get("section_path", "")
            if section_path:
                section_chunks = self.get_section_chunks(section_path)
                _add_candidates(section_chunks)
                if len(result) >= max_expansion:
                    return result

        # 3) Same page chunks
        page_number = self.get_page_for_chunk(chunk_id)
        if page_number is not None:
            page_chunks = self.get_page_chunks(page_number)
            _add_candidates(page_chunks)

        return result[:max_expansion]
    
    def expand_from_chunks(self, chunk_ids: List[int], max_expansion: int = 30) -> List[int]:
        """Expand retrieval by following graph edges from seed chunks"""
        expanded_ids = set(chunk_ids)
        to_process = list(chunk_ids)
        
        while to_process and len(expanded_ids) < max_expansion:
            current_id = to_process.pop(0)
            
            # Get parent section and all chunks in that section
            section_node = self.get_parent_section(current_id)
            if section_node:
                section_path = self.graph.nodes[section_node].get("section_path", "")
                section_chunks = self.get_section_chunks(section_path)
                for chunk_id in section_chunks:
                    if chunk_id not in expanded_ids:
                        expanded_ids.add(chunk_id)
                        to_process.append(chunk_id)
            
            # Get adjacent chunks
            adjacent = self.get_adjacent_chunks(current_id, window=1)
            for adj_id in adjacent:
                if adj_id not in expanded_ids:
                    expanded_ids.add(adj_id)
                    to_process.append(adj_id)
            
            # Get similar chunks via similarity edges
            similar = self.get_similar_chunks(current_id)
            for sim_id in similar:
                if sim_id not in expanded_ids:
                    expanded_ids.add(sim_id)
                    to_process.append(sim_id)
        
        return list(expanded_ids)
    
    def load(self, filepath: Path):
        """Load graph from JSON format"""
        if not filepath.exists():
            logger.warning(f"Graph file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        self.graph.clear()
        self.chunk_nodes.clear()
        self.section_nodes.clear()
        self.page_nodes.clear()
        
        # Add nodes
        for node_data in graph_data.get("nodes", []):
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)
            
            # Rebuild index
            if node_data.get("type") == "chunk":
                chunk_id = node_data.get("chunk_id")
                if chunk_id is not None:
                    self.chunk_nodes[chunk_id] = node_id
            elif node_data.get("type") == "section":
                section_path = node_data.get("section_path", "")
                if section_path:
                    self.section_nodes[section_path] = node_id
            elif node_data.get("type") == "page":
                page_number = node_data.get("page_number")
                if page_number:
                    self.page_nodes[page_number] = node_id
        
        # Add edges
        for edge_data in graph_data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self.graph.add_edge(source, target, **edge_data)
        
        logger.info(f"Loaded graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

# ---------------- STATE DEFINITION ---------------- 
class AgentState(TypedDict):
    """State for the retrieval agent workflow"""
    query: str
    is_page_summary: bool  # Whether query is asking for page summary
    page_number: Optional[int]  # Page number if is_page_summary is True
    seed_chunk_ids: List[int]  # Initial vector search results
    seed_chunk_scores: Dict[int, float]  # Similarity scores for seed chunks
    graph_expanded_ids: List[int]  # Chunks expanded via graph
    retrieved_chunks: List[Document]  # All retrieved chunks
    reranked_chunks: Optional[List[Document]]  # Re-ranked chunks by relevance
    rerank_scores: Optional[Dict[int, float]]  # Re-ranking scores
    chunk_analysis: str
    needs_more_info: bool
    new_query: Optional[str]
    second_seed_ids: List[int]
    second_seed_scores: Dict[int, float]  # Similarity scores for second seed chunks
    second_expanded_ids: List[int]
    second_retrieval_chunks: List[Document]
    final_answer: str
    iteration_count: int
    document_folder: Optional[str]  # Path to document folder for page summarization
    debug_info: Dict[str, Any]  # Debugging information
    token_usage: Optional[List[Dict[str, Any]]]  # Per-step token counts for economics reporting
    # Suggested follow-up questions based on the original query and final answer
    next_questions: Optional[List[str]]
    # When True, generate_final_answer only builds the prompt and sets answer_prompt (no LLM call).
    streaming_mode: Optional[bool]
    # Filled by generate_final_answer when streaming_mode is True; API then streams via llm.stream().
    answer_prompt: Optional[str]
    # Chunk indices used for page summarization (so API can return them in response).
    page_summary_chunk_indices: Optional[List[int]]
    # Past N messages for conversation context: list of {role, content, chunks?}.
    past_messages: Optional[List[Dict[str, Any]]]
    # When True, answer from session context only (no retrieval).
    is_use_history: Optional[bool]
    # Chunk indices that most strongly support the final answer (post-hoc grounding).
    chunk_indices_used_for_answer: Optional[List[int]]

# ---------------- TOKEN USAGE HELPERS (economics) ----------------
def _est_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(str(text).strip()) // 4)

def _extract_stream_delta(chunk: Any) -> str:
    """Extract text delta from a streaming LLM chunk (LangChain / OpenAI style)."""
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
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return delta["content"]
    return ""


def _append_token_usage(
    state: AgentState,
    step: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one step's token usage to state for economics logging."""
    usage = state.get("token_usage") or []
    rec = {"step": step, "input_tokens": input_tokens, "output_tokens": output_tokens, "embedding_tokens": embedding_tokens}
    if extra:
        rec["extra"] = extra
    usage.append(rec)
    state["token_usage"] = usage

# ---------------- LOAD FUNCTIONS ---------------- 
def load_chunks_from_mapping(mapping_file: Path) -> List[Document]:
    """Load chunks from vector mapping JSON file"""
    if not mapping_file.exists():
        logger.error(f"Vector mapping file not found: {mapping_file}")
        return []
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    chunks = []
    for item in mapping_data:
        raw_content = item.get("content", "")
        content_str = "\n".join(raw_content) if isinstance(raw_content, list) else raw_content
        # Preserve the exact content representation from the mapping for downstream consumers
        metadata = item.get("metadata", {}).copy()
        metadata["raw_content_lines"] = raw_content
        doc = Document(
            page_content=content_str,
            metadata=metadata
        )
        chunks.append(doc)
    
    logger.info(f"Loaded {len(chunks)} chunks from mapping file")
    return chunks

def load_vector_store(vector_db_path: Path) -> Chroma:
    """Load Chroma vector store (uses inference_config for embeddings)."""
    if not vector_db_path.exists():
        logger.error(f"Vector DB path not found: {vector_db_path}")
        sys.exit(1)
    embeddings = get_embeddings()
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(vector_db_path)
    )
    logger.info(f"Loaded vector store from: {vector_db_path}")
    return vector_store

# Global stores
_vector_store: Optional[Chroma] = None
_document_graph: Optional[DocumentGraph] = None
_chunks: List[Document] = []
_llm: Optional[Any] = None  # BaseChatModel (Ollama or Hugging Face)
_page_agent: Optional[Any] = None  # PageSummarizationAgent
_document_folder: Optional[Path] = None

def set_agent_resources(vector_store: Chroma, document_graph: DocumentGraph, chunks: List[Document], llm: Any, document_folder: Optional[Path] = None):
    """Set global resources for agent nodes"""
    global _vector_store, _document_graph, _chunks, _llm, _page_agent, _document_folder
    _vector_store = vector_store
    _document_graph = document_graph
    _chunks = chunks
    _llm = llm
    _document_folder = document_folder
    
    # Load page summarization agent if document folder is provided (reuse same LLM)
    if document_folder:
        try:
            from page_summarization import load_page_agent
            _page_agent = load_page_agent(document_folder, llm=llm)
            if _page_agent:
                logger.info("Page summarization agent loaded successfully")
            else:
                logger.warning("Failed to load page summarization agent")
        except Exception as e:
            logger.warning(f"Could not load page summarization agent: {e}")
            _page_agent = None

# ---------------- QUERY CLASSIFICATION ----------------
def _format_past_for_classifier(past: List[Dict[str, Any]], max_turns: int = 5, max_content_len: int = 600) -> str:
    """Format past messages for the classifier so it can see if the current query is already answered."""
    if not past:
        return "No previous conversation."
    lines = []
    for entry in past[-max_turns:]:
        role = entry.get("role", "")
        content = (entry.get("content") or "")[:max_content_len]
        if len(entry.get("content") or "") > max_content_len:
            content += "..."
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def classify_query(state: AgentState) -> AgentState:
    """
    Classify the query using the LLM with a single general prompt. No heuristics.
    Three outcomes: FULL_RETRIEVAL, USE_HISTORY, PAGE_SUMMARY.
    """
    query = (state.get("query") or "").strip()
    llm = _llm
    past = state.get("past_messages") or []

    logger.info("Classifying query type...")

    past_context = _format_past_for_classifier(past)

    classification_prompt = f"""You are a classifier for a document Q&A system. You must choose exactly one of three options based on the current query and the previous conversation.

Previous conversation:
{past_context}

Current query: {query}

Options:

FULL_RETRIEVAL — Choose when the current query cannot be fully answered from the previous conversation. The system will then search the document to answer it.

USE_HISTORY — Choose when the current query can be answered from the previous conversation.

PAGE_SUMMARY — Choose when the user is asking for a summary or explanation of a specific page of the document. You must then provide the page number.

Reply with exactly two lines in this format:
CLASS: FULL_RETRIEVAL
PAGE_NUMBER: None

or

CLASS: USE_HISTORY
PAGE_NUMBER: None

or

CLASS: PAGE_SUMMARY
PAGE_NUMBER: <integer>"""

    try:
        response = llm.invoke(classification_prompt)
        response_text = (getattr(response, "content", None) or str(response)).strip()
        _append_token_usage(
            state, "query_classification",
            input_tokens=_est_tokens(classification_prompt),
            output_tokens=_est_tokens(response_text),
        )

        state["is_page_summary"] = False
        state["page_number"] = None
        state["is_use_history"] = False

        response_upper = response_text.upper()

        # Parse CLASS from response
        class_val = "FULL_RETRIEVAL"
        if "CLASS:" in response_upper:
            for line in response_text.split("\n"):
                if "CLASS:" in line.upper():
                    if "PAGE_SUMMARY" in line.upper():
                        class_val = "PAGE_SUMMARY"
                    elif "USE_HISTORY" in line.upper():
                        class_val = "USE_HISTORY"
                    else:
                        class_val = "FULL_RETRIEVAL"
                    break

        # Parse PAGE_NUMBER when CLASS is PAGE_SUMMARY
        page_num = None
        if class_val == "PAGE_SUMMARY" and "PAGE_NUMBER:" in response_upper:
            for line in response_text.split("\n"):
                if "PAGE_NUMBER:" in line.upper():
                    rest = line.split(":", 1)[-1].strip() if ":" in line else line
                    num_match = re.search(r"\d+", rest)
                    if num_match:
                        page_num = int(num_match.group())
                    break

        if class_val == "PAGE_SUMMARY" and page_num is not None:
            state["is_page_summary"] = True
            state["page_number"] = page_num
            logger.info(f"Query classified as page summary for page {page_num}")
        elif class_val == "USE_HISTORY":
            state["is_use_history"] = True
            logger.info("Query classified as use_history (answer from session context)")
        else:
            state["is_use_history"] = False
            logger.info("Query classified as full retrieval")

    except Exception as e:
        logger.error(f"Error classifying query: {e}")
        state["is_page_summary"] = False
        state["page_number"] = None
        state["is_use_history"] = False

    return state


# ---------------- PAGE SUMMARIZATION ---------------- 
def summarize_page(state: AgentState) -> AgentState:
    """Generate page summary using PageSummarizationAgent"""
    page_number = state.get("page_number")
    page_agent = _page_agent
    
    if not page_number:
        state["final_answer"] = "Error: No page number specified for page summary."
        return state
    
    if not page_agent:
        state["final_answer"] = f"Error: Page summarization is not available. Cannot summarize page {page_number}."
        return state
    
    logger.info(f"Generating summary for page {page_number}...")

    # When streaming, only build the prompt; API will stream via llm.stream().
    if state.get("streaming_mode"):
        result = page_agent.get_summary_prompt(page_number, use_adjacent_if_empty=True)
        if result:
            prompt, chunk_indices = result
            # Page summarization: prepend past conversation context (content only, no chunks)
            past = state.get("past_messages") or []
            if past:
                conv_block = _format_past_messages_for_prompt(past, include_chunks=False) + "\n\n"
                prompt = conv_block + prompt
            state["answer_prompt"] = prompt
            state["page_summary_chunk_indices"] = chunk_indices
            state["final_answer"] = ""
        else:
            state["page_summary_chunk_indices"] = []
            state["final_answer"] = "No content available for this page. The page may be blank, contain only images, or have insufficient text for processing."
        return state

    try:
        page_summary = page_agent.summarize_page(page_number, use_adjacent_if_empty=True)

        # Format the response
        answer_parts = []

        if page_summary.page_classification:
            answer_parts.append(f"**Page {page_number} Classification:** {page_summary.page_classification}\n")

        answer_parts.append(f"**Summary:**\n{page_summary.summary}\n")

        if page_summary.key_points:
            answer_parts.append("**Key Points:**")
            for point in page_summary.key_points:
                answer_parts.append(f"- {point}")
            answer_parts.append("")

        if page_summary.used_adjacent_pages and page_summary.adjacent_pages_used:
            answer_parts.append(f"\n*Note: This summary is based on adjacent pages {page_summary.adjacent_pages_used} as page {page_number} had limited direct content.*")

        state["final_answer"] = "\n".join(answer_parts)
        state["page_summary_chunk_indices"] = page_summary.chunks_used if page_summary.chunks_used else []
        logger.info(f"Page summary generated successfully for page {page_number}")

    except Exception as e:
        logger.error(f"Error generating page summary: {e}", exc_info=True)
        state["final_answer"] = f"Error generating summary for page {page_number}: {str(e)}"
        state["page_summary_chunk_indices"] = []

    return state


def _format_past_messages_for_prompt(
    past_messages: List[Dict[str, Any]], include_chunks: bool = True
) -> str:
    """Format past_messages as a string for the LLM. If include_chunks=False (page summarization), only content."""
    if not past_messages:
        return ""
    lines = ["Previous conversation:"]
    for entry in past_messages:
        role = entry.get("role", "")
        content = (entry.get("content") or "").strip()
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")
            if include_chunks:
                chunks = entry.get("chunks") or []
                for c in chunks[:2]:
                    cid = c.get("chunk_index")
                    text = c.get("content")
                    if isinstance(text, list):
                        text = "\n".join(str(t) for t in text)
                    text = (text or "")[:400] + ("..." if len(str(text or "")) > 400 else "")
                    if cid is not None and text:
                        lines.append(f"  [Chunk {cid}]: {text}")
    return "\n".join(lines)


def answer_from_history(state: AgentState) -> AgentState:
    """Answer from session context only (no retrieval). Uses past_messages + current query."""
    query = (state.get("query") or "").strip()
    past = state.get("past_messages") or []
    llm = _llm

    if not past:
        state["final_answer"] = "I don't have any previous messages in this conversation to refer to. Please ask a question about the document."
        return state

    context_str = _format_past_messages_for_prompt(past, include_chunks=True)

    answer_prompt = f"""You are answering a follow-up question based only on the conversation below. Do not use any external knowledge.

{context_str}

Current question: {query}

Answer concisely based only on the previous conversation above. If the question cannot be answered from the context, say so."""

    if state.get("streaming_mode"):
        state["answer_prompt"] = answer_prompt
        state["final_answer"] = ""
        return state

    try:
        response = llm.invoke(answer_prompt)
        response_str = (getattr(response, "content", None) or str(response)).strip()
        state["final_answer"] = response_str
        _append_token_usage(
            state, "answer_from_history",
            input_tokens=_est_tokens(answer_prompt),
            output_tokens=_est_tokens(response_str),
        )
    except Exception as e:
        logger.error(f"Error in answer_from_history: {e}", exc_info=True)
        state["final_answer"] = "I couldn't generate an answer from the conversation context. Please try again."
    return state


# ---------------- RETRIEVAL FUNCTIONS ----------------
def distance_to_similarity(distance: float, scale_factor: float = 150.0) -> float:
    """Convert L2 distance to similarity score [0, 1]"""
    similarity = 1.0 / (1.0 + (distance / scale_factor))
    return max(0.0, min(1.0, similarity))

def initial_retrieval(state: AgentState) -> AgentState:
    """Initial retrieval: vector search + prioritized expansion for top seeds + vector rerank."""
    query = state["query"]
    vector_store = _vector_store
    document_graph = _document_graph
    
    logger.info(f"Initial graph-enhanced retrieval for query: '{query}'")
    
    # Step 1: Vector similarity search (get seed chunks)
    seed_chunk_ids: List[int] = []
    seed_chunk_scores: Dict[int, float] = {}
    seed_chunks_with_scores: List[tuple[int, float, float]] = []  # (chunk_id, similarity_score, distance)

    try:
        vector_results = vector_store.similarity_search_with_score(query, k=INITIAL_VECTOR_SEARCH_K)
        for doc, distance in vector_results:
            chunk_id = doc.metadata.get("chunk_index")
            if chunk_id is not None:
                similarity_score = distance_to_similarity(distance)
                seed_chunks_with_scores.append((chunk_id, similarity_score, distance))
                seed_chunk_ids.append(chunk_id)
                seed_chunk_scores[chunk_id] = similarity_score
                logger.debug(f"  Seed chunk {chunk_id} (distance: {distance:.3f}, similarity: {similarity_score:.3f})")
    except Exception as e:
        logger.warning(f"Vector search error: {e}")

    logger.info(f"Found {len(seed_chunk_ids)} seed chunks from vector search (k={INITIAL_VECTOR_SEARCH_K})")

    # Economics: query embedding for vector search
    _append_token_usage(state, "initial_retrieval", embedding_tokens=_est_tokens(query))

    # Sort by similarity score (descending)
    seed_chunks_with_scores.sort(reverse=True, key=lambda x: x[1])

    # Optional: count high-quality seeds for debug (threshold does not affect behavior directly)
    high_quality_seeds = [chunk_id for chunk_id, score, _ in seed_chunks_with_scores if score >= SEED_SIMILARITY_THRESHOLD]
    logger.info(f"High quality seeds (score >= {SEED_SIMILARITY_THRESHOLD}): {len(high_quality_seeds)}")

    # Initial top-5 seeds used directly as starting context
    initial_top_ids = [chunk_id for chunk_id, _, _ in seed_chunks_with_scores[:INITIAL_VECTOR_SEARCH_K]]

    # Build mapping from chunk_index -> Document
    chunk_dict: Dict[int, Document] = {chunk.metadata.get("chunk_index"): chunk for chunk in _chunks}

    initial_top_docs: List[Document] = []
    for cid in initial_top_ids:
        if cid in chunk_dict:
            initial_top_docs.append(chunk_dict[cid])

    # Prepare rerank scores storage
    rerank_scores: Dict[int, float] = {}

    # Helper: vector-based rerank within a candidate set for a given seed
    def _rerank_for_seed(seed_id: int, expanded_ids: List[int]) -> List[Document]:
        if seed_id is None or not expanded_ids:
            return []

        seed_doc = chunk_dict.get(seed_id)
        if not seed_doc:
            return []

        expanded_set = set(expanded_ids)
        candidates: List[tuple[int, float]] = []  # (chunk_id, similarity)

        try:
            # Use seed content as the query into the same vector space
            search_k = max(RERANK_SEARCH_K, len(expanded_set))
            vr = vector_store.similarity_search_with_score(seed_doc.page_content, k=search_k)
            for doc, distance in vr:
                cid = doc.metadata.get("chunk_index")
                if cid is None or cid not in expanded_set:
                    continue
                sim = distance_to_similarity(distance)
                candidates.append((cid, sim))
        except Exception as exc:
            logger.warning(f"Vector rerank error for seed {seed_id}: {exc}")
            return []

        # Sort candidates by similarity (desc) and take top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_ids: List[int] = []
        for cid, sim in candidates:
            if cid in rerank_scores:
                # Keep the max similarity across seeds if a chunk is reused
                rerank_scores[cid] = max(rerank_scores[cid], sim)
            else:
                rerank_scores[cid] = sim
            top_ids.append(cid)
            if len(top_ids) >= RERANK_TOP_K:
                break

        return [chunk_dict[cid] for cid in top_ids if cid in chunk_dict]

    # Step 2: prioritized expansion and rerank for top seeds
    graph_expanded_ids: List[int] = []
    top_seeds_for_expansion = [chunk_id for chunk_id, _, _ in seed_chunks_with_scores[:NUM_SEEDS_TO_EXPAND]]

    seed1_expanded_docs: List[Document] = []
    seed2_expanded_docs: List[Document] = []

    if top_seeds_for_expansion:
        logger.info(f"Top {NUM_SEEDS_TO_EXPAND} seeds selected for prioritized expansion: {top_seeds_for_expansion}")

        if len(top_seeds_for_expansion) >= 1:
            seed1_id = top_seeds_for_expansion[0]
            expanded_1 = document_graph.expand_from_single_chunk_prioritized(seed1_id, max_expansion=EXPANSION_PER_SEED)
            logger.info(f"Seed {seed1_id}: prioritized expansion produced {len(expanded_1)} candidates")
            graph_expanded_ids.extend(expanded_1)
            seed1_expanded_docs = _rerank_for_seed(seed1_id, expanded_1)

        if len(top_seeds_for_expansion) >= 2:
            seed2_id = top_seeds_for_expansion[1]
            expanded_2 = document_graph.expand_from_single_chunk_prioritized(seed2_id, max_expansion=EXPANSION_PER_SEED)
            logger.info(f"Seed {seed2_id}: prioritized expansion produced {len(expanded_2)} candidates")
            graph_expanded_ids.extend(expanded_2)
            seed2_expanded_docs = _rerank_for_seed(seed2_id, expanded_2)
    else:
        logger.warning("No seeds available for prioritized expansion")

    # Step 3: build final chunk list:
    # first 5 initial chunks + top similar chunks of seed1 + top similar chunks of seed2
    seen_ids: set[int] = set()
    chunks_used: List[Document] = []

    # Initial 5
    for doc in initial_top_docs:
        cid = doc.metadata.get("chunk_index")
        if cid is None or cid in seen_ids:
            continue
        seen_ids.add(cid)
        chunks_used.append(doc)

    # Top from seed 1
    for doc in seed1_expanded_docs:
        cid = doc.metadata.get("chunk_index")
        if cid is None or cid in seen_ids:
            continue
        seen_ids.add(cid)
        chunks_used.append(doc)
        if len(chunks_used) >= FINAL_CHUNKS_FOR_LLM:
            break

    # Top from seed 2
    if len(chunks_used) < FINAL_CHUNKS_FOR_LLM:
        for doc in seed2_expanded_docs:
            cid = doc.metadata.get("chunk_index")
            if cid is None or cid in seen_ids:
                continue
            seen_ids.add(cid)
            chunks_used.append(doc)
            if len(chunks_used) >= FINAL_CHUNKS_FOR_LLM:
                break

    # Enforce final cap
    if len(chunks_used) > FINAL_CHUNKS_FOR_LLM:
        chunks_used = chunks_used[:FINAL_CHUNKS_FOR_LLM]

    logger.info(f"Initial retrieval built {len(chunks_used)} chunks (target {FINAL_CHUNKS_FOR_LLM})")

    # Update state
    state["seed_chunk_ids"] = seed_chunk_ids
    state["seed_chunk_scores"] = seed_chunk_scores
    state["graph_expanded_ids"] = graph_expanded_ids
    state["retrieved_chunks"] = chunks_used
    state["reranked_chunks"] = chunks_used
    state["rerank_scores"] = rerank_scores
    # Store top expanded docs per seed so second retrieval can reuse them
    state["top5_seed1_expanded"] = seed1_expanded_docs
    state["top5_seed2_expanded"] = seed2_expanded_docs
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    # Add debug info
    debug_info = state.get("debug_info", {})
    debug_info["initial_retrieval"] = {
        "total_seed_chunks": len(seed_chunk_ids),
        "high_quality_seeds": len(high_quality_seeds),
        "seeds_used_for_expansion": len(top_seeds_for_expansion),
        "graph_expanded_chunks": len(graph_expanded_ids),
        "total_retrieved": len(chunks_used),
        "total_used": len(chunks_used),
        "similarity_threshold": SEED_SIMILARITY_THRESHOLD,
        "top_seeds_for_expansion": top_seeds_for_expansion,
        "seed_scores": {str(k): round(v, 3) for k, v in list(seed_chunk_scores.items())[:10]},
    }
    state["debug_info"] = debug_info
    
    logger.info(f"Initial retrieval complete: {len(chunks_used)} chunks")
    return state

def analyze_chunks(state: AgentState) -> AgentState:
    """Analyze retrieved chunks to determine if more information is needed"""
    query = state["query"]
    # Use reranked chunks if available (they're already in retrieved_chunks after initial_retrieval)
    retrieved_chunks = state["retrieved_chunks"]
    llm = _llm
    
    logger.info(f"Analyzing {len(retrieved_chunks)} retrieved chunks...")
    
    # Prepare context from chunks (use top chunks, which are already re-ranked)
    chunks_text = []
    for i, chunk in enumerate(retrieved_chunks[:12], 1):
        heading = chunk.metadata.get("heading", "No heading")
        section = chunk.metadata.get("section_path", "No section")
        summary = chunk.metadata.get("summary", "")
        # Use full chunk content so the model can see answers that occur later in long paragraphs
        content_preview = chunk.page_content
        
        chunks_text.append(f"""
Chunk {i}:
- Heading: {heading}
- Section: {section}
- Summary: {summary}
- Content: {content_preview}
""")
    
    context = "\n".join(chunks_text)
    
    # Ask LLM to analyze
    analysis_prompt = f"""You are checking whether the following chunks ALREADY contain a direct answer.

If ANY chunk includes a sentence that explicitly answers the question, you MUST treat the question as answerable from these chunks alone.
Only if, after carefully reading ALL chunks, you are sure the answer is NOT present, may you say that more information is needed.

Query: {query}

Chunks:
{context}

Do these chunks fully answer the query? If not, give ONE short follow-up search query (5-15 words). Reply EXACTLY:

ANALYSIS: [brief]
MISSING: [what is missing or None]
NEEDS_MORE: [YES or NO]
RELATED_QUERY: [one query or None]"""
    
    try:
        response = llm.invoke(analysis_prompt)
        response_str = (getattr(response, "content", None) or str(response)).strip()
        analysis = response_str
        _append_token_usage(
            state, "analyze_chunks",
            input_tokens=_est_tokens(analysis_prompt),
            output_tokens=_est_tokens(response_str),
        )
        
        # Parse response
        needs_more = False
        new_query = None
        
        if "NEEDS_MORE:" in analysis.upper():
            needs_more_line = [line for line in analysis.split("\n") if "NEEDS_MORE:" in line.upper()]
            if needs_more_line:
                needs_more_value = needs_more_line[0].split(":")[-1].strip().upper()
                needs_more = needs_more_value == "YES"
        
        if needs_more:
            if "RELATED_QUERY:" in analysis.upper():
                query_line = None
                for line in analysis.split("\n"):
                    if "RELATED_QUERY:" in line.upper():
                        query_line = line
                        break
                
                if query_line:
                    query_text = query_line.split(":", 1)[-1].strip()
                    query_text = query_text.strip('"').strip("'").strip()
                    
                    prefixes_to_remove = [
                        "a related query could be",
                        "a query could be",
                        "related query:",
                        "query:",
                        "the query is",
                        "the query should be"
                    ]
                    for prefix in prefixes_to_remove:
                        if query_text.lower().startswith(prefix):
                            query_text = query_text[len(prefix):].strip()
                            query_text = query_text.strip('"').strip("'").strip()
                    
                    if " or " in query_text.lower():
                        query_text = query_text.split(" or ")[0].strip()
                        query_text = query_text.strip('"').strip("'").strip()
                    
                    explanation_markers = ["?", ".", " -", ":", " or", " could", " should"]
                    for marker in explanation_markers:
                        if marker in query_text and query_text.index(marker) < len(query_text) - 10:
                            parts = query_text.split(marker, 1)
                            if len(parts[0]) > 10:
                                query_text = parts[0].strip()
                    
                    query_text = query_text.strip('"').strip("'").strip()
                    query_text = " ".join(query_text.split())
                    
                    invalid_values = ["none", "n/a", "na", "not needed", "sufficient"]
                    if (query_text.lower() not in invalid_values and 
                        len(query_text) > 10 and 
                        len(query_text) < 200):
                        new_query = query_text
                    else:
                        new_query = None
                        needs_more = False
        
        state["chunk_analysis"] = analysis
        state["needs_more_info"] = needs_more
        state["new_query"] = new_query
        
        logger.info(f"Analysis complete. Needs more info: {needs_more}")
        if new_query:
            logger.info(f"New query: {new_query}")
        else:
            logger.info("No new query generated")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        state["chunk_analysis"] = "Analysis failed"
        state["needs_more_info"] = False
        state["new_query"] = None
    
    return state

def second_retrieval(state: AgentState) -> AgentState:
    """
    Perform second retrieval with new query.
    Behavior: new top-5 from vector search, then reuse top expanded chunks
    from the first two seeds computed during initial_retrieval.
    """
    new_query = state["new_query"]
    vector_store = _vector_store

    logger.info(f"Second retrieval (refined query) for: '{new_query}'")

    # Step 1: Vector search with new query (no graph expansion)
    second_seed_ids: List[int] = []
    second_seed_scores: Dict[int, float] = {}

    try:
        vector_results = vector_store.similarity_search_with_score(new_query, k=SECOND_VECTOR_SEARCH_K)
        for doc, distance in vector_results:
            chunk_id = doc.metadata.get("chunk_index")
            if chunk_id is not None:
                similarity_score = distance_to_similarity(distance)
                second_seed_ids.append(chunk_id)
                second_seed_scores[chunk_id] = similarity_score
                logger.debug(f"  Second retrieval chunk {chunk_id} (distance: {distance:.3f}, similarity: {similarity_score:.3f})")
    except Exception as e:
        logger.warning(f"Vector search error in second_retrieval: {e}")

    logger.info(f"Second retrieval vector search returned {len(second_seed_ids)} chunks (k={SECOND_VECTOR_SEARCH_K})")
    _append_token_usage(state, "second_retrieval", embedding_tokens=_est_tokens(new_query or ""))

    # Map chunk_index -> Document
    chunk_dict: Dict[int, Document] = {chunk.metadata.get("chunk_index"): chunk for chunk in _chunks}

    # New initial top-5 for second retrieval
    second_top_docs: List[Document] = []
    for cid in second_seed_ids[:SECOND_VECTOR_SEARCH_K]:
        if cid in chunk_dict:
            second_top_docs.append(chunk_dict[cid])

    # Reuse expanded top-5 from the first two seeds of initial_retrieval
    seed1_expanded_docs: List[Document] = state.get("top5_seed1_expanded") or []
    seed2_expanded_docs: List[Document] = state.get("top5_seed2_expanded") or []

    # Build final second retrieval chunks:
    # new top 5 initial chunks + top similar chunks of 1st seed + top similar chunks of 2nd seed
    seen_ids: set[int] = set()
    second_chunks: List[Document] = []

    for doc in second_top_docs:
        cid = doc.metadata.get("chunk_index")
        if cid is None or cid in seen_ids:
            continue
        seen_ids.add(cid)
        second_chunks.append(doc)

    for doc in seed1_expanded_docs:
        if len(second_chunks) >= FINAL_CHUNKS_FOR_LLM:
            break
        cid = doc.metadata.get("chunk_index")
        if cid is None or cid in seen_ids:
            continue
        seen_ids.add(cid)
        second_chunks.append(doc)

    if len(second_chunks) < FINAL_CHUNKS_FOR_LLM:
        for doc in seed2_expanded_docs:
            if len(second_chunks) >= FINAL_CHUNKS_FOR_LLM:
                break
            cid = doc.metadata.get("chunk_index")
            if cid is None or cid in seen_ids:
                continue
            seen_ids.add(cid)
            second_chunks.append(doc)

    if len(second_chunks) > FINAL_CHUNKS_FOR_LLM:
        second_chunks = second_chunks[:FINAL_CHUNKS_FOR_LLM]

    state["second_seed_ids"] = second_seed_ids
    state["second_seed_scores"] = second_seed_scores
    # No new graph expansion in second retrieval; keep for compatibility
    state["second_expanded_ids"] = []
    state["second_retrieval_chunks"] = second_chunks
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Update debug info
    debug_info = state.get("debug_info", {})
    debug_info["second_retrieval"] = {
        "total_seed_chunks": len(second_seed_ids),
        "graph_expanded_chunks": 0,
        "total_second_chunks": len(second_chunks),
    }
    state["debug_info"] = debug_info

    logger.info(f"Second retrieval complete: {len(second_chunks)} chunks (target {FINAL_CHUNKS_FOR_LLM})")
    return state


def _select_supporting_chunk_ids_from_answer(
    answer: str,
    chunks: List[Document],
    top_k: int = 4,
) -> List[int]:
    """
    Post-hoc grounding: choose which chunks support the answer using simple token overlap.
    This is deterministic and does not rely on the LLM to pick ids.
    """
    if not answer or not chunks:
        return []

    # Normalize text: lowercase alphanumeric tokens
    def _tokens(text: str) -> set[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
        return {t for t in cleaned.split() if t}

    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return []

    scores: List[tuple[int, float]] = []
    for chunk in chunks:
        cid = chunk.metadata.get("chunk_index")
        if cid is None:
            continue
        chunk_tokens = _tokens(chunk.page_content)
        if not chunk_tokens:
            continue
        overlap = len(answer_tokens & chunk_tokens)
        if overlap <= 0:
            continue
        # Simple normalized score
        score = overlap / max(1.0, float(len(answer_tokens)))
        scores.append((cid, score))

    # Sort by score descending and take top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    top_ids: List[int] = []
    for cid, _ in scores:
        top_ids.append(cid)
        if len(top_ids) >= top_k:
            break
    return top_ids

def generate_final_answer(state: AgentState) -> AgentState:
    """Generate final answer from all retrieved chunks (re-ranked and filtered)"""
    # Always use the user's original question; fallback to refined query if missing (e.g. state merge issue)
    query = (state.get("query") or "").strip() or (state.get("new_query") or "").strip()
    if not query:
        logger.warning("No question in state (query and new_query both empty); cannot generate answer.")
        state["final_answer"] = "Error: No question was provided. Please ask a question about the document."
        return state

    logger.info(f"Generating answer for question: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    # Prefer second_retrieval chunks when available; otherwise fall back to initial retrieval chunks.
    second_chunks = state.get("second_retrieval_chunks") or []
    primary_chunks = state.get("reranked_chunks") or state.get("retrieved_chunks", [])

    if second_chunks:
        all_chunks = list(second_chunks)
    else:
        all_chunks = list(primary_chunks)

    # Always cap to FINAL_CHUNKS_FOR_LLM so LLM context is bounded and predictable.
    chunks_to_use = all_chunks[:FINAL_CHUNKS_FOR_LLM]

    llm = _llm

    logger.info(
        f"Generating final answer from {len(chunks_to_use)} chunks "
        f"(primary pool: {len(primary_chunks)}, second pool: {len(second_chunks)})..."
    )

    # Prepare comprehensive context
    chunks_text = []
    rerank_scores = state.get("rerank_scores", {})

    for chunk in chunks_to_use:
        chunk_id = chunk.metadata.get("chunk_index")
        heading = chunk.metadata.get("heading", "No heading")
        section = chunk.metadata.get("section_path", "No section")
        summary = chunk.metadata.get("summary", "")
        # Use full chunk content so the model has complete visibility when answering
        content = chunk.page_content

        # Add relevance score if available
        relevance_note = ""
        if chunk_id in rerank_scores:
            relevance_note = f" (Relevance: {rerank_scores[chunk_id]:.2f})"

        chunks_text.append(f"""
[ChunkId: {chunk_id}]
Section: {section}
Heading: {heading}
Summary: {summary}
Content: {content}{relevance_note}
""")

    context = "\n".join(chunks_text)

    # Full retrieval path: do NOT send past conversation context. Answer from retrieved chunks + current query only.
    # Put the question first and repeat before answer so the model always sees it.
    answer_prompt = f"""QUESTION (answer this using only the document chunks below):
{query}

---
Document chunks (use only this information):
{context}
---

Again, the question to answer is: {query}

Write the answer in CLEAR, WELL-FORMATTED MARKDOWN:

- Start with a one-line **Short Answer**.
- If the document does not state something explicitly, do not add any details; instead say that this information is not available in pdf.
- Then add a *Explanation* section with the size of the text required for a chat responce with as is wording from the context.
- Use **bold subheadings** for important parts (for example: **Definition**, **Conditions**, **Exceptions**, **Important Dates**).
- When listing items, steps, obligations, or conditions, use bullet points.
- If the document does not state something explicitly, say that clearly (do not invent details).
- Do NOT mention or guess any chunk numbers, chunk ids, or sources (for example: \"Chunk 6\", \"[C12]\", \"Source: Chunk 3\").
- Do NOT include any citation markers or source labels of any kind in your answer.

Answer (use only the chunks above; include citation markers as described):"""

    # When streaming_mode is set, only attach the prompt for the API to stream via llm.stream().
    if state.get("streaming_mode"):
        state["answer_prompt"] = answer_prompt
        state["final_answer"] = ""
        return state

    try:
        # Use llm.stream() and accumulate so we support both streaming and non-streaming callers.
        response_parts: List[str] = []
        if hasattr(llm, "stream"):
            for chunk in llm.stream(answer_prompt):
                delta = _extract_stream_delta(chunk)
                if delta:
                    response_parts.append(delta)
            response_str = "".join(response_parts).strip()
        else:
            response = llm.invoke(answer_prompt)
            response_str = (getattr(response, "content", None) or str(response)).strip()
        final_answer = response_str
        _append_token_usage(
            state, "generate_answer",
            input_tokens=_est_tokens(answer_prompt),
            output_tokens=_est_tokens(response_str),
        )
        
        # Post-process to remove any chunk references that might have slipped through
        # Remove patterns like "Chunk 12", "(Chunk 12)", "[Chunk 12]", etc.
        # IMPORTANT: Preserve newlines and markdown formatting!
        final_answer = re.sub(r'\([Cc]hunk\s+\d+\)', '', final_answer)
        final_answer = re.sub(r'\[[Cc]hunk\s+\d+\]', '', final_answer)
        final_answer = re.sub(r'[Cc]hunk\s+\d+', '', final_answer)
        
        # Clean up punctuation issues but preserve newlines
        # Only collapse multiple spaces on the same line (not newlines)
        final_answer = re.sub(r'[ \t]+', ' ', final_answer)  # Multiple spaces/tabs to single space
        final_answer = re.sub(r'\s*,\s*,', ',', final_answer)  # Double commas
        final_answer = re.sub(r'\s*\.\s*\.', '.', final_answer)  # Double periods
        # Clean up spaces before punctuation (but preserve newlines)
        final_answer = re.sub(r' +([,.!?;:])', r'\1', final_answer)
        # Remove trailing spaces from lines (but keep the newline)
        final_answer = re.sub(r' +(\n)', r'\1', final_answer)
        # Clean up multiple consecutive newlines (keep max 2 for paragraph breaks)
        final_answer = re.sub(r'\n{3,}', '\n\n', final_answer)
        final_answer = final_answer.strip()
        
        # Validate that answer is based on retrieved chunks
        if not all_chunks:
            logger.warning("No chunks available for answer generation")
            state["final_answer"] = "I could not retrieve sufficient information from the document to answer this question."
            state["next_questions"] = []
        else:
            # Post-hoc grounding: decide which chunks support the generated answer
            supporting_ids = _select_supporting_chunk_ids_from_answer(final_answer, chunks_to_use)
            if supporting_ids:
                # Deterministic, ordered list of chunk indices actually used (by overlap with answer)
                state["chunk_indices_used_for_answer"] = supporting_ids
            else:
                # Fallback: if overlap score failed, expose all chunk indices we sent to the LLM
                all_ids: List[int] = []
                for c in chunks_to_use:
                    cid = c.metadata.get("chunk_index")
                    if cid is not None and cid not in all_ids:
                        all_ids.append(cid)
                state["chunk_indices_used_for_answer"] = all_ids
            state["final_answer"] = final_answer
            logger.info(f"Final answer generated from {len(chunks_to_use)} chunks")

            # Generate suggested follow-up questions based on the original query and answer
            followup_prompt = f"""You are helping a user explore a long insurance policy document.

Original user question:
{query}

Your answer:
{final_answer}

Now propose 3–7 SHORT, concrete follow-up questions that the user might naturally ask next
to go deeper or clarify details. Focus on practical, answerable questions about this document.

Reply in the following format ONLY:
- question 1
- question 2
- question 3
..."""
            try:
                fu_response = llm.invoke(followup_prompt)
                fu_text = (getattr(fu_response, "content", None) or str(fu_response)).strip()
                _append_token_usage(
                    state,
                    "generate_followups",
                    input_tokens=_est_tokens(followup_prompt),
                    output_tokens=_est_tokens(fu_text),
                )

                suggestions: List[str] = []
                for line in fu_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Accept "- q", "* q", or numbered "1. q"
                    if line[0] in "-*":
                        q = line[1:].strip()
                    else:
                        q = line
                        # Strip leading numbering like "1. ", "2) "
                        q = re.sub(r"^\d+[\.\)]\s*", "", q)
                    q = q.strip()
                    if q:
                        suggestions.append(q)
                # Deduplicate and cap
                deduped = []
                seen = set()
                for q in suggestions:
                    k = q.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    deduped.append(q)
                state["next_questions"] = deduped[:7]
            except Exception as fe:
                logger.warning(f"Failed to generate follow-up questions: {fe}")
                state["next_questions"] = []
        
        # Update debug info
        debug_info = state.get("debug_info", {})
        debug_info["answer_generation"] = {
            "total_chunks_available": len(all_chunks),
            "chunks_used": len(chunks_to_use),
            "primary_chunks": len(primary_chunks),
            "second_chunks": len(second_chunks),
            "answer_length": len(final_answer) if all_chunks else 0,
            "next_questions_count": len(state.get("next_questions") or []),
            "chunk_indices_used_for_answer": state.get("chunk_indices_used_for_answer") or [],
        }
        state["debug_info"] = debug_info
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        state["final_answer"] = "Error generating answer. Please try again."
        debug_info = state.get("debug_info", {})
        debug_info["answer_generation"] = {
            "error": str(e),
        }
        state["debug_info"] = debug_info
    
    return state

# ---------------- CONDITIONAL EDGE FUNCTIONS ---------------- 
def route_query_type(state: AgentState) -> Literal["summarize_page", "use_history", "normal_retrieval"]:
    """Route query to page summarization, use_history (session context only), or normal retrieval"""
    is_page_summary = state.get("is_page_summary", False)
    page_number = state.get("page_number")
    is_use_history = state.get("is_use_history", False)

    if is_page_summary and page_number:
        logger.info(f"Routing to page summarization for page {page_number}")
        return "summarize_page"
    if is_use_history:
        logger.info("Routing to use_history (answer from session context)")
        return "use_history"
    logger.info("Routing to normal retrieval")
    return "normal_retrieval"


def should_continue_search(state: AgentState) -> Literal["second_retrieval", "generate_answer"]:
    """Decide whether to do second retrieval or generate answer"""
    needs_more = state.get("needs_more_info", False)
    iteration = state.get("iteration_count", 0)
    
    if iteration >= 3:
        logger.info("Maximum iterations reached, generating answer")
        return "generate_answer"
    
    if needs_more and state.get("new_query"):
        logger.info("Proceeding to second graph-enhanced retrieval")
        return "second_retrieval"
    else:
        logger.info("Sufficient information, generating answer")
        return "generate_answer"

# ---------------- WORKFLOW CREATION ---------------- 
def create_retrieval_agent(vector_store: Chroma, document_graph: DocumentGraph, chunks: List[Document], llm: Any, document_folder: Optional[Path] = None) -> StateGraph:
    """Create the LangGraph retrieval agent with page summarization support"""
    
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("summarize_page", summarize_page)
    workflow.add_node("answer_from_history", answer_from_history)
    workflow.add_node("initial_retrieval", initial_retrieval)
    workflow.add_node("analyze_chunks", analyze_chunks)
    workflow.add_node("second_retrieval", second_retrieval)
    workflow.add_node("generate_answer", generate_final_answer)

    # Define edges
    workflow.set_entry_point("classify_query")

    # Route based on query type: summarize_page | use_history | normal_retrieval
    workflow.add_conditional_edges(
        "classify_query",
        route_query_type,
        {
            "summarize_page": "summarize_page",
            "use_history": "answer_from_history",
            "normal_retrieval": "initial_retrieval",
        },
    )

    # Page summarization path
    workflow.add_edge("summarize_page", END)

    # Use history path (no retrieval)
    workflow.add_edge("answer_from_history", END)

    # Normal retrieval path
    workflow.add_edge("initial_retrieval", "analyze_chunks")

    # Conditional edge for second retrieval
    workflow.add_conditional_edges(
        "analyze_chunks",
        should_continue_search,
        {
            "second_retrieval": "second_retrieval",
            "generate_answer": "generate_answer",
        },
    )

    workflow.add_edge("second_retrieval", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

# ---------------- HELPER FUNCTIONS ---------------- 
def find_vector_mapping_file(directory: Path) -> Optional[Path]:
    """Find vector mapping JSON file in directory"""
    # First, check for unified_mapping.json (for unified/multi-PDF graphs)
    unified_file = directory / "unified_mapping.json"
    if unified_file.exists():
        return unified_file
    
    # Look for files matching pattern: *_vector_mapping.json
    pattern = "*_vector_mapping.json"
    mapping_files = list(directory.glob(pattern))
    
    if not mapping_files:
        # Try alternative pattern: file_vector_mapping.json
        alt_file = directory / "file_vector_mapping.json"
        if alt_file.exists():
            return alt_file
        return None
    
    # If multiple files found, use the most recent one
    if len(mapping_files) > 1:
        logger.info(f"Found {len(mapping_files)} mapping files, using most recent:")
        for mf in sorted(mapping_files, key=lambda p: p.stat().st_mtime, reverse=True):
            logger.info(f"  - {mf.name}")
    
    return max(mapping_files, key=lambda p: p.stat().st_mtime)

def find_graph_file(directory: Path) -> Optional[Path]:
    """Find document graph JSON file in directory"""
    # First, check for unified_graph.json (for unified/multi-PDF graphs)
    unified_file = directory / "unified_graph.json"
    if unified_file.exists():
        return unified_file
    
    # Look for files matching pattern: *_document_graph.json
    pattern = "*_document_graph.json"
    graph_files = list(directory.glob(pattern))
    
    if not graph_files:
        # Try alternative pattern: file_document_graph.json
        alt_file = directory / "file_document_graph.json"
        if alt_file.exists():
            return alt_file
        return None
    
    # If multiple files found, use the most recent one
    if len(graph_files) > 1:
        logger.info(f"Found {len(graph_files)} graph files, using most recent:")
        for gf in sorted(graph_files, key=lambda p: p.stat().st_mtime, reverse=True):
            logger.info(f"  - {gf.name}")
    
    return max(graph_files, key=lambda p: p.stat().st_mtime)

def find_vector_db_path(directory: Path, document_stem: Optional[str] = None) -> Optional[Path]:
    """Find vector database directory"""
    vector_db_dir = directory / "vector_db"
    
    if not vector_db_dir.exists():
        return None
    
    # First, check for unified vector store (for unified/multi-PDF graphs)
    unified_db = vector_db_dir / "unified"
    if unified_db.exists():
        return unified_db
    
    # If document_stem is provided, try that first
    if document_stem:
        db_path = vector_db_dir / document_stem
        if db_path.exists():
            return db_path
    
    # Otherwise, find any subdirectory in vector_db
    subdirs = [d for d in vector_db_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    # Use the most recently modified directory
    if len(subdirs) > 1:
        logger.info(f"Found {len(subdirs)} vector DB directories, using most recent:")
        for sd in sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True):
            logger.info(f"  - {sd.name}")
    
    return max(subdirs, key=lambda p: p.stat().st_mtime)

# ---------------- MAIN FUNCTION ---------------- 
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Graph-enhanced retrieval agent using LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use folder name (auto-detects E subfolder)
  python retrivalAgentE.py output/HDFC-Life-Cancer-Care-101N106V04-Policy-Document
  
  # Use most recent folder in output directory
  python retrivalAgentE.py
  
  # Specify explicit paths (overrides auto-detection)
  python retrivalAgentE.py output/folder --mapping-file path/to/mapping.json --graph-file path/to/graph.json
        """
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to folder containing document (will look for E subfolder), or use most recent folder if not specified"
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        help="Explicit path to vector mapping JSON file (overrides auto-detection)"
    )
    parser.add_argument(
        "--graph-file",
        type=str,
        help="Explicit path to document graph JSON file (overrides auto-detection)"
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        help="Explicit path to vector database directory (overrides auto-detection)"
    )
    
    args = parser.parse_args()
    
    # Determine base directory and files
    if args.path:
        input_path = Path(args.path)
    else:
        # Default: look for most recent folder in output directory
        output_dir = Path("output")
        if output_dir.exists():
            folders = [d for d in output_dir.iterdir() if d.is_dir()]
            if folders:
                input_path = max(folders, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent folder: {input_path}")
            else:
                logger.error("No folders found in output directory. Please specify a folder path.")
                sys.exit(1)
        else:
            logger.error("Output directory not found. Please specify a folder path.")
            sys.exit(1)
    
    # If input_path is a file, use its parent directory
    if input_path.is_file():
        base_folder = input_path.parent
    elif input_path.is_dir():
        base_folder = input_path
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)
    
    # Look for E subfolder (Plan E output)
    plan_e_dir = base_folder / "E"
    
    # Auto-detect or use explicit paths
    if args.mapping_file:
        vector_mapping_file = Path(args.mapping_file)
        document_stem = vector_mapping_file.stem.replace("_vector_mapping", "")
    else:
        if plan_e_dir.exists():
            vector_mapping_file = find_vector_mapping_file(plan_e_dir)
        else:
            vector_mapping_file = find_vector_mapping_file(base_folder)
        if vector_mapping_file:
            document_stem = vector_mapping_file.stem.replace("_vector_mapping", "")
        else:
            document_stem = None
    
    # Find graph file
    if args.graph_file:
        graph_file = Path(args.graph_file)
    else:
        if plan_e_dir.exists():
            graph_file = find_graph_file(plan_e_dir)
        else:
            graph_file = find_graph_file(base_folder)
    
    # Find vector DB path
    if args.vector_db:
        vector_db_path = Path(args.vector_db)
    else:
        if plan_e_dir.exists():
            vector_db_path = find_vector_db_path(plan_e_dir, document_stem)
        else:
            vector_db_path = find_vector_db_path(base_folder, document_stem)
    
    # Validate files
    if not vector_mapping_file or not vector_mapping_file.exists():
        logger.error(f"Vector mapping file not found in: {base_folder}")
        logger.error("Please run vectorizerE.py first to generate the vector mapping")
        logger.error("Or specify the file with --mapping-file")
        sys.exit(1)
    
    if not graph_file or not graph_file.exists():
        logger.error(f"Graph file not found in: {base_folder}")
        logger.error("Please run vectorizerE.py first to generate the graph")
        logger.error("Or specify the file with --graph-file")
        sys.exit(1)
    
    if not vector_db_path or not vector_db_path.exists():
        logger.error(f"Vector DB path not found")
        logger.error("Please run vectorizerE.py first to generate the vector database")
        logger.error("Or specify the path with --vector-db")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("RetrievalAgentE - Plan E: Graph-Enhanced Retrieval Agent")
    logger.info("=" * 80)
    logger.info(f"Document folder: {base_folder}")
    logger.info(f"Vector mapping: {vector_mapping_file.name}")
    logger.info(f"Graph file: {graph_file.name}")
    logger.info(f"Vector DB: {vector_db_path.name}")
    
    # Load data
    chunks = load_chunks_from_mapping(vector_mapping_file)
    if not chunks:
        logger.error("Failed to load chunks. Exiting.")
        sys.exit(1)
    
    document_graph = DocumentGraph()
    document_graph.load(graph_file)
    
    vector_store = load_vector_store(vector_db_path)
    llm = get_llm(temperature=0.3)
    
    logger.info("=" * 80)
    logger.info("Data loaded successfully!")
    logger.info(f"Graph: {len(document_graph.graph.nodes)} nodes, {len(document_graph.graph.edges)} edges")
    logger.info("=" * 80)
    
    # Set global resources (include document folder for page summarization)
    set_agent_resources(vector_store, document_graph, chunks, llm, base_folder)
    
    # Create agent (include document folder for page summarization)
    agent = create_retrieval_agent(vector_store, document_graph, chunks, llm, base_folder)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("GRAPH-ENHANCED RETRIEVAL AGENT - Interactive Mode")
    print("=" * 80)
    print("Enter queries. The agent will:")
    print("  1. Perform vector similarity search (seed chunks)")
    print("  2. Expand via graph traversal (sections, adjacent chunks)")
    print("  3. Analyze if more information is needed")
    print("  4. Optionally perform a second graph-enhanced search")
    print("  5. Generate a comprehensive answer")
    print("Type 'exit' or 'quit' to exit")
    print("=" * 80)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if not query or query.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            # Initialize state
            initial_state: AgentState = {
                "query": query,
                "is_page_summary": False,
                "page_number": None,
                "seed_chunk_ids": [],
                "graph_expanded_ids": [],
                "retrieved_chunks": [],
                "chunk_analysis": "",
                "needs_more_info": False,
                "new_query": None,
                "second_seed_ids": [],
                "second_expanded_ids": [],
                "second_retrieval_chunks": [],
                "final_answer": "",
                "iteration_count": 0,
                "document_folder": None,
            }
            
            # Run agent
            print("\n" + "=" * 80)
            print("AGENT EXECUTION")
            print("=" * 80)
            
            final_state = agent.invoke(initial_state)
            
            # Display results
            print("\n" + "=" * 80)
            print("FINAL ANSWER")
            print("=" * 80)
            print(final_state["final_answer"])
            print("=" * 80)
            
            if final_state.get("chunk_analysis"):
                print("\n" + "=" * 80)
                print("CHUNK ANALYSIS")
                print("=" * 80)
                print(final_state["chunk_analysis"])
                print("=" * 80)
            
            print(f"\nRetrieval Statistics:")
            print(f"  - Seed chunks (vector search): {len(final_state['seed_chunk_ids'])}")
            print(f"  - Graph-expanded chunks: {len(final_state['graph_expanded_ids'])}")
            print(f"  - Total initial chunks: {len(final_state['retrieved_chunks'])}")
            if final_state.get("second_seed_ids"):
                print(f"  - Second seed chunks: {len(final_state['second_seed_ids'])}")
                print(f"  - Second expanded chunks: {len(final_state['second_expanded_ids'])}")
                print(f"  - Total second chunks: {len(final_state['second_retrieval_chunks'])}")
            print(f"  - Total chunks used: {len(final_state['retrieved_chunks']) + len(final_state.get('second_retrieval_chunks', []))}")
            
            if final_state.get("new_query"):
                print(f"  - Second query used: {final_state['new_query']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()