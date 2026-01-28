"""
RetrievalE - Plan E Implementation: Graph-Enhanced Retrieval
Retrieves documents using graph-enhanced search combining vector similarity with graph traversal.
Uses the vectorized data and graph structure from vectorizerE.py output.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Graph library
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.error("networkx is required. Install with: pip install networkx")
    sys.exit(1)

# LangChain imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)

# ---------------- CONFIGURATION ---------------- 
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
OLLAMA_BASE_URL = "http://localhost:11434"

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
        
        for predecessor in self.graph.predecessors(chunk_node):
            if self.graph.nodes[predecessor].get("type") == "section":
                edge_data = self.graph.get_edge_data(predecessor, chunk_node)
                if edge_data and edge_data.get("relation") in ["contains", "belongs_to"]:
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
                if section_path:
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
        doc = Document(
            page_content=item.get("content", ""),
            metadata=item.get("metadata", {})
        )
        chunks.append(doc)
    
    logger.info(f"Loaded {len(chunks)} chunks from mapping file")
    return chunks

def load_vector_store(vector_db_path: Path) -> Chroma:
    """Load Chroma vector store"""
    if not vector_db_path.exists():
        logger.error(f"Vector DB path not found: {vector_db_path}")
        sys.exit(1)
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(vector_db_path)
    )
    
    logger.info(f"Loaded vector store from: {vector_db_path}")
    return vector_store

# ---------------- GRAPH-ENHANCED RETRIEVAL FUNCTION ---------------- 
def graph_retrieve(
    query: str,
    vector_store: Chroma,
    document_graph: DocumentGraph,
    chunks: List[Document],
    k: int = 10,
    page_hint: Optional[int] = None,
    section_hint: Optional[str] = None,
    expand_via_graph: bool = True,
    max_expansion: int = 25
) -> List[tuple]:
    """
    Graph-enhanced retrieval combining vector similarity with graph traversal.
    
    Args:
        query: Search query
        vector_store: Chroma vector store
        document_graph: Document graph for traversal
        chunks: List of all chunks (for lookup)
        k: Number of results to return
        page_hint: Optional page number to prioritize
        section_hint: Optional section path to filter
        expand_via_graph: Whether to expand via graph traversal
        max_expansion: Maximum chunks to expand via graph
    
    Returns:
        List of (document, distance, source) tuples
    """
    results = []
    seen_ids = set()
    chunk_dict = {chunk.metadata.get("chunk_index"): chunk for chunk in chunks}
    
    # Strategy 1: Vector similarity search (seed chunks)
    logger.info(f"Performing vector similarity search for: '{query}'")
    seed_chunk_ids = []
    try:
        vector_results = vector_store.similarity_search_with_score(query, k=k * 2)
        for doc, distance in vector_results:
            chunk_id = doc.metadata.get("chunk_index")
            if chunk_id is not None:
                seed_chunk_ids.append(chunk_id)
                seen_ids.add(chunk_id)
                results.append((doc, distance, "vector"))
                logger.debug(f"  Seed chunk {chunk_id}, distance: {distance:.3f}")
    except Exception as e:
        logger.warning(f"Vector search error: {e}")
    
    logger.info(f"Found {len(seed_chunk_ids)} seed chunks from vector search")
    
    # Strategy 2: Graph expansion from seed chunks
    if expand_via_graph and seed_chunk_ids:
        logger.info(f"Expanding via graph traversal (max {max_expansion} chunks)...")
        graph_expanded_ids = document_graph.expand_from_chunks(seed_chunk_ids, max_expansion=max_expansion)
        
        # Filter out already seen chunks
        new_expanded = [cid for cid in graph_expanded_ids if cid not in seen_ids]
        
        # Get max vector distance to rank graph results
        max_vector_distance = max((score for _, score, src in results if src == "vector"), default=1.0)
        graph_distance = max_vector_distance + 0.4  # Rank after vector results
        
        for chunk_id in new_expanded:
            if chunk_id in chunk_dict:
                seen_ids.add(chunk_id)
                doc = chunk_dict[chunk_id]
                results.append((doc, graph_distance, "graph_expansion"))
                logger.debug(f"  Graph-expanded chunk {chunk_id}")
        
        logger.info(f"Graph expansion: {len(new_expanded)} additional chunks")
    
    # Strategy 3: Graph-based page filtering
    if page_hint is not None:
        logger.info(f"Filtering by page via graph: {page_hint}")
        page_chunk_ids = document_graph.get_page_chunks(page_hint)
        
        max_vector_distance = max((score for _, score, src in results if src == "vector"), default=1.0)
        page_distance = max_vector_distance + 0.6
        
        for chunk_id in page_chunk_ids:
            if chunk_id not in seen_ids and chunk_id in chunk_dict:
                seen_ids.add(chunk_id)
                doc = chunk_dict[chunk_id]
                results.append((doc, page_distance, "graph_page"))
                logger.debug(f"  Page chunk {chunk_id}")
    
    # Strategy 4: Graph-based section filtering
    if section_hint:
        logger.info(f"Filtering by section via graph: {section_hint}")
        section_chunk_ids = document_graph.get_section_chunks(section_hint)
        
        max_vector_distance = max((score for _, score, src in results if src == "vector"), default=1.0)
        section_distance = max_vector_distance + 0.5
        
        for chunk_id in section_chunk_ids:
            if chunk_id not in seen_ids and chunk_id in chunk_dict:
                seen_ids.add(chunk_id)
                doc = chunk_dict[chunk_id]
                results.append((doc, section_distance, "graph_section"))
                logger.debug(f"  Section chunk {chunk_id}")
    
    # Re-rank by distance (lower is better, so sort ascending)
    results.sort(key=lambda x: x[1])
    
    # Return top k with scores
    final_results = [(doc, distance, source) for doc, distance, source in results[:k]]
    
    logger.info(f"Retrieved {len(final_results)} chunks using graph-enhanced search")
    return final_results

# ---------------- DISPLAY FUNCTIONS ---------------- 
def display_results(results: List[tuple], show_content: bool = False):
    """Display retrieval results in a readable format"""
    print("\n" + "=" * 80)
    print("GRAPH-ENHANCED RETRIEVAL RESULTS")
    print("=" * 80)
    print("(Lower distance = better match)")
    print("=" * 80)
    
    # Count by source
    source_counts = {}
    for _, _, source in results:
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nRetrieval breakdown:")
    for source, count in sorted(source_counts.items()):
        print(f"  - {source}: {count}")
    print()
    
    for i, (doc, distance, source) in enumerate(results, 1):
        chunk_id = doc.metadata.get("chunk_index", "N/A")
        heading = doc.metadata.get("heading", "No heading")
        section = doc.metadata.get("section_path", "No section")
        summary = doc.metadata.get("summary", "No summary")
        
        print(f"\n[{i}] Chunk ID: {chunk_id} | Distance: {distance:.3f} | Source: {source}")
        print(f"    Heading: {heading}")
        print(f"    Section: {section}")
        print(f"    Summary: {summary}")
        
        if show_content:
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"    Content: {content_preview}")
    
    print("\n" + "=" * 80)

# ---------------- MAIN FUNCTION ---------------- 
def main():
    """Main entry point for retrieval"""
    # Accept folder name or path as argument
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
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
    
    # If it's a folder, use it; if it's a file, use its parent
    if input_path.is_file():
        base_folder = input_path.parent
    elif input_path.is_dir():
        base_folder = input_path
    else:
        logger.error(f"Path not found: {input_path}")
        sys.exit(1)
    
    # Look for E subfolder (Plan E output)
    plan_e_dir = base_folder / "E"
    if not plan_e_dir.exists():
        logger.error(f"Plan E folder not found: {plan_e_dir}")
        logger.error("Please run vectorizerE.py first to generate the vectorized data.")
        sys.exit(1)
    
    # Find vector mapping and graph files
    mapping_files = list(plan_e_dir.glob("*_vector_mapping.json"))
    graph_files = list(plan_e_dir.glob("*_document_graph.json"))
    
    if not mapping_files:
        logger.error(f"Vector mapping file not found in: {plan_e_dir}")
        logger.error("Please run vectorizerE.py first to generate the vectorized data.")
        sys.exit(1)
    
    if not graph_files:
        logger.error(f"Graph file not found in: {plan_e_dir}")
        logger.error("Please run vectorizerE.py first to generate the graph.")
        sys.exit(1)
    
    # Use first files found (or most recent if multiple)
    vector_mapping_file = max(mapping_files, key=lambda p: p.stat().st_mtime)
    graph_file = max(graph_files, key=lambda p: p.stat().st_mtime)
    
    # Get document stem from mapping file
    doc_stem = vector_mapping_file.stem.replace("_vector_mapping", "")
    vector_db_path = plan_e_dir / "vector_db" / doc_stem
    
    # Check if files exist
    if not vector_mapping_file.exists():
        logger.error(f"Vector mapping file not found: {vector_mapping_file}")
        logger.error("Please run vectorizerE.py first to generate the vectorized data.")
        sys.exit(1)
    
    if not graph_file.exists():
        logger.error(f"Graph file not found: {graph_file}")
        logger.error("Please run vectorizerE.py first to generate the graph.")
        sys.exit(1)
    
    if not vector_db_path.exists():
        logger.error(f"Vector DB path not found: {vector_db_path}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("RetrievalE - Plan E: Graph-Enhanced Retrieval")
    logger.info("=" * 80)
    logger.info(f"Loading data from: {base_folder}")
    logger.info(f"Vector mapping: {vector_mapping_file.name}")
    logger.info(f"Graph file: {graph_file.name}")
    
    # Load chunks from mapping
    chunks = load_chunks_from_mapping(vector_mapping_file)
    if not chunks:
        logger.error("Failed to load chunks. Exiting.")
        sys.exit(1)
    
    # Load document graph
    document_graph = DocumentGraph()
    document_graph.load(graph_file)
    
    # Load vector store
    vector_store = load_vector_store(vector_db_path)
    
    logger.info("=" * 80)
    logger.info("Data loaded successfully!")
    logger.info(f"Graph: {len(document_graph.graph.nodes)} nodes, {len(document_graph.graph.edges)} edges")
    logger.info("=" * 80)
    
    # Interactive query loop
    print("\n" + "=" * 80)
    print("GRAPH-ENHANCED RETRIEVAL - Interactive Mode")
    print("=" * 80)
    print("Enter queries to search. Commands:")
    print("  - 'exit' or 'quit' to exit")
    print("  - 'page:N' to add page hint (e.g., 'page:7')")
    print("  - 'section:NAME' to add section hint (e.g., 'section:6.1.3 Compliance obligations')")
    print("  - 'k:N' to set number of results (e.g., 'k:15')")
    print("  - 'no-graph' to disable graph expansion")
    print("  - 'expand:N' to set max graph expansion (e.g., 'expand:30')")
    print("  - 'show-content' to display full content")
    print("=" * 80)
    
    show_content = False
    default_k = 10
    expand_via_graph = True
    max_expansion = 25
    
    while True:
        try:
            query_input = input("\nQuery: ").strip()
            
            if not query_input or query_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            # Parse commands
            query = query_input
            page_hint = None
            section_hint = None
            k = default_k
            
            # Check for commands
            if 'page:' in query:
                parts = query.split('page:')
                query = parts[0].strip()
                try:
                    page_hint = int(parts[1].split()[0])
                except:
                    logger.warning("Invalid page number format")
            
            if 'section:' in query:
                parts = query.split('section:')
                query = parts[0].strip()
                section_hint = parts[1].strip()
            
            if 'k:' in query:
                parts = query.split('k:')
                query = parts[0].strip()
                try:
                    k = int(parts[1].split()[0])
                except:
                    logger.warning("Invalid k value")
            
            if 'expand:' in query:
                parts = query.split('expand:')
                query = parts[0].strip()
                try:
                    max_expansion = int(parts[1].split()[0])
                except:
                    logger.warning("Invalid expansion value")
            
            if 'no-graph' in query.lower():
                query = query.replace('no-graph', '').strip()
                expand_via_graph = False
            
            if 'show-content' in query.lower():
                query = query.replace('show-content', '').strip()
                show_content = True
            
            if not query:
                print("Please enter a query.")
                continue
            
            # Perform graph-enhanced retrieval
            results = graph_retrieve(
                query=query,
                vector_store=vector_store,
                document_graph=document_graph,
                chunks=chunks,
                k=k,
                page_hint=page_hint,
                section_hint=section_hint,
                expand_via_graph=expand_via_graph,
                max_expansion=max_expansion
            )
            
            # Display results
            display_results(results, show_content=show_content)
            show_content = False  # Reset after one use
            expand_via_graph = True  # Reset after one use
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)

if __name__ == "__main__":
    main()
