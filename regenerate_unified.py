"""
Regenerate Unified Graph from Existing Vectorized Data
Reads existing vectorized PDFs from output/E and creates a unified graph.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

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
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

# Import DocumentGraph from vectorizerE
try:
    from vectorizerE import DocumentGraph
except ImportError:
    # Type stub for linter
    class DocumentGraph:  # type: ignore
        pass
    logging.error("Could not import DocumentGraph from vectorizerE")

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
SIMILARITY_THRESHOLD = 0.50  # Threshold for inter-PDF similarity edges
MAX_SIMILAR_PER_CHUNK = 3  # Maximum inter-PDF similarity edges per chunk

# ---------------- UNIFIED GRAPH CLASS ---------------- 
class UnifiedDocumentGraph:
    """Unified graph for multiple PDFs with PDF clusters"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.chunk_nodes = {}  # (pdf_name, chunk_id) -> node_id
        self.section_nodes = {}  # (pdf_name, section_path) -> node_id
        self.page_nodes = {}  # (pdf_name, page_number) -> node_id
        self.pdf_nodes = {}  # pdf_name -> node_id
        self.chunk_to_pdf = {}  # global_chunk_id -> pdf_name
        self.global_chunk_counter = 0  # Global counter for unique chunk IDs across PDFs
    
    def add_pdf_node(self, pdf_name: str, pdf_path: str = ""):
        """Add a PDF/document node to represent a cluster"""
        node_id = f"pdf:{pdf_name}"
        if node_id not in self.graph:
            self.graph.add_node(node_id,
                              type="pdf",
                              pdf_name=pdf_name,
                              pdf_path=pdf_path)
            self.pdf_nodes[pdf_name] = node_id
        return node_id
    
    def add_section_node(self, section_path: str, section_title: str, level: int, start_line: int, pdf_name: str):
        """Add a section node with PDF context"""
        node_id = f"section:{pdf_name}:{section_path}"
        if node_id not in self.graph:
            self.graph.add_node(node_id,
                              type="section",
                              section_path=section_path,
                              section_title=section_title,
                              level=level,
                              start_line=start_line,
                              pdf_name=pdf_name)
            self.section_nodes[(pdf_name, section_path)] = node_id
        
        # Connect section to its PDF
        pdf_node = self.pdf_nodes.get(pdf_name)
        if pdf_node:
            self.graph.add_edge(pdf_node, node_id, relation="contains")
        
        return node_id
    
    def add_chunk_node(self, original_chunk_id: int, chunk: Document, pdf_name: str):
        """Add a chunk node with PDF context, using global chunk ID"""
        global_chunk_id = self.global_chunk_counter
        self.global_chunk_counter += 1
        
        node_id = f"chunk:{pdf_name}:{original_chunk_id}"
        if node_id not in self.graph:
            self.graph.add_node(node_id,
                              type="chunk",
                              chunk_id=global_chunk_id,  # Use global ID
                              original_chunk_id=original_chunk_id,  # Keep original for reference
                              heading=chunk.metadata.get("heading", ""),
                              section_path=chunk.metadata.get("section_path", ""),
                              page_number=chunk.metadata.get("page_number"),
                              pdf_name=pdf_name)
            self.chunk_nodes[(pdf_name, original_chunk_id)] = node_id
            self.chunk_to_pdf[global_chunk_id] = pdf_name
        return node_id, global_chunk_id
    
    def add_page_node(self, page_number: int, pdf_name: str):
        """Add a page node with PDF context"""
        if page_number <= 0:
            return None
        
        node_id = f"page:{pdf_name}:{page_number}"
        if node_id not in self.graph:
            self.graph.add_node(node_id,
                              type="page",
                              page_number=page_number,
                              pdf_name=pdf_name)
            self.page_nodes[(pdf_name, page_number)] = node_id
        
        # Connect page to its PDF
        pdf_node = self.pdf_nodes.get(pdf_name)
        if pdf_node:
            self.graph.add_edge(pdf_node, node_id, relation="contains")
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, relation: str, **kwargs):
        """Add an edge between nodes"""
        if source_id and target_id:
            self.graph.add_edge(source_id, target_id, relation=relation, **kwargs)
    
    def save(self, filepath: Path):
        """Save unified graph to JSON format"""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node_id, node_data in self.graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node_id,
                **node_data
            })
        
        for source, target, edge_data in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": source,
                "target": target,
                **edge_data
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved unified graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")

# ---------------- LOAD EXISTING DATA ---------------- 
def find_vectorized_documents(base_dir: Path) -> List[Dict[str, Path]]:
    """
    Find all vectorized documents under base_dir.
    
    This now searches recursively so it works with layouts like:
      output/<group>/<pdf_stem>/E/<pdf_stem>_document_graph.json
    """
    documents = []
    
    # Find all document graph files recursively
    graph_files = list(base_dir.rglob("*_document_graph.json"))
    
    for graph_file in graph_files:
        # Extract PDF name from graph file name
        pdf_name = graph_file.stem.replace("_document_graph", "")
        
        # Find corresponding mapping file in the same directory as the graph
        mapping_file = graph_file.with_name(f"{pdf_name}_vector_mapping.json")
        
        if mapping_file.exists():
            documents.append({
                "pdf_name": pdf_name,
                "graph_file": graph_file,
                "mapping_file": mapping_file
            })
        else:
            logger.warning(f"Mapping file not found for {pdf_name}: {mapping_file}")
    
    return documents

def load_document_data(graph_file: Path, mapping_file: Path) -> Tuple[Optional[DocumentGraph], Optional[List[Document]], Optional[Dict[int, np.ndarray]]]:
    """Load graph and chunks from existing files"""
    try:
        # Load graph
        from vectorizerE import DocumentGraph  # type: ignore
        document_graph = DocumentGraph()
        document_graph.load(graph_file)
        
        # Load chunks from mapping
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        chunks = []
        for item in mapping_data:
            doc = Document(
                page_content=item.get("content", ""),
                metadata=item.get("metadata", {})
            )
            chunks.append(doc)
        
        # Generate embeddings for similarity computation
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        chunk_embeddings = {}
        logger.info(f"  Generating embeddings for {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("chunk_index")
            if chunk_id is not None:
                embedding = embeddings.embed_query(chunk.page_content)
                chunk_embeddings[chunk_id] = np.array(embedding)
                if (i + 1) % 10 == 0:
                    logger.info(f"    Generated {i + 1}/{len(chunks)} embeddings...")
        
        logger.info(f"✓ Loaded {len(chunks)} chunks from {graph_file.name}")
        return document_graph, chunks, chunk_embeddings
        
    except Exception as e:
        logger.error(f"Error loading {graph_file.name}: {e}", exc_info=True)
        return None, None, None

# ---------------- GRAPH MERGING ---------------- 
def merge_graph_into_unified(pdf_name: str, source_graph: Any, 
                            source_chunks: List[Document], unified_graph: UnifiedDocumentGraph,
                            chunk_id_mapping: Dict[int, int]) -> Dict[int, int]:
    """Merge a single PDF's graph into the unified graph"""
    logger.info(f"Merging graph for PDF: {pdf_name}")
    
    # Add PDF node
    pdf_node = unified_graph.add_pdf_node(pdf_name)
    
    # Map old node IDs to new node IDs
    node_id_map = {}
    local_chunk_id_mapping = {}  # original_chunk_id -> global_chunk_id
    
    # Add all nodes from source graph
    for node_id, node_data in source_graph.graph.nodes(data=True):
        node_type = node_data.get("type")
        
        if node_type == "section":
            section_path = node_data.get("section_path", "")
            section_title = node_data.get("section_title", "")
            level = node_data.get("level", 0)
            start_line = node_data.get("start_line", 0)
            new_node_id = unified_graph.add_section_node(section_path, section_title, level, start_line, pdf_name)
            node_id_map[node_id] = new_node_id
            
        elif node_type == "chunk":
            original_chunk_id = node_data.get("chunk_id")
            if original_chunk_id is not None:
                # Find the chunk document
                chunk_doc = next((c for c in source_chunks if c.metadata.get("chunk_index") == original_chunk_id), None)
                if chunk_doc:
                    new_node_id, global_chunk_id = unified_graph.add_chunk_node(original_chunk_id, chunk_doc, pdf_name)
                    node_id_map[node_id] = new_node_id
                    local_chunk_id_mapping[original_chunk_id] = global_chunk_id
                    chunk_id_mapping[original_chunk_id] = global_chunk_id
                    
        elif node_type == "page":
            page_number = node_data.get("page_number")
            if page_number:
                new_node_id = unified_graph.add_page_node(page_number, pdf_name)
                node_id_map[node_id] = new_node_id
    
    # Add all edges from source graph (only intra-PDF edges)
    for source, target, edge_data in source_graph.graph.edges(data=True):
        new_source = node_id_map.get(source)
        new_target = node_id_map.get(target)
        
        if new_source and new_target:
            relation = edge_data.get("relation", "")
            # Only add non-similarity edges (similarity edges will be computed inter-PDF)
            if relation != "similar_to":
                unified_graph.add_edge(new_source, new_target, relation=relation, **{k: v for k, v in edge_data.items() if k != "relation"})
    
    logger.info(f"✓ Merged {len([n for n in unified_graph.graph.nodes() if unified_graph.graph.nodes[n].get('pdf_name') == pdf_name])} nodes for {pdf_name}")
    return local_chunk_id_mapping

# ---------------- INTER-PDF SIMILARITY ---------------- 
def compute_inter_pdf_similarity(unified_graph: UnifiedDocumentGraph, 
                                all_chunk_embeddings: Dict[str, Dict[int, np.ndarray]],
                                chunk_id_mappings: Dict[str, Dict[int, int]]):
    """Compute similarity edges between chunks from different PDFs"""
    logger.info("Computing inter-PDF similarity edges...")
    
    pdf_names = list(all_chunk_embeddings.keys())
    similarity_edges_added = 0
    
    # Compare chunks from different PDFs
    for i, pdf1 in enumerate(pdf_names):
        for pdf2 in pdf_names[i+1:]:
            logger.info(f"  Comparing {pdf1} <-> {pdf2}")
            
            embeddings1 = all_chunk_embeddings[pdf1]
            embeddings2 = all_chunk_embeddings[pdf2]
            mapping1 = chunk_id_mappings[pdf1]
            mapping2 = chunk_id_mappings[pdf2]
            
            for original_chunk_id1, embedding1 in embeddings1.items():
                # Get global chunk ID and node
                global_chunk_id1 = mapping1.get(original_chunk_id1)
                if global_chunk_id1 is None:
                    continue
                
                chunk_node1 = unified_graph.chunk_nodes.get((pdf1, original_chunk_id1))
                if not chunk_node1:
                    continue
                
                # Normalize embedding
                embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
                
                similarities = []
                for original_chunk_id2, embedding2 in embeddings2.items():
                    # Get global chunk ID and node
                    global_chunk_id2 = mapping2.get(original_chunk_id2)
                    if global_chunk_id2 is None:
                        continue
                    
                    chunk_node2 = unified_graph.chunk_nodes.get((pdf2, original_chunk_id2))
                    if not chunk_node2:
                        continue
                    
                    # Normalize embedding
                    embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
                    
                    # Compute cosine similarity
                    similarity = np.dot(embedding1_norm, embedding2_norm)
                    similarity = max(0.0, min(1.0, similarity))
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        similarities.append((original_chunk_id2, similarity))
                
                # Sort by similarity and take top N
                similarities.sort(key=lambda x: x[1], reverse=True)
                for original_chunk_id2, similarity in similarities[:MAX_SIMILAR_PER_CHUNK]:
                    chunk_node2 = unified_graph.chunk_nodes.get((pdf2, original_chunk_id2))
                    if chunk_node2:
                        # Check if edge already exists
                        existing_edge = unified_graph.graph.get_edge_data(chunk_node1, chunk_node2)
                        if not existing_edge or existing_edge.get("relation") != "similar_to":
                            unified_graph.add_edge(chunk_node1, chunk_node2, 
                                                 relation="similar_to",
                                                 similarity=similarity,
                                                 inter_pdf=True)
                            unified_graph.add_edge(chunk_node2, chunk_node1,
                                                 relation="similar_to",
                                                 similarity=similarity,
                                                 inter_pdf=True)
                            similarity_edges_added += 1
    
    logger.info(f"✓ Added {similarity_edges_added} inter-PDF similarity edges")

# ---------------- VECTOR STORE MERGING ---------------- 
def merge_vector_stores(pdf_data: List[Dict], output_dir: Path) -> Chroma:
    """Merge all PDF vector stores into one unified vector store"""
    logger.info("Merging vector stores...")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # Create unified vector store
    unified_vector_db = output_dir / "vector_db" / "unified"
    unified_vector_db.mkdir(parents=True, exist_ok=True)
    
    unified_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(unified_vector_db)
    )
    
    # Add all chunks from all PDFs
    all_chunks = []
    total_chunks = sum(len(pdf_info["chunks"]) for pdf_info in pdf_data)
    logger.info(f"Preparing {total_chunks} chunks for vector store...")
    
    processed = 0
    for pdf_info in pdf_data:
        pdf_name = pdf_info["name"]
        chunks = pdf_info["chunks"]
        
        for chunk in chunks:
            # Update metadata to include PDF name
            chunk.metadata["pdf_name"] = pdf_name
            
            # Filter out complex metadata (lists, dicts) that ChromaDB doesn't support
            # ChromaDB only accepts: str, int, float, bool, None
            filtered_metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered_metadata[key] = value
                # Skip lists, dicts, and other complex types
            
            filtered_chunk = Document(
                page_content=chunk.page_content,
                metadata=filtered_metadata
            )
            all_chunks.append(filtered_chunk)
            processed += 1
            
            if processed % 50 == 0:
                logger.info(f"  Prepared {processed}/{total_chunks} chunks...")
    
    logger.info(f"Adding {len(all_chunks)} chunks to vector store (this may take a while)...")
    
    # Add chunks in batches to avoid memory issues and show progress
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        unified_store.add_documents(batch)
        logger.info(f"  Added batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size} ({min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks)")
    
    logger.info(f"✓ Merged {len(all_chunks)} chunks into unified vector store")
    return unified_store

# ---------------- MAIN PIPELINE ---------------- 
def main():
    """Main pipeline for regenerating unified graph from existing data"""
    # Default to output root
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("output")
    
    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Regenerating Unified Graph from Existing Vectorized Data")
    logger.info("=" * 80)
    logger.info(f"Source directory: {base_dir}")
    logger.info(f"Inter-cluster similarity threshold: {SIMILARITY_THRESHOLD}")
    logger.info("=" * 80)
    
    # Find all vectorized documents
    documents = find_vectorized_documents(base_dir)
    
    if not documents:
        logger.error(f"No vectorized documents found in {base_dir}")
        logger.error("Expected files: *_document_graph.json and *_vector_mapping.json")
        sys.exit(1)
    
    logger.info(f"Found {len(documents)} vectorized documents")
    
    # Create output directory
    output_dir = Path("output") / "unified" / "regenerated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all documents
    pdf_data = []
    unified_graph = UnifiedDocumentGraph()
    all_chunk_embeddings = {}
    chunk_id_mappings = {}
    
    for i, doc_info in enumerate(documents, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading document {i}/{len(documents)}: {doc_info['pdf_name']}")
        logger.info(f"{'='*80}")
        
        pdf_name = doc_info["pdf_name"]
        graph_file = doc_info["graph_file"]
        mapping_file = doc_info["mapping_file"]
        
        # Load document data
        doc_graph, chunks, chunk_embeddings = load_document_data(graph_file, mapping_file)
        if not doc_graph or not chunks:
            logger.warning(f"Skipping {pdf_name} - failed to load")
            continue
        
        # Store data
        pdf_data.append({
            "name": pdf_name,
            "chunks": chunks,
            "graph": doc_graph
        })
        all_chunk_embeddings[pdf_name] = chunk_embeddings
        
        # Merge graph into unified graph
        chunk_id_mapping = {}
        local_mapping = merge_graph_into_unified(pdf_name, doc_graph, chunks, unified_graph, chunk_id_mapping)
        chunk_id_mappings[pdf_name] = local_mapping
    
    if not pdf_data:
        logger.error("No documents were successfully loaded")
        sys.exit(1)
    
    logger.info(f"\n{'='*80}")
    logger.info("Computing inter-PDF similarity edges...")
    logger.info(f"{'='*80}")
    
    # Compute inter-PDF similarity edges
    compute_inter_pdf_similarity(unified_graph, all_chunk_embeddings, chunk_id_mappings)
    
    # Merge vector stores
    unified_vector_store = merge_vector_stores(pdf_data, output_dir)
    
    # Save unified graph
    graph_output_path = output_dir / "unified_graph.json"
    unified_graph.save(graph_output_path)
    
    # Save unified mapping
    unified_mapping = []
    global_chunk_id = 0
    for pdf_info in pdf_data:
        for chunk in pdf_info["chunks"]:
            chunk_data = {
                "vector_number": global_chunk_id,
                "pdf_name": pdf_info["name"],
                "heading": chunk.metadata.get("heading", ""),
                "summary": chunk.metadata.get("summary", ""),
                "section_path": chunk.metadata.get("section_path", ""),
                "content": chunk.page_content,
                "content_length": len(chunk.page_content),
                "metadata": {**chunk.metadata, "pdf_name": pdf_info["name"]}
            }
            unified_mapping.append(chunk_data)
            global_chunk_id += 1
    
    mapping_output_path = output_dir / "unified_mapping.json"
    with open(mapping_output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_mapping, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info("Regeneration Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Processed {len(pdf_data)} documents")
    logger.info(f"Total chunks: {len(unified_mapping)}")
    logger.info(f"Unified graph: {len(unified_graph.graph.nodes)} nodes, {len(unified_graph.graph.edges)} edges")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Graph file: {graph_output_path.name}")
    logger.info(f"Mapping file: {mapping_output_path.name}")
    logger.info(f"{'='*80}")
    
    # Visualize unified graph
    logger.info("\nGenerating visualization...")
    try:
        import subprocess
        # Pass the unified graph file directly to visualizeGraphE.py
        result = subprocess.run(
            [sys.executable, "visualizeGraphE.py", "--graph-file", str(graph_output_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✓ Visualization generated successfully")
        else:
            logger.warning(f"Visualization warning: {result.stderr}")
    except Exception as e:
        logger.warning(f"Could not generate visualization: {e}")

if __name__ == "__main__":
    main()
