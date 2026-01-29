"""
Fill Document_Processing_Pipeline_Steps.xlsx Sheet 2 (Pipeline Steps) with
Step #, Step Name, Module/File, Models & Technologies, Algorithms & Methods, Key Functions.
Based on full backend codebase analysis.
"""
import openpyxl
from pathlib import Path

# Pipeline steps data: (Step#, Step Name, Module/File, Models & Technologies, Algorithms & Methods, Key Functions)
# Columns A, B, D, E, F, G (C=Description, H=Input, I=Output already in file)
STEPS = [
    (1, "Upload PDF", "main.py", "FastAPI", "File upload, UUID generation", "upload_pdf(), File(), UploadFile"),
    (2, "Background PDF processing", "main.py, detection.py", "—", "Background task scheduling", "process_pdf_background(), process_single_pdf()"),
    (3, "PDF to Markdown conversion", "detection.py", "Docling (DocumentConverter)", "Native text extraction, layout parsing", "DocumentConverter.convert(), export_to_markdown(page_break_placeholder)"),
    (4, "Hierarchical heading correction", "detection.py", "docling-hierarchical (ResultPostprocessor)", "TOC/bookmark-based heading level inference", "ResultPostprocessor(doc, source=pdf_path).process()"),
    (5, "Markdown table fixing", "detection.py", "—", "Table detection, separator/header/group row handling", "process_markdown(), fix_table(), trim_trailing_empty_columns()"),
    (6, "Page mapping extraction", "detection.py", "—", "Line-to-page mapping from page break markers", "extract_page_mapping_from_markdown(), create_approximate_page_mapping()"),
    (7, "Trigger vectorization", "main.py", "FastAPI, LangGraph", "Background task, state initialization", "vectorize_document(), vectorize_background(), create_vectorization_workflow()"),
    (8, "Load markdown and chunk", "vectorizerE.py", "—", "Markdown parsing, structure-aware chunking", "load_markdown(), parse_markdown_enhanced(), get_section_for_line()"),
    (9, "Extract document structure", "vectorizerE.py", "—", "Regex header detection, full hierarchy path building", "extract_document_structure(), header match (#+), section path tree"),
    (10, "Create enhanced chunks", "vectorizerE.py", "—", "Size-based splitting, overlap, section assignment", "create_enhanced_chunk(), is_chunk_empty(), adjacency metadata"),
    (11, "Initialize LLM and vector store", "vectorizerE.py", "Ollama (llama3.1:8b), nomic-embed-text:v1.5, Chroma", "Embedding model, LLM, persistent vector DB", "Ollama(), OllamaEmbeddings(), Chroma(persist_directory)"),
    (12, "Build section graph", "vectorizerE.py", "NetworkX", "Section nodes, parent-child edges (contains_section)", "add_section_node(), add_edge(relation='contains_section')"),
    (13, "Page classification", "vectorizerE.py", "Ollama LLM", "Per-page content classification (1–4 word labels)", "classify_pages(), classify_page_with_llm()"),
    (14, "Process chunks (embed + graph)", "vectorizerE.py", "Ollama, Chroma, NetworkX", "Per-chunk: summary, embed, add to store and graph", "process_chunks_one_by_one() loop, add_chunk_node(), add_edge()"),
    (15, "Chunk summary generation", "vectorizerE.py", "Ollama LLM", "One-line summary, token limit 500", "get_summary_from_llm(), token_tracker.check_llm_limit()"),
    (16, "Chunk embedding", "vectorizerE.py", "Ollama (nomic-embed-text:v1.5), Chroma", "Token/char limit check, vector storage", "token_tracker.check_embedding_limit(), vector_store.add_documents()"),
    (17, "Chunk graph edges", "vectorizerE.py", "NetworkX", "belongs_to section, on_page, follows prev/next", "add_chunk_node(), add_edge(contains, belongs_to, on_page, follows)"),
    (18, "Similarity edges", "vectorizerE.py", "Chroma, NetworkX", "L2 distance → similarity, threshold 0.50, similar_to edges", "similarity_search_with_score(), add_edge(relation='similar_to')"),
    (19, "Persist outputs", "vectorizerE.py", "Chroma, JSON", "Graph JSON, vector mapping JSON, Chroma persist", "document_graph.save(), json.dump(), Chroma persist_directory"),
    (20, "Submit query", "main.py", "FastAPI", "Request validation, agent loading", "query_document(), load_agent_for_document()"),
    (21, "Load agent resources", "main.py, retrivalAgentE.py", "Chroma, NetworkX, Ollama", "Vector store, graph, chunks, LLM, page agent", "load_chunks_from_mapping(), load_vector_store(), DocumentGraph.load(), load_page_agent()"),
    (22, "Query classification", "retrivalAgentE.py", "Ollama LLM, regex", "Page-summary vs normal retrieval", "classify_query(), route_query_type()"),
    (23, "Vector similarity search", "retrivalAgentE.py", "Chroma, nomic-embed-text", "L2 distance, k=20, distance_to_similarity()", "similarity_search_with_score(query, k=20)"),
    (24, "Graph expansion", "retrivalAgentE.py", "NetworkX", "BFS from top-5 seeds: sections, adjacent, similar; max 15", "expand_from_chunks(), get_parent_section(), get_adjacent_chunks(), get_similar_chunks()"),
    (25, "Re-rank by relevance", "retrivalAgentE.py", "Ollama LLM", "LLM relevance scoring 0.0–1.0, top 25", "rerank_chunks_by_relevance(), rerank_scores"),
    (26, "Chunk analysis", "retrivalAgentE.py", "Ollama LLM", "Need more info?, new_query generation", "analyze_chunks(), should_continue_search()"),
    (27, "Second retrieval", "retrivalAgentE.py", "Chroma, NetworkX, Ollama", "New query vector search, graph expand, rerank", "second_retrieval(), similarity_search_with_score(new_query, k=15)"),
    (28, "Generate answer", "retrivalAgentE.py", "Ollama LLM", "Context from top 25 chunks, prompt with query", "generate_final_answer()"),
    (29, "Format response", "main.py", "FastAPI, Pydantic", "ChunkDetail, retrieval_stats, debug_info", "QueryResponse(), ChunkDetail(), retrieval_stats"),
    (30, "Page summarization", "main.py, page_summarization.py", "Ollama LLM, DocumentGraph", "Page chunks or adjacent, LLM summary + key points", "load_page_agent(), summarize_page(), _generate_summary_with_llm()"),
]

def main():
    path = Path("Document_Processing_Pipeline_Steps.xlsx")
    if not path.exists():
        print("File not found:", path.absolute())
        return

    wb = openpyxl.load_workbook(path)
    if "Pipeline Steps" not in wb.sheetnames:
        print("Sheet 'Pipeline Steps' not found")
        return

    ws = wb["Pipeline Steps"]
    # Row 1 = header. Data rows 2–31 (steps 1–30).
    for i, row_data in enumerate(STEPS):
        step_num, step_name, module_file, models_tech, algorithms, key_functions = row_data
        row_idx = i + 2
        ws.cell(row=row_idx, column=1, value=step_num)
        ws.cell(row=row_idx, column=2, value=step_name)
        ws.cell(row=row_idx, column=4, value=module_file)
        ws.cell(row=row_idx, column=5, value=models_tech)
        ws.cell(row=row_idx, column=6, value=algorithms)
        ws.cell(row=row_idx, column=7, value=key_functions)

    wb.save(path)
    print("Filled Pipeline Steps (Sheet 2). Saved:", path.absolute())

if __name__ == "__main__":
    main()
