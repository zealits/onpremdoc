"""
Fill Document_Processing_Pipeline_Steps.xlsx:
- Pipeline Steps: Step #, Step Name, Description, Module/File, Models & Technologies,
  Algorithms & Methods, Key Functions, Input, Output, Examples.
- Summary: Remove re-ranking; no mention of reranking (pipeline does not use it).
- All descriptions in-depth; simple examples for each step.
"""
import openpyxl
from pathlib import Path

# Step#, Step Name, Module/File, Models & Technologies, Algorithms & Methods, Key Functions
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
    (25, "Cap initial chunks", "retrivalAgentE.py", "—", "Select top 25 chunks from combined vector + graph results", "retrieved_chunks[:25], state['retrieved_chunks']"),
    (26, "Chunk analysis", "retrivalAgentE.py", "Ollama LLM", "Shortened prompt; need more info?, single follow-up query", "analyze_chunks(), should_continue_search()"),
    (27, "Second retrieval", "retrivalAgentE.py", "Chroma, NetworkX", "New query vector search, graph expand; cap 15 chunks", "second_retrieval(), similarity_search_with_score(new_query, k=15)"),
    (28, "Generate answer", "retrivalAgentE.py", "Ollama LLM", "Question in prompt (first line + repeated); primary + second chunks", "generate_final_answer(), answer_prompt with QUESTION block"),
    (29, "Format response", "main.py", "FastAPI, Pydantic", "ChunkDetail, retrieval_stats, debug_info", "QueryResponse(), ChunkDetail(), retrieval_stats"),
    (30, "Page summarization", "main.py, page_summarization.py", "Ollama LLM, DocumentGraph", "Page chunks or adjacent, LLM summary + key points", "load_page_agent(), summarize_page(), _generate_summary_with_llm()"),
]

# In-depth descriptions (no mention of re-ranking). Column C.
DESCRIPTIONS = {
    1: "User uploads a PDF file via the FastAPI endpoint. The file is saved under a document-specific folder with a unique UUID. The API returns the document_id and status so the client can poll or use the document for later steps (e.g. vectorization, query).",
    2: "A background task is started so the API can respond immediately. The task runs PDF processing: it converts the PDF to markdown, fixes tables, and extracts a line-to-page mapping. Results are written to the document folder (e.g. .md file and _page_mapping.json).",
    3: "Docling converts the PDF into structured markdown. It extracts text and layout and inserts page-break placeholders. This preserves document structure (headings, lists, tables) and gives a consistent format for downstream chunking and graph building.",
    4: "Heading levels from the PDF TOC or bookmarks are used to correct markdown heading hierarchy. This step ensures sections like 'Part A', 'Section 1.2' have the right nesting so the vectorizer can build an accurate section graph.",
    5: "Markdown tables are detected and normalized: separator rows, header rows, and grouping are fixed. Trailing empty columns are trimmed. Output is clean markdown suitable for chunking without broken table boundaries.",
    6: "Page-break markers in the markdown are used to build a line-to-page map. Each line (or block) is assigned a page number. This supports page-level features (e.g. page summaries, page classification) and citation by page.",
    7: "The API triggers the vectorization workflow for a document (e.g. after upload or on demand). A LangGraph workflow is created and run in the background. Initial state includes the document_id and paths to the markdown and page mapping.",
    8: "The markdown file and page mapping are loaded. Markdown is parsed with structure awareness (headings, lists, tables). Each segment is associated with a section and page via the structure and page mapping.",
    9: "The full document structure is extracted from the markdown: sections and headers are detected with regex, and a full hierarchy path is built for each (e.g. 'Part A > Section 2 > 2.1'). This drives section nodes and chunk–section links in the graph.",
    10: "Content is split into fixed-size chunks (e.g. 2000 chars, 150 overlap). Empty or duplicate chunks are dropped. Each chunk gets metadata: heading, section_path, page_number, prev/next chunk IDs. Chunks are assigned to the most specific section containing their start line.",
    11: "Ollama LLM and embedding model are initialized (e.g. llama3.1:8b, nomic-embed-text:v1.5). Chroma is set up with a persist directory. These are used for chunk summarization, embedding, and later for the retrieval agent.",
    12: "A directed graph is built from the document structure: one node per section, with edges from parent to child (e.g. 'Part A' → 'Section 2'). This supports graph expansion at query time (e.g. follow section links from a seed chunk).",
    13: "Each page is classified by the LLM into a short label (1–4 words) based on its content. Labels are stored in chunk metadata and used for display or filtering (e.g. 'Terms and conditions', 'Claim form').",
    14: "For each chunk, the pipeline generates a one-line summary (LLM), computes an embedding (nomic-embed-text), adds the chunk to Chroma, and adds a chunk node and edges to the graph (section, page, prev/next, and later similarity).",
    15: "The LLM produces a single concise summary line per chunk, within a token limit (e.g. 500). Summaries are stored in chunk metadata and used in retrieval (e.g. in analysis or answer prompts) and in the vector mapping JSON.",
    16: "Chunk text is checked against token/character limits for the embedding model; if over, it is truncated. The embedding is computed and the chunk (with metadata) is added to the Chroma collection.",
    17: "Each chunk is added as a node in the graph. Edges link chunk to its section (belongs_to), to its page (on_page), and to previous/next chunks (follows). These support graph expansion (e.g. 'give me adjacent chunks').",
    18: "For each chunk, a vector search finds other chunks with L2 distance below a threshold (e.g. 0.50). Pairs are connected with a 'similar_to' edge. This adds semantic neighbourhoods to the graph for expansion.",
    19: "The document graph is saved as JSON (nodes and edges). The vector mapping (chunks with metadata and optional summaries) is saved as JSON. The Chroma collection is persisted to disk so the retrieval agent can load it.",
    20: "The user sends a query (and optional document_id, flags) via the API. The request is validated and the agent for that document is loaded (vector store, graph, chunks, LLM). The query is passed into the agent state.",
    21: "For the given document, the vector store (Chroma), document graph (NetworkX), chunk list (from vector mapping JSON), and LLM are loaded. If configured, the page summarization agent is also loaded. These are stored for use by the retrieval nodes.",
    22: "The user query is classified as either a page-summary request (e.g. 'summarize page 5') or a normal retrieval request. Pattern matching and optionally the LLM are used. The result routes to either page summarization or the retrieval pipeline.",
    23: "The query is embedded and compared to chunk embeddings in Chroma. The top-k (e.g. 20) chunks by L2 distance are taken as seeds. Distances are converted to similarity scores and stored for logging or display.",
    24: "From the top seed chunks (e.g. top 5), the graph is traversed: parent sections, adjacent chunks (prev/next), and similar chunks (similar_to). Expansion is capped (e.g. 15 extra chunks) to keep context size manageable.",
    25: "The combined set of seed and expanded chunks is trimmed to a fixed limit (e.g. 25) for the first pass. Order is preserved (e.g. seeds first). These chunks are stored in state and passed to chunk analysis. No LLM scoring step is used.",
    26: "The LLM analyzes the current chunks with a short prompt: do they sufficiently answer the query? If not, it produces a single follow-up query. The result (needs_more_info, new_query) drives whether a second retrieval is run.",
    27: "When the analysis requests more information, a second retrieval is run with the new query: vector search (e.g. k=15) and graph expansion from those seeds, excluding chunks already in the first pass. Results are capped (e.g. 15) and appended to state.",
    28: "The final answer is generated by the LLM from the selected chunks (primary plus any second retrieval). The user question is placed at the top of the prompt and repeated before the answer instruction. Only information from the chunks is used; chunk numbers are not cited in the answer.",
    29: "The agent output (final answer, chunk list, retrieval stats, debug info) is mapped into a structured API response: answer text, optional chunk details (with metadata), retrieval statistics, and optional second query used.",
    30: "For page-summary queries, the page summarization agent loads chunks for the requested page (and optionally adjacent pages). The LLM produces a short summary, key points, and section labels. The result is returned as the final answer.",
}

# Input (column H) and Output (column I)
INPUT_OUTPUT = {
    1: ("PDF file (UploadFile)", "Saved PDF path, document_id, ProcessPDFResponse"),
    2: ("PDF file path", "Markdown file (.md), Page mapping JSON (_page_mapping.json)"),
    3: ("PDF file", "Raw markdown with page break markers (<!-- page break -->)"),
    4: ("Docling document object", "Document with corrected heading hierarchy"),
    5: ("Markdown text with tables", "Fixed markdown with properly formatted tables"),
    6: ("Markdown with page break markers", "Page mapping JSON: {line_to_page, page_boundaries, total_pages}"),
    7: ("document_id", "VectorizerState with markdown_file path"),
    8: ("Markdown file path, page mapping JSON", "Parsed chunks, document structure dict"),
    9: ("Markdown content", "Structure dict: {sections, headers, tables} with full paths"),
    10: ("Markdown content, structure, page mapping", "List of Document chunks with metadata (heading, section_path, page_number, etc.)"),
    11: ("Model names, base URL", "Initialized LLM, embeddings, vector store, DocumentGraph"),
    12: ("Document structure (sections)", "Graph with section nodes and hierarchical edges"),
    13: ("Chunks grouped by page, page mapping", "Dictionary: {page_number: classification_label}"),
    14: ("Chunk content", "Embedded chunk in Chroma, chunk node and edges in graph, JSON mapping entry"),
    15: ("Chunk content (max 2000 chars)", "One-line summary string"),
    16: ("Chunk content (max 2000 chars, 1500 tokens)", "Vector embedding stored in Chroma"),
    17: ("Chunk metadata (section_path, page_number, prev/next chunk IDs)", "Graph nodes and edges for chunks"),
    18: ("Chunk embeddings", "Graph edges with similarity scores (similar_to relation)"),
    19: ("DocumentGraph, JSON mapping, vector store", "Graph JSON file, vector mapping JSON file, persisted Chroma DB"),
    20: ("Query string, document_id", "Initialized AgentState with query"),
    21: ("Vector DB path, mapping file, graph file", "Loaded vector store, graph, chunks, LLM, page agent"),
    22: ("User query", "is_page_summary (bool), page_number (optional)"),
    23: ("Query string", "Seed chunk IDs, similarity scores"),
    24: ("Seed chunk IDs", "Expanded chunk IDs (via graph edges)"),
    25: ("Combined seed + expanded chunk list", "Top 25 chunks in state['retrieved_chunks']"),
    26: ("Query, retrieved chunks (top 12)", "Analysis text, needs_more_info (bool), new_query (optional)"),
    27: ("New query, existing chunk IDs (to avoid duplicates)", "Additional chunks (up to 15) in state['second_retrieval_chunks']"),
    28: ("Query, primary chunks, second chunks (if any)", "Final answer text"),
    29: ("Final answer, chunks, stats", "QueryResponse JSON"),
    30: ("Page number", "Page summary with key points, sections, classification"),
}

# Simple examples. Column J (new column).
EXAMPLES = {
    1: "User selects policy.pdf in the UI → POST /upload → response: { document_id: 'a1b2c3-...', status: 'processing' }.",
    2: "document_id triggers background job → task reads PDF from output/a1b2c3.../file.pdf → writes output/a1b2c3.../file.md and _page_mapping.json.",
    3: "policy.pdf → Docling → markdown with '## Part A', '<!-- page break -->', tables, lists.",
    4: "PDF bookmarks: 'Part A' (L1), 'Section 1' (L2) → markdown headings adjusted to # Part A, ## Section 1.",
    5: "Markdown table with broken separators → fix_table() → valid markdown table with aligned columns.",
    6: "Markdown with 10 page breaks → extract_page_mapping_from_markdown() → { line_to_page: {0:1, 45:2, ...}, total_pages: 10 }.",
    7: "POST /vectorize { document_id } → 202 Accepted → background workflow starts; client polls status.",
    8: "load_markdown('output/.../file.md') + page_mapping → list of (text, section_path, page_number) segments.",
    9: "Markdown with ## Part A, ### 1.1 Definitions → structure: sections with full paths like 'Part A > 1.1 Definitions'.",
    10: "8000-char section split into 4 chunks (2000 chars, 150 overlap), each with section_path 'Part A > 2. Payment', page_number, prev/next chunk IDs.",
    11: "Ollama(base_url), OllamaEmbeddings(model='nomic-embed-text'), Chroma(persist_directory='.../vector_db') created.",
    12: "Nodes: section:Part A, section:Section 2; edges: Part A → Section 2 (contains_section).",
    13: "Page 3 chunks → LLM → page_classification[3] = 'Terms and conditions'.",
    14: "Chunk 'The policy shall be governed...' → summary generated, embedded, added to Chroma and graph with edges to section and page.",
    15: "Chunk text → LLM → 'Clause stating governing law is India.'.",
    16: "Chunk (1800 chars) → truncated if needed → embedding vector → Chroma.add_documents([doc]).",
    17: "Chunk 12 → edges: chunk:12 --belongs_to--> section:10. Jurisdiction, chunk:12 --on_page--> page:5, chunk:12 --follows--> chunk:11.",
    18: "Chunk 5 embedding → similarity_search_with_score(k=10) → chunks 3, 7, 9 within threshold → add similar_to edges.",
    19: "document_graph.save('.../graph.json'), json.dump(mapping, '.../vector_mapping.json'), Chroma persists to disk.",
    20: "POST /query { document_id, query: 'What is the grace period?' } → load agent → state = { query: '...', ... }.",
    21: "document_id → find vector_db path, mapping JSON, graph JSON → load Chroma, DocumentGraph, chunks list, Ollama LLM.",
    22: "Query 'Summarize page 5' → is_page_summary=True, page_number=5 → route to summarize_page. Query 'What is the grace period?' → normal_retrieval.",
    23: "Query 'grace period' → vector_search(k=20) → seed_chunk_ids=[7, 12, 3, ...], seed_chunk_scores={7:0.82, 12:0.78, ...}.",
    24: "Seeds [7, 12, 3, 40, 11] → expand_from_chunks() → add parent sections, prev/next chunks, similar_to neighbours → graph_expanded_ids (e.g. 15 more).",
    25: "28 chunks (20 seeds + 15 expanded, some overlap) → take first 25 → state['retrieved_chunks'] = list of 25 Document objects.",
    26: "Query 'governing law'; 25 chunks → LLM: 'Partially; missing which country.' → needs_more_info=True, new_query='Which country laws govern this policy?'.",
    27: "new_query='Which country laws govern...' → vector_search(k=15), graph expand → 4 new chunks (excluding already retrieved) → second_retrieval_chunks.",
    28: "Query 'policy governed by which country?' + 29 chunks → prompt starts with 'QUESTION: policy governed by which country?' and repeats it → LLM returns answer citing document only.",
    29: "final_answer, retrieved_chunks, second_retrieval_chunks, debug_info → QueryResponse(answer=..., retrieval_stats={...}, chunks_detail=[...]).",
    30: "Query 'Summarize page 5' → load chunks for page 5 (and maybe 4, 6) → LLM → 'Page 5 covers Premium payment. Key points: ...'.",
}


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
    # Add Examples header if column 10 not present
    if ws.max_column < 10:
        ws.cell(row=1, column=10, value="Examples")
    else:
        ws.cell(row=1, column=10, value="Examples")

    for i, row_data in enumerate(STEPS):
        step_num, step_name, module_file, models_tech, algorithms, key_functions = row_data
        row_idx = i + 2
        ws.cell(row=row_idx, column=1, value=step_num)
        ws.cell(row=row_idx, column=2, value=step_name)
        ws.cell(row=row_idx, column=3, value=DESCRIPTIONS.get(step_num, ""))
        ws.cell(row=row_idx, column=4, value=module_file)
        ws.cell(row=row_idx, column=5, value=models_tech)
        ws.cell(row=row_idx, column=6, value=algorithms)
        ws.cell(row=row_idx, column=7, value=key_functions)
        inp_out = INPUT_OUTPUT.get(step_num, ("", ""))
        ws.cell(row=row_idx, column=8, value=inp_out[0])
        ws.cell(row=row_idx, column=9, value=inp_out[1])
        ws.cell(row=row_idx, column=10, value=EXAMPLES.get(step_num, ""))

    # Summary sheet: remove any mention of re-ranking (pipeline does not use it)
    if "Summary" in wb.sheetnames:
        sum_ws = wb["Summary"]
        for row_idx in range(1, sum_ws.max_row + 1):
            val = sum_ws.cell(row=row_idx, column=1).value
            if val and "re-ranking" in str(val).lower():
                sum_ws.cell(row=row_idx, column=1).value = "- Chunk selection (top-k from vector + graph)"
                if sum_ws.cell(row=row_idx, column=2).value:
                    sum_ws.cell(row=row_idx, column=2).value = None
                break

    wb.save(path)
    print("Filled Pipeline Steps: descriptions, examples, no re-ranking. Saved:", path.absolute())


if __name__ == "__main__":
    main()
