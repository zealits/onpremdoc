# Document Processing Pipeline

FastAPI-based pipeline for PDF upload, conversion to markdown, vectorization (Chroma + graph), and retrieval over documents. Supports **Ollama** (local) or **Hugging Face** (Inference API) for embeddings and LLM.

## Quick start

1. **Environment**
   - Copy `.env.example` to `.env` and set your tokens if using Hugging Face.
   - Default is Ollama; set `INFERENCE_PROVIDER=huggingface` and `HUGGINGFACEHUB_API_TOKEN` for HF.

2. **Install**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run API**
   ```bash
   python main.py
   # or: uvicorn main:app --reload
   ```

4. **Endpoints**
   - Upload PDF → `POST /upload`
   - Trigger vectorization → `POST /vectorize`
   - Query document → `POST /query`

## MCP server (AI clients)

The pipeline is also exposed as an **MCP (Model Context Protocol)** server so AI clients (e.g. Cursor, Claude Desktop) can call the same operations as tools.

1. **Install MCP dependency** (after base deps):
   ```bash
   pip install -r requirements-mcp.txt
   ```

2. **Run the MCP server** (stdio; use project root as working directory):
   ```bash
   cd /path/to/onpremdoc
   python mcp_server.py
   ```
   The server reads JSON-RPC from stdin and writes to stdout. Cursor (or another client) typically runs this as a subprocess.

3. **Use with Cursor**  
   This repo includes **`.cursor/mcp.json`** so Cursor can run the MCP server when you have this project open.

   - **Open the project** in Cursor (e.g. `File > Open Folder` → `onpremdoc`).
   - **Turn on MCP:** Cursor Settings → **Features** → enable **MCP** (or **Cursor Settings** → **MCP**). The `onpremdoc` server should appear; enable it if needed.
   - Cursor will start `mcp_server.py` via stdio (using the Python and `cwd` in `.cursor/mcp.json`). You can then use the tools from the AI chat (e.g. “list my documents”, “query document X with …”).

   If you cloned the repo to a **different path**, edit `.cursor/mcp.json` and set `command` and `cwd` to your project root and venv Python (e.g. `"command": "C:/your/path/onpremdoc/.venv/Scripts/python.exe"`, `"cwd": "C:/your/path/onpremdoc"`).

**Tools:** `list_documents`, `get_document_info`, `upload_document` (local `file_path`), `vectorize_document`, `query_document`, `summarize_page`, `get_graph_stats`, `get_document_markdown`.

**Resources (optional):** `document://list`, `document://{document_id}/markdown`, `document://{document_id}/info`.

### Testing with MCP Inspector (visual UI)

The official **MCP Inspector** is a Node.js app (not a pip package). Use it to visually list tools, call them, and inspect responses.

**Option A – Streamable HTTP (recommended)**

1. Install Node.js (18+), then run the inspector (no install):
   ```bash
   npx @modelcontextprotocol/inspector
   ```
2. In another terminal, start this MCP server in HTTP mode (from project root):
   ```bash
   python mcp_server.py --http
   ```
3. In the Inspector UI (URL shown in the terminal, often `http://localhost:6274`), choose **Streamable HTTP** and enter:
   ```text
   http://127.0.0.1:8765/mcp
   ```
4. Click **Connect**. You can then list and run tools from the UI.

**Option B – STDIO**

In the Inspector, choose **STDIO**, set **Command** to `python`, **Arguments** to `mcp_server.py`, and **Working directory** to your project root (e.g. `D:\Projects\DocProcessing\zealits\onpremdoc`), then connect.

See **docs/MCP_INSPECTOR.md** for more detail.

## Project layout

```
onpremdoc/
├── main.py                 # FastAPI app (upload, vectorize, query)
├── mcp_server.py           # MCP server (stdio) for AI clients
├── requirements-mcp.txt    # MCP SDK (install after requirements.txt)
├── services/
│   └── document_service.py # Shared pipeline logic (main + MCP)
├── detection.py            # PDF → markdown, page mapping
├── vectorizerE.py          # Chunking, Chroma, document graph (Plan E)
├── retrivalAgentE.py       # Retrieval agent (vector + graph expansion)
├── page_summarization.py   # Page-level summaries
├── economics_tracker.py    # Usage / economics logging
├── visualizeGraphE.py     # Graph visualization (CLI + API)
├── regenerate_unified.py   # Regenerate unified graph from vectorized docs
├── config/
│   ├── __init__.py
│   └── inference_config.py # Ollama / Hugging Face backend (get_llm, get_embeddings)
├── docs/                   # Additional documentation
├── scripts/                # Standalone/maintenance scripts
├── frontend/               # Frontend app (if any)
├── output/                 # Processed documents (gitignored)
├── .env.example
├── .gitignore
└── requirements.txt
```

## Configuration

- **Inference:** `config/inference_config.py` reads `INFERENCE_PROVIDER`, `OLLAMA_*`, `HF_*`, `HUGGINGFACEHUB_API_TOKEN` (see `.env.example`).
- **Output:** Processed documents live under `output/{document_id}/` with markdown, vector DB, and graph JSON.

## License

Internal / project-specific.
==============================================================================
standalone commands to check standalone files 


1)python detection.py .\HDFC-Life-Cancer-Care-101N106V04-Policy-Document.pdf
2)python .\vectorizerE.py .\output\HDFC-Life-Cancer-Care-101N106V04-Policy-Document
3)python .\visualizeGraphE.py .\output\HDFC-Life-Cancer-Care-101N106V04-Policy-Document\