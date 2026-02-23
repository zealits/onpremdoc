"""
MCP (Model Context Protocol) server for the Document Processing Pipeline.
Exposes pipeline operations as tools for AI clients (e.g. Cursor, Claude Desktop).

Usage:
  stdio (default, for Cursor):  python mcp_server.py
  Streamable HTTP (for MCP Inspector):  python mcp_server.py --http
  Then in MCP Inspector use URL: http://127.0.0.1:8765/mcp
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Prevent deprecation/other warnings from being written to stdout and breaking
# the MCP stdio JSON-RPC stream (client expects only JSON on the protocol stream).
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*was deprecated.*")

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from services import document_service

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("MCP SDK not installed. Run: pip install -r requirements-mcp.txt", file=sys.stderr)
    sys.exit(1)

# Port for Streamable HTTP (avoid 8000 used by FastAPI main.py)
MCP_HTTP_PORT = int(os.environ.get("MCP_HTTP_PORT", "8765"))

mcp = FastMCP(
    "Document Processing Pipeline",
    json_response=True,
    host="127.0.0.1",
    port=MCP_HTTP_PORT,
)


@mcp.tool()
def list_documents() -> str:
    """List all processed documents. Returns a JSON list with document_id, status, total_pages, total_chunks for each."""
    documents = []
    if document_service.OUTPUT_DIR.exists():
        for doc_dir in document_service.OUTPUT_DIR.iterdir():
            if doc_dir.is_dir():
                doc_info = document_service.get_document_info(doc_dir.name)
                if doc_info:
                    documents.append({
                        "document_id": doc_info["document_id"],
                        "name": doc_info["name"],
                        "status": doc_info["status"],
                        "total_pages": doc_info.get("total_pages"),
                        "total_chunks": doc_info.get("total_chunks"),
                    })
    return json.dumps(documents, indent=2)


@mcp.tool()
def get_document_info(document_id: str) -> str:
    """Get metadata for one document (status, paths, stats). Returns JSON or an error message if not found."""
    doc_info = document_service.get_document_info(document_id)
    if not doc_info:
        return json.dumps({"error": f"Document '{document_id}' not found"})
    return json.dumps(doc_info, indent=2)


@mcp.tool()
def upload_document(file_path: str) -> str:
    """Upload a PDF from a local file path. Copies the file, runs detection (steps 3–6), and returns the document_id."""
    try:
        doc_id = document_service.upload_pdf_from_path(Path(file_path))
        return json.dumps({"document_id": doc_id, "status": "processing", "message": "PDF uploaded and processing complete"})
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Upload failed: {e}"})


@mcp.tool()
def vectorize_document(document_id: str) -> str:
    """Run vectorization for a document. Synchronous; may take several minutes for large documents."""
    try:
        document_service.trigger_vectorize(document_id)
        return json.dumps({"document_id": document_id, "status": "ok", "message": "Vectorization complete"})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Vectorization failed: {e}"})


@mcp.tool()
def query_document(document_id: str, query: str, include_chunks: bool = True) -> str:
    """RAG query over a vectorized document. Returns answer, retrieval_stats, and optional chunks."""
    try:
        result = document_service.query_document(document_id, query, include_chunks=include_chunks)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Query failed: {e}"})


@mcp.tool()
def summarize_page(document_id: str, page_number: int) -> str:
    """Generate a page-level summary for a given page number (1-based)."""
    try:
        result = document_service.summarize_page(document_id, page_number)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Summarize failed: {e}"})


@mcp.tool()
def get_graph_stats(document_id: str) -> str:
    """Get graph node/edge counts and optional similarity stats for a document."""
    try:
        stats = document_service.get_graph_stats(document_id)
        return json.dumps(stats, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Failed to get stats: {e}"})


@mcp.tool()
def get_document_markdown(document_id: str) -> str:
    """Return the full markdown content of the document (useful for the AI to read full text)."""
    try:
        content = document_service.get_document_markdown(document_id)
        return content
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Failed to get markdown: {e}"})


# ----- Optional MCP resources -----

@mcp.resource("document://list")
def resource_document_list() -> str:
    """Resource: list of document IDs and short info (JSON)."""
    documents = []
    if document_service.OUTPUT_DIR.exists():
        for doc_dir in document_service.OUTPUT_DIR.iterdir():
            if doc_dir.is_dir():
                info = document_service.get_document_info(doc_dir.name)
                if info:
                    documents.append({
                        "document_id": info["document_id"],
                        "name": info["name"],
                        "status": info["status"],
                        "total_pages": info.get("total_pages"),
                        "total_chunks": info.get("total_chunks"),
                    })
    return json.dumps(documents, indent=2)


@mcp.resource("document://{document_id}/markdown")
def resource_document_markdown(document_id: str) -> str:
    """Resource: full markdown content for one document."""
    try:
        return document_service.get_document_markdown(document_id)
    except (ValueError, Exception) as e:
        return json.dumps({"error": str(e)})


@mcp.resource("document://{document_id}/info")
def resource_document_info(document_id: str) -> str:
    """Resource: JSON metadata for one document."""
    info = document_service.get_document_info(document_id)
    if not info:
        return json.dumps({"error": f"Document '{document_id}' not found"})
    return json.dumps(info, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Document Processing Pipeline MCP server (stdio or Streamable HTTP)."
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run with Streamable HTTP transport so MCP Inspector can connect at http://127.0.0.1:%s/mcp" % MCP_HTTP_PORT,
    )
    args = parser.parse_args()
    if args.http:
        print("MCP server (Streamable HTTP) starting at http://127.0.0.1:%s/mcp" % MCP_HTTP_PORT, file=sys.stderr)
        print("In MCP Inspector: choose 'Streamable HTTP' and URL: http://127.0.0.1:%s/mcp" % MCP_HTTP_PORT, file=sys.stderr)
        mcp.run(transport="streamable-http")
    else:
        # Stdio mode: ensure no logging goes to stdout (reserved for MCP JSON-RPC)
        import logging
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.root.addHandler(h)
        logging.root.setLevel(logging.INFO)
        mcp.run()


if __name__ == "__main__":
    main()
