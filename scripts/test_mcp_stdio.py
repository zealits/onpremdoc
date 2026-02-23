"""
Optional: test MCP server over stdio by sending JSON-RPC requests.
Run from project root: python scripts/test_mcp_stdio.py
Requires: pip install -r requirements-mcp.txt (and full project deps).
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def send_request(proc, method: str, params: dict | None = None) -> dict:
    req = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        req["params"] = params
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    return json.loads(line)


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "mcp_server.py")],
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Initialize (MCP often expects initialize first)
        init_resp = send_request(proc, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0.1.0"}})
        if "result" in init_resp:
            print("Initialize OK")
        # List tools (optional)
        tools_resp = send_request(proc, "tools/list")
        if "result" in tools_resp:
            names = [t.get("name") for t in tools_resp["result"].get("tools", [])]
            print("Tools:", names)
        # Call list_documents
        list_resp = send_request(proc, "tools/call", {"name": "list_documents", "arguments": {}})
        if "error" in list_resp:
            print("list_documents error:", list_resp["error"])
            return 1
        content = list_resp.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            docs = json.loads(content[0].get("text", "[]"))
            print("list_documents result:", json.dumps(docs, indent=2))
        else:
            print("list_documents result:", list_resp.get("result"))
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
