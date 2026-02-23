# Testing the MCP server with MCP Inspector

MCP Inspector is a **browser-based UI** to list tools, call them with parameters, and inspect responses. It is **not** a Python package; it runs via Node.js.

## Prerequisites

- **Node.js** 18+ (e.g. from [nodejs.org](https://nodejs.org/) or `winget install OpenJS.NodeJS`).

## Install / run Inspector (no permanent install)

From any directory:

```bash
npx @modelcontextprotocol/inspector
```

A URL is printed (e.g. `http://localhost:6274` with a token). Open it in your browser.

## Connect to this project’s MCP server

This repo’s MCP server can run in two modes. Use **Streamable HTTP** for the Inspector.

### 1. Start the MCP server in HTTP mode

From the **project root** (`onpremdoc`), in a separate terminal:

```bash
cd D:\Projects\DocProcessing\zealits\onpremdoc
python mcp_server.py --http
```

You should see something like:

```
MCP server (Streamable HTTP) starting at http://127.0.0.1:8765/mcp
In MCP Inspector: choose 'Streamable HTTP' and URL: http://127.0.0.1:8765/mcp
```

Leave this process running.

### 2. Connect from MCP Inspector

1. In the Inspector UI, select transport **Streamable HTTP**.
2. In the **URL** field, enter: **`http://127.0.0.1:8765/mcp`**
3. Click **Connect** (do **not** open this URL in a new browser tab — the server returns 406 for normal browser requests).

After connecting you can:

- **List tools** – see `list_documents`, `get_document_info`, `query_document`, etc.
- **Call a tool** – e.g. run `list_documents` (no args) or `get_document_info` with a `document_id`.
- Inspect request/response and logs in the UI.

### Changing the HTTP port

To use another port (e.g. 9000):

```bash
set MCP_HTTP_PORT=9000
python mcp_server.py --http
```

Then in Inspector use: `http://127.0.0.1:9000/mcp`.

## Alternative: STDIO in Inspector

If you prefer not to run `--http`:

1. In Inspector, choose **STDIO**.
2. **Do not leave the default** — the Inspector starts with a sample server name like `mcp-server-everything`, which is not installed and will fail with `spawn mcp-server-everything ENOENT`.
3. Set:
   - **Command:** `python` (or full path to your Python, e.g. `D:\Projects\DocProcessing\zealits\onpremdoc\.venv\Scripts\python.exe`).
   - **Arguments:** `mcp_server.py` (or full path: `D:\Projects\DocProcessing\zealits\onpremdoc\mcp_server.py`).
   - **Working directory:** project root, e.g. `D:\Projects\DocProcessing\zealits\onpremdoc`.
4. Click **Connect**; the Inspector will spawn the server process.

This uses the same process as Cursor: stdio transport, one server per client.

---

## Troubleshooting

| What you see | What to do |
|--------------|------------|
| `spawn mcp-server-everything ENOENT` | You're on STDIO but still using the Inspector's default server. Change **Command** to `python`, **Arguments** to `mcp_server.py`, and **Working directory** to your project root (see STDIO section above). |
| `GET /mcp HTTP/1.1" 406 Not Acceptable` | You opened `http://127.0.0.1:8765/mcp` in a browser tab. Use the Inspector's URL field and click **Connect** instead; do not open the MCP URL in the browser. |
| Connection fails with Streamable HTTP | Prefer **STDIO** with Command=`python`, Args=`mcp_server.py`, cwd=project root. Or ensure the MCP server is running in another terminal with `python mcp_server.py --http` before clicking Connect. |
