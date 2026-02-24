# Chat PDF Frontend

React + Vite frontend for the OnPremDoc backend. PDF-only chat UI: upload a PDF, wait for processing and vectorization, then view the PDF and ask questions.

## Setup

```bash
npm install
```

## Development

Start the backend (from project root):

```bash
python -m uvicorn main:app --reload --port 8000
```

Start the frontend (from `frontend/`):

```bash
npm run dev
```

The app will be at http://localhost:5173. API requests are proxied to http://127.0.0.1:8000 via Vite config.

To use a different API URL, set `VITE_API_URL` (e.g. in `.env`):

```
VITE_API_URL=http://localhost:8000
```

## Build

```bash
npm run build
```

Output is in `dist/`. Serve it with any static host; set `VITE_API_URL` to your backend URL for production.
