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

## Project layout

```
onpremdoc/
├── main.py                 # FastAPI app (upload, vectorize, query)
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