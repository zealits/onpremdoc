# docling-deepseek-ocr

Docling OCR plugin that uses DeepSeek-OCR as the text recognition backend while Docling continues to handle layout analysis (headings, tables, checkboxes, etc.).

## Development install

From the `docling-deepseek-ocr` directory, install in editable mode:

```bash
pip install -e .
```

## Configuration

The plugin is configured via environment variables (all optional, with sensible defaults):

- `DEEPSEEK_OCR_MODE`: `"api"` (default) or `"ollama"`.
- `DEEPSEEK_OCR_BASE_URL`: DeepSeek OCR API base URL (default `https://api.deepsee-ocr.ai`).
- `DEEPSEEK_OCR_API_KEY`: API key for the DeepSeek OCR API.
- `DEEPSEEK_OCR_OLLAMA_BASE_URL` / `OLLAMA_BASE_URL`: Ollama server base URL (default `http://localhost:11434`).
- `DEEPSEEK_OCR_OLLAMA_MODEL`: Ollama model name, e.g. `deepseek-ocr:3b`.

After installation, Docling can use the engine via `DeepseekOcrOptions` with `allow_external_plugins=True`.

