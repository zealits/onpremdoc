Token usage and pipeline economics for stakeholder reporting.
Files: usage_YYYY-MM-DD.jsonl (one JSON object per line, one line per pipeline step).
Phases: upload, pdf_processing, vectorization, retrieval, page_summary.
Fields per line: timestamp, step, phase, document_id, input_tokens, output_tokens, embedding_tokens, total_tokens, model.
API: GET /economics/summary?date=YYYY-MM-DD for aggregated totals by phase and step.
