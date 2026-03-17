"""
Document conversion using Docling with DeepSeek OCR instead of PP-OCRv5.

Same input (file/folder path) and output (markdown, page_mapping.json, confidence.json)
as detection.py. When OCR is needed (image input or PDF with no native text),
DeepSeek OCR is used instead of RapidOCR/PP-OCRv5.

OCR can be run in three ways (first matching wins):
1) Ollama (easiest): Set DEEPSEEK_OCR_OLLAMA_MODEL=deepseek-ocr:3b (and optionally
   DEEPSEEK_OCR_OLLAMA_BASE_URL or OLLAMA_BASE_URL). Run `ollama pull deepseek-ocr:3b` first.
2) Local Hugging Face model: Set DEEPSEEK_OCR_USE_LOCAL=1 and optionally
   DEEPSEEK_OCR_MODEL_PATH. Weights download from Hugging Face on first use.
   Requires: PyTorch with CUDA, ~6GB+ VRAM.
3) API: Set DEEPSEEK_OCR_API_KEY and optionally DEEPSEEK_OCR_BASE_URL.
"""
# Load .env so DEEPSEEK_OCR_* and OLLAMA_* are set when running this script standalone
try:
    from dotenv import load_dotenv
    from pathlib import Path as _EnvPath
    _env_path = _EnvPath(__file__).resolve().parent / ".env"
    load_dotenv(_env_path)
except Exception:
    pass

from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import LayoutOptions, PdfPipelineOptions
try:
    from hierarchical.postprocessor import ResultPostprocessor  # from docling-hierarchical-pdf (installed)
except ImportError:
    from hierarchical_stub.postprocessor import ResultPostprocessor  # local stub when pkg not installed
import logging
import sys
import re
import json
import os
import io
import tempfile
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress RapidOCR warnings when docling is used without OCR
logging.getLogger("RapidOCR").setLevel(logging.ERROR)

# Ollama: use local Ollama server with vision/OCR model (e.g. deepseek-ocr:3b)
# Set DEEPSEEK_OCR_OLLAMA_MODEL to enable; leave empty to skip Ollama.
OLLAMA_BASE_URL_DEFAULT = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
DEEPSEEK_OCR_OLLAMA_MODEL = os.getenv("DEEPSEEK_OCR_OLLAMA_MODEL", "").strip()
DEEPSEEK_OCR_OLLAMA_BASE_URL = os.getenv("DEEPSEEK_OCR_OLLAMA_BASE_URL", "").strip() or OLLAMA_BASE_URL_DEFAULT

# DeepSeek OCR API (https://www.deepseek-ocr.ai/docs) – used when not using Ollama or local HF
DEEPSEEK_OCR_BASE_URL = os.getenv("DEEPSEEK_OCR_BASE_URL", "https://api.deepsee-ocr.ai").rstrip("/")
DEEPSEEK_OCR_API_KEY = os.getenv("DEEPSEEK_OCR_API_KEY", "")

# Local Hugging Face model: Hugging Face model id or path to downloaded weights
DEEPSEEK_OCR_MODEL_PATH = os.getenv("DEEPSEEK_OCR_MODEL_PATH", "").strip() or "deepseek-ai/DeepSeek-OCR"
DEEPSEEK_OCR_USE_LOCAL = os.getenv("DEEPSEEK_OCR_USE_LOCAL", "").strip().lower() in ("1", "true", "yes")

# DPI for rendering PDF pages to images when using DeepSeek OCR fallback
PDF_PAGE_DPI = int(os.getenv("DEEPSEEK_OCR_PDF_DPI", "200"))

# Supported image extensions for direct DeepSeek OCR (no docling conversion)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# Lazy-loaded local model and tokenizer (set by _get_deepseek_local_model())
_deepseek_model = None
_deepseek_tokenizer = None


def _use_deepseek_ollama() -> bool:
    """True if we should use Ollama (e.g. deepseek-ocr:3b) for OCR."""
    return bool(DEEPSEEK_OCR_OLLAMA_MODEL)


def _use_deepseek_local() -> bool:
    """True if we should use the local Hugging Face DeepSeek-OCR model instead of API."""
    return DEEPSEEK_OCR_USE_LOCAL or bool(os.getenv("DEEPSEEK_OCR_MODEL_PATH", "").strip())


def _deepseek_ocr_ollama_extract(image_base64: str, prompt: str = "Free OCR.") -> str:
    """
    Send image (base64) to Ollama and return the extracted text.
    Uses DEEPSEEK_OCR_OLLAMA_MODEL and DEEPSEEK_OCR_OLLAMA_BASE_URL.
    Tries /api/generate first (images at top level), then /api/chat (images in message).
    """
    # Ollama expects raw base64 (no data URL prefix)
    raw_b64 = image_base64
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[-1] if "," in raw_b64 else raw_b64

    def _err(resp: requests.Response) -> None:
        if not resp.ok:
            try:
                err_body = resp.text
                if len(err_body) > 500:
                    err_body = err_body[:500] + "..."
                logger.warning("Ollama error %s: %s", resp.status_code, err_body)
            except Exception:
                pass

    # 1) Try /api/generate (images at top level) – works well for vision models
    url_gen = f"{DEEPSEEK_OCR_OLLAMA_BASE_URL}/api/generate"
    payload_gen = {
        "model": DEEPSEEK_OCR_OLLAMA_MODEL,
        "prompt": prompt,
        "images": [raw_b64],
        "stream": False,
    }
    resp = requests.post(url_gen, json=payload_gen, timeout=300)
    if resp.ok:
        data = resp.json()
        return (data.get("response") or "").strip()
    _err(resp)

    # 2) Fallback: /api/chat with images in message
    url_chat = f"{DEEPSEEK_OCR_OLLAMA_BASE_URL}/api/chat"
    payload_chat = {
        "model": DEEPSEEK_OCR_OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [raw_b64]}],
        "stream": False,
    }
    resp = requests.post(url_chat, json=payload_chat, timeout=300)
    if resp.ok:
        data = resp.json()
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()
    _err(resp)
    resp.raise_for_status()
    return ""


def _get_deepseek_local_model():
    """
    Load DeepSeek-OCR model and tokenizer from Hugging Face (or local path).
    Weights are downloaded on first use if DEEPSEEK_OCR_MODEL_PATH is the HF id.
    Returns (model, tokenizer). Requires CUDA for reasonable performance.
    """
    global _deepseek_model, _deepseek_tokenizer
    if _deepseek_model is not None and _deepseek_tokenizer is not None:
        return _deepseek_model, _deepseek_tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = DEEPSEEK_OCR_MODEL_PATH or "deepseek-ai/DeepSeek-OCR"
    logger.info("Loading DeepSeek-OCR model: %s (first run may download weights from Hugging Face)", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    attn = "flash_attention_2"
    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            _attn_implementation=attn,
            use_safetensors=True,
        )
    except Exception as e:
        logger.warning("Flash Attention 2 not available (%s), falling back to SDPA/eager", e)
        attn = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            _attn_implementation=attn,
            use_safetensors=True,
        )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda().to(torch.bfloat16)
    else:
        logger.warning("CUDA not available; running on CPU (may be slow)")
        model = model.to(torch.float32)
    _deepseek_model = model
    _deepseek_tokenizer = tokenizer
    return model, tokenizer


def _deepseek_ocr_local_infer(
    image_file_path: str,
    output_path: str,
    prompt: str = "<image>\nFree OCR. ",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
) -> str:
    """
    Run local DeepSeek-OCR inference on an image file.
    Uses model.infer(..., eval_mode=True) which returns the extracted text.
    """
    model, tokenizer = _get_deepseek_local_model()
    os.makedirs(output_path, exist_ok=True)
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file_path,
        output_path=output_path,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        test_compress=False,
        save_results=False,
        eval_mode=True,
    )
    return (res or "").strip()


def _deepseek_ocr_request(
    file_obj,
    filename: str,
    prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    POST file to DeepSeek OCR API; returns extracted text.
    Raises on HTTP error or missing API key.
    """
    if not DEEPSEEK_OCR_API_KEY:
        raise ValueError(
            "DEEPSEEK_OCR_API_KEY is not set. Set it in .env, or use Ollama (DEEPSEEK_OCR_OLLAMA_MODEL=deepseek-ocr:3b), "
            "or local HF model (DEEPSEEK_OCR_USE_LOCAL=1)."
        )
    url = f"{DEEPSEEK_OCR_BASE_URL}/v1/ocr"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_OCR_API_KEY}",
        "Accept": "application/json",
    }
    files = {"file": (filename, file_obj, "application/octet-stream")}
    data: Dict[str, str] = {}
    if prompt:
        data["prompt"] = prompt
    if language:
        data["language"] = language

    resp = requests.post(url, headers=headers, files=files, data=data or None, timeout=120)
    resp.raise_for_status()
    out = resp.json()
    return out.get("text", "").strip()


def deepseek_ocr_extract(
    file_path: Path,
    prompt: Optional[str] = None,
    language: Optional[str] = "en",
) -> str:
    """
    Extract text from a single file (image or PDF) using DeepSeek OCR.
    Backend order: Ollama (if DEEPSEEK_OCR_OLLAMA_MODEL set) -> local HF model -> API.
    """
    if _use_deepseek_ollama():
        import base64
        with open(file_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        fmt = "png" if file_path.suffix.lower() in (".png",) else "jpeg"
        image_b64 = f"data:image/{fmt};base64,{b64}"
        return _deepseek_ocr_ollama_extract(image_b64, prompt=prompt or "Free OCR.")
    if _use_deepseek_local():
        with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as tmpdir:
            return _deepseek_ocr_local_infer(
                str(file_path),
                output_path=tmpdir,
                prompt=prompt or "<image>\nFree OCR. ",
            )
    with open(file_path, "rb") as f:
        return _deepseek_ocr_request(f, file_path.name, prompt=prompt, language=language)


def deepseek_ocr_extract_from_bytes(
    image_bytes: bytes,
    filename: str = "page.png",
    prompt: Optional[str] = None,
    language: Optional[str] = "en",
) -> str:
    """Extract text from image bytes (e.g. a PDF page rendered to PNG)."""
    if _use_deepseek_ollama():
        import base64
        b64 = base64.b64encode(image_bytes).decode("ascii")
        fmt = "png" if (filename or "").lower().endswith(".png") else "jpeg"
        image_b64 = f"data:image/{fmt};base64,{b64}"
        return _deepseek_ocr_ollama_extract(image_b64, prompt=prompt or "Free OCR.")
    if _use_deepseek_local():
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as tmpdir:
                return _deepseek_ocr_local_infer(
                    tmp_path,
                    output_path=tmpdir,
                    prompt=prompt or "<image>\nFree OCR. ",
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return _deepseek_ocr_request(
        io.BytesIO(image_bytes),
        filename,
        prompt=prompt,
        language=language,
    )


def render_pdf_to_images(pdf_path: Path, dpi: int = PDF_PAGE_DPI) -> List[bytes]:
    """
    Render each PDF page to PNG bytes using PyMuPDF.
    Returns a list of PNG byte strings, one per page.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        result = []
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=dpi)
            try:
                png_bytes = pix.pil_tobytes(fmt="png")
            except Exception:
                # Fallback for older PyMuPDF: use PIL image and save to bytes
                pil_img = pix.pil_image()
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
            result.append(png_bytes)
        return result
    finally:
        doc.close()


def is_image_file(path: Path) -> bool:
    """True if path has an image extension we handle with DeepSeek OCR."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def get_pipeline_options_no_ocr() -> PdfPipelineOptions:
    """Pipeline options with OCR disabled (for PDF first pass)."""
    return PdfPipelineOptions(
        do_ocr=False,
        layout_options=LayoutOptions(create_orphan_clusters=True),
    )


# ---------------- MARKDOWN FIXING HELPERS ----------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def is_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"[\d.,%-]+", normalize(text)))

def is_header_like(row):
    """
    Header rows are mostly non-numeric.
    """
    non_numeric = sum(not is_numeric(c) for c in row if normalize(c))
    return non_numeric / max(len(row), 1) > 0.7

def is_group_header(row):
    """
    Example:
    | 5. | Course Contents (Semester V) |   |
    """
    if len(row) < 3:
        return False
    return (
        re.fullmatch(r"\d+\.", row[0].strip()) and
        normalize(row[-1]) == "" and
        normalize(row[1]) != ""
    )

def is_child_row(row):
    return normalize(row[0]) == "" and normalize(row[-1]) != ""

def has_only_one_filled_cell(row):
    """
    Check if a row has only one non-empty cell.
    Returns True if exactly one cell has content, False otherwise.
    For header-like rows, check after applying blank_repeated_adjacent_columns.
    """
    if is_header_like(row):
        transformed = blank_repeated_adjacent_columns(row)
        filled_count = sum(1 for c in transformed if normalize(c) != "")
        return filled_count == 1
    else:
        filled_count = sum(1 for c in row if normalize(c) != "")
        return filled_count == 1

def blank_repeated_adjacent_columns(row):
    """
    For header-like rows:
    keep first occurrence, blank consecutive duplicates.
    """
    new_row = []
    prev = None

    for cell in row:
        if prev is not None and normalize(cell) == normalize(prev):
            new_row.append("")
        else:
            new_row.append(cell)
            prev = cell

    return new_row

def is_separator_row(row):
    """
    Check if a row is a markdown table separator (all dashes or mostly dashes).
    """
    if not row:
        return False
    normalized_cells = [normalize(cell) for cell in row]
    dash_count = 0
    total_chars = 0
    for cell_text in normalized_cells:
        if cell_text:
            total_chars += len(cell_text)
            dash_count += cell_text.count("-")

    if total_chars == 0:
        return False

    return dash_count / total_chars > 0.8

def trim_trailing_empty_columns(rows):
    """
    Remove trailing empty columns from all rows.
    """
    if not rows:
        return rows

    max_cols = max(len(row) for row in rows) if rows else 0
    rightmost_filled = -1

    for col_idx in range(max_cols - 1, -1, -1):
        for row in rows:
            if col_idx < len(row) and normalize(row[col_idx]) != "":
                rightmost_filled = col_idx
                break
        if rightmost_filled >= 0:
            break

    if rightmost_filled >= 0:
        return [row[:rightmost_filled + 1] for row in rows]
    return rows

# ---------------- TABLE FIX ----------------
def fix_table(table_lines):
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    table_groups = []
    current_group = []

    for i, row in enumerate(rows):
        if is_separator_row(row):
            current_group.append(row)
            continue

        filled_cells = [normalize(c) for c in row if normalize(c) != ""]
        unique_filled = len(set(filled_cells))

        if i > 0 and unique_filled == 1:
            if current_group:
                table_groups.append(current_group)
            current_group = [row]
        else:
            current_group.append(row)

    if current_group:
        table_groups.append(current_group)

    all_tables = []
    for group in table_groups:
        fixed = []
        last_was_group = False
        has_separator = False

        for row in group:
            if is_separator_row(row):
                has_separator = True
                continue

            if is_group_header(row):
                fixed.append(row)
                last_was_group = True
                continue

            if last_was_group and is_child_row(row):
                row[1] = "↳ " + row[1]
                fixed.append(row)
                continue

            last_was_group = False

            if is_header_like(row):
                row = blank_repeated_adjacent_columns(row)

            fixed.append(row)

        fixed = trim_trailing_empty_columns(fixed)

        if not fixed:
            continue

        num_cols = max(len(row) for row in fixed) if fixed else 0

        if num_cols == 0:
            continue

        for row in fixed:
            while len(row) < num_cols:
                row.append("")

        out = []
        separator_added = False

        for idx, row in enumerate(fixed):
            line = "| " + " | ".join(row) + " |"
            out.append(line)

            if idx == 0 and not separator_added:
                separator_line = "| " + " | ".join(["---"] * num_cols) + " |"
                out.append(separator_line)
                separator_added = True

        all_tables.append(out)

    return all_tables

# ---------------- FILE PROCESSOR ----------------
def process_markdown(text, preserve_page_breaks: bool = True):
    """
    Process markdown and fix tables while preserving page break markers.
    """
    PAGE_BREAK_MARKER = "<!-- page break -->"
    lines = text.splitlines()
    output = []
    buffer = []
    in_table = False

    for line in lines:
        has_page_break = PAGE_BREAK_MARKER in line if preserve_page_breaks else False

        if line.strip().startswith("|") and "|" in line:
            buffer.append(line)
            in_table = True
        else:
            if in_table:
                tables = fix_table(buffer)
                for i, table in enumerate(tables):
                    output.extend(table)
                    if i < len(tables) - 1:
                        output.append("")
                buffer = []
                in_table = False

            if has_page_break:
                output.append(PAGE_BREAK_MARKER)
            else:
                output.append(line)

    if buffer:
        tables = fix_table(buffer)
        for i, table in enumerate(tables):
            output.extend(table)
            if i < len(tables) - 1:
                output.append("")

    return "\n".join(output)


def is_image_only_markdown(markdown: str) -> bool:
    """
    Heuristic check to see if markdown contains only image placeholders and page breaks.
    """
    PAGE_BREAK_MARKER = "<!-- page break -->"
    IMAGE_MARKER = "<!-- image -->"

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in (PAGE_BREAK_MARKER, IMAGE_MARKER):
            continue
        return False

    return True

# ---------------- PAGE MAPPING EXTRACTION ----------------
def extract_page_mapping_from_markdown(markdown_with_breaks: str) -> Dict[str, Any]:
    """
    Extract page boundaries from markdown that contains page break placeholders.
    """
    logger.info("Extracting page boundaries from markdown with page break markers...")

    PAGE_BREAK_MARKER = "<!-- page break -->"

    lines = markdown_with_breaks.splitlines()
    line_to_page = []
    page_boundaries = []
    current_page = 1
    page_start_line = 0

    for line_idx, line in enumerate(lines):
        if PAGE_BREAK_MARKER in line:
            line_to_page.append(current_page)

            if page_start_line < line_idx:
                page_boundaries.append((page_start_line, line_idx, current_page))

            current_page += 1
            page_start_line = line_idx + 1
        else:
            line_to_page.append(current_page)

    if page_start_line < len(lines):
        page_boundaries.append((page_start_line, len(lines) - 1, current_page))

    total_pages = current_page

    logger.info(f"Extracted page mapping: {total_pages} pages, {len(page_boundaries)} boundaries")

    return {
        "line_to_page": line_to_page,
        "page_boundaries": page_boundaries,
        "total_pages": total_pages
    }

def extract_page_mapping_page_by_page(doc) -> Dict[str, Any]:
    """
    Extract page mapping by processing document page by page.
    """
    logger.info("Extracting page mapping using page-by-page approach...")

    try:
        pages_content = []
        total_pages = 0

        if hasattr(doc.document, 'pages') and doc.document.pages:
            logger.info("Found pages attribute in document")
            for page in doc.document.pages:
                total_pages += 1
                page_md = page.export_to_markdown() if hasattr(page, 'export_to_markdown') else ""
                pages_content.append(page_md)

        elif hasattr(doc.document, 'texts'):
            logger.info("Grouping text elements by page...")
            pages_dict = {}

            for text_item in doc.document.texts:
                if hasattr(text_item, 'prov') and text_item.prov:
                    for prov in text_item.prov:
                        if hasattr(prov, 'page_no'):
                            page_no = prov.page_no
                            if page_no not in pages_dict:
                                pages_dict[page_no] = []
                            pages_dict[page_no].append(text_item.text)
                            break

            for page_no in sorted(pages_dict.keys()):
                page_texts = pages_dict[page_no]
                page_content = "\n".join(page_texts)
                pages_content.append(page_content)
                total_pages = max(total_pages, page_no)

        if pages_content:
            logger.info(f"Extracted {total_pages} pages using page-by-page method")
            return {
                "page_contents": pages_content,
                "total_pages": total_pages,
                "method": "page_by_page"
            }
        else:
            logger.warning("Could not extract pages using page-by-page method")
            return None

    except Exception as e:
        logger.warning(f"Error in page-by-page extraction: {e}")
        return None

def create_approximate_page_mapping(markdown_lines: List[str]) -> Dict[str, Any]:
    """
    Create approximate page mapping when page information is not available.
    """
    logger.info("Creating approximate page mapping (50 lines per page)")
    lines_per_page = 50

    line_to_page = []
    page_boundaries = []
    total_pages = (len(markdown_lines) // lines_per_page) + 1

    for line_idx in range(len(markdown_lines)):
        page_num = (line_idx // lines_per_page) + 1
        line_to_page.append(page_num)

        if line_idx == 0 or line_to_page[line_idx - 1] != page_num:
            if line_idx > 0:
                page_boundaries.append((line_idx - lines_per_page, line_idx - 1, page_num - 1))
            page_boundaries.append((line_idx, min(line_idx + lines_per_page - 1, len(markdown_lines) - 1), page_num))

    return {
        "line_to_page": line_to_page,
        "page_boundaries": page_boundaries,
        "total_pages": total_pages
    }


def _minimal_confidence(total_pages: int, ocr_source: str = "deepseek") -> Dict[str, Any]:
    """Build a minimal confidence dict when OCR was done via DeepSeek (no docling scores)."""
    return {
        "layout_score": None,
        "ocr_score": None,
        "parse_score": None,
        "table_score": None,
        "mean_grade": None,
        "low_grade": None,
        "pages": {str(p): {"layout_score": None, "ocr_score": None, "parse_score": None, "table_score": None} for p in range(1, total_pages + 1)},
        "ocr_source": ocr_source,
    }


def process_single_pdf(pdf_path: Path, base_output_dir: Path, target_dir: Optional[Path] = None):
    """
    Process a single document (PDF or image):
    - Convert to markdown (using Docling for PDF with native text; DeepSeek OCR for images or PDF fallback)
    - Fix tables
    - Add page break markers
    - Generate page mapping JSON and confidence JSON

    Same signature and output files as detection.process_single_pdf.
    """
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return

    logger.info(f"Processing file: {pdf_path}")
    logger.info(f"File type: {pdf_path.suffix.upper() if pdf_path.suffix else 'Unknown'}")

    if target_dir is not None:
        doc_output_dir = Path(target_dir)
    else:
        input_stem = pdf_path.stem
        doc_output_dir = base_output_dir / input_stem

    doc_output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = pdf_path.stem
    md_path = doc_output_dir / f"{input_stem}.md"
    page_mapping_path = doc_output_dir / f"{input_stem}_page_mapping.json"
    confidence_path = doc_output_dir / f"{input_stem}_confidence.json"

    # ---------- Image input: use DeepSeek OCR only (no docling) ----------
    if is_image_file(pdf_path):
        logger.info("Image input: using DeepSeek OCR (no docling)")
        try:
            text = deepseek_ocr_extract(pdf_path, language="en")
        except Exception as e:
            logger.error(f"DeepSeek OCR failed: {e}")
            return
        markdown = text or ""
        fixed_markdown = process_markdown(markdown, preserve_page_breaks=True)
        page_mapping = {
            "line_to_page": [1] * max(1, len(fixed_markdown.splitlines())),
            "page_boundaries": [(0, max(0, len(fixed_markdown.splitlines()) - 1), 1)],
            "total_pages": 1,
        }
        confidence_data = _minimal_confidence(1)
        md_path.write_text(fixed_markdown, encoding="utf-8")
        with open(page_mapping_path, "w", encoding="utf-8") as f:
            json.dump(page_mapping, f, indent=2, ensure_ascii=False)
        with open(confidence_path, "w", encoding="utf-8") as f:
            json.dump(confidence_data, f, indent=2, ensure_ascii=False)
        logger.info("DONE (image, DeepSeek OCR)")
        logger.info(f"Output directory: {doc_output_dir}")
        logger.info(f"Fixed markdown saved to: {md_path}")
        logger.info(f"Page mapping saved to: {page_mapping_path}")
        logger.info(f"Total pages: 1")
        return

    # ---------- PDF: Docling first pass (no OCR) ----------
    logger.info("Initializing Docling converter (PDF, native text first pass)")
    pipeline_options_pdf = get_pipeline_options_no_ocr()
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options_pdf),
        InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options_pdf),
    }
    converter = DocumentConverter(format_options=format_options)

    logger.info(f"Converting document: {pdf_path.name}")
    doc = converter.convert(str(pdf_path))

    # ---------------- CONFIDENCE (from docling when available) ----------------
    confidence = getattr(doc, "confidence", None)
    if confidence is not None:
        layout_score = getattr(confidence, "layout_score", None)
        ocr_score = getattr(confidence, "ocr_score", None)
        parse_score = getattr(confidence, "parse_score", None)
        table_score = getattr(confidence, "table_score", None)
        mean_grade = getattr(confidence, "mean_grade", None)
        low_grade = getattr(confidence, "low_grade", None)

        logger.info("Docling conversion confidence (document-level):")
        logger.info(
            "  Scores - layout: %s, ocr: %s, parse: %s, table: %s",
            layout_score, ocr_score, parse_score, table_score,
        )
        logger.info("  Grades - mean: %s, low: %s", mean_grade, low_grade)

        pages_conf = getattr(confidence, "pages", None)
        if isinstance(pages_conf, dict) and pages_conf:
            logger.info("Docling conversion confidence (per-page scores):")
            for page_no, page_scores in sorted(pages_conf.items()):
                p_layout = getattr(page_scores, "layout_score", None)
                p_ocr = getattr(page_scores, "ocr_score", None)
                p_parse = getattr(page_scores, "parse_score", None)
                p_table = getattr(page_scores, "table_score", None)
                logger.info(
                    "  Page %s - layout: %s, ocr: %s, parse: %s, table: %s",
                    page_no, p_layout, p_ocr, p_parse, p_table,
                )

        try:
            if hasattr(confidence, "model_dump_json"):
                confidence_data = json.loads(confidence.model_dump_json())
            elif hasattr(confidence, "dict"):
                confidence_data = confidence.dict()
            else:
                confidence_data = json.loads(json.dumps(confidence, default=lambda o: o.__dict__))
            with open(confidence_path, "w", encoding="utf-8") as f:
                json.dump(confidence_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Confidence report saved to: {confidence_path}")
        except Exception as e:
            logger.warning(f"Failed to save confidence report: {e}")

    # ---------------- HIERARCHICAL POSTPROCESSOR ----------------
    logger.info("Applying hierarchical postprocessor to fix heading levels...")
    try:
        ResultPostprocessor(doc, source=str(pdf_path)).process()
        logger.info("✓ Successfully applied hierarchical postprocessor")
    except Exception as e:
        logger.warning(f"Hierarchical postprocessor failed: {e}")
        try:
            ResultPostprocessor(doc).process()
            logger.info("✓ Applied hierarchical postprocessor (without explicit source)")
        except Exception as e2:
            logger.warning(f"Hierarchical postprocessor failed even without source: {e2}")

    # ---------------- EXPORT MARKDOWN WITH PAGE BREAKS ----------------
    def export_markdown_with_page_breaks(current_doc):
        logger.info("Exporting structured Markdown with page break markers")
        md = None
        try:
            md = current_doc.document.export_to_markdown(page_break_placeholder="<!-- page break -->")
            logger.info("✓ Exported markdown with page break markers (page_break_placeholder)")
        except TypeError as e:
            logger.warning(f"page_break_placeholder not supported: {e}")
            page_by_page_data = extract_page_mapping_page_by_page(current_doc)
            if page_by_page_data and page_by_page_data.get("page_contents"):
                page_contents = page_by_page_data["page_contents"]
                md = "\n<!-- page break -->\n".join(page_contents)
                logger.info(f"✓ Reconstructed markdown with {page_by_page_data['total_pages']} pages")
            else:
                md = current_doc.document.export_to_markdown()
        if md is None:
            md = current_doc.document.export_to_markdown()
        return md

    markdown = export_markdown_with_page_breaks(doc)

    # ---------- PDF image-only fallback: DeepSeek OCR per page ----------
    if markdown and is_image_only_markdown(markdown):
        logger.warning(
            "Markdown contains only image placeholders. Re-running with DeepSeek OCR (per-page)..."
        )
        try:
            page_images = render_pdf_to_images(pdf_path, dpi=PDF_PAGE_DPI)
        except Exception as e:
            logger.error(f"Failed to render PDF to images: {e}")
            page_images = []
        if not page_images:
            logger.error("No pages rendered from PDF")
        else:
            page_texts = []
            for i, png_bytes in enumerate(page_images):
                try:
                    text = deepseek_ocr_extract_from_bytes(
                        png_bytes,
                        filename=f"page_{i + 1}.png",
                        language="en",
                    )
                    page_texts.append(text or "")
                except Exception as e:
                    logger.warning(f"DeepSeek OCR failed for page {i + 1}: {e}")
                    page_texts.append("")
            markdown = "\n<!-- page break -->\n".join(page_texts)
            total_pages = len(page_texts)
            confidence_data = _minimal_confidence(total_pages)
            with open(confidence_path, "w", encoding="utf-8") as f:
                json.dump(confidence_data, f, indent=2, ensure_ascii=False)

    # ---------------- FIX MARKDOWN TABLES ----------------
    logger.info("Fixing markdown table formatting (preserving page breaks)")
    if markdown:
        fixed_markdown = process_markdown(markdown, preserve_page_breaks=True)
    else:
        fixed_markdown = ""

    # ---------------- EXTRACT PAGE MAPPING ----------------
    page_mapping = extract_page_mapping_from_markdown(fixed_markdown)
    if not page_mapping or page_mapping.get("total_pages", 0) == 0:
        logger.warning("No page break markers found, using approximate mapping")
        markdown_lines = fixed_markdown.splitlines()
        page_mapping = create_approximate_page_mapping(markdown_lines)

    # ---------------- SAVE ----------------
    md_path.write_text(fixed_markdown, encoding="utf-8")
    with open(page_mapping_path, "w", encoding="utf-8") as f:
        json.dump(page_mapping, f, indent=2, ensure_ascii=False)

    logger.info("DONE")
    logger.info(f"Output directory: {doc_output_dir}")
    logger.info(f"Fixed markdown saved to: {md_path}")
    logger.info(f"Page mapping saved to: {page_mapping_path}")
    logger.info(f"Total pages detected: {page_mapping['total_pages']}")


def main():
    """
    Entry point. Same as detection.py:
    - Single file: process that document into output/{stem}/
    - Folder: process all *.pdf (and supported images) into output/{folder_stem}/{stem}/
    """
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path("invoice4.pdf")

    if not input_path.exists():
        print(f"Error: Path '{input_path}' not found!")
        sys.exit(1)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    if input_path.is_file():
        base_output_dir = out_dir
        process_single_pdf(input_path, base_output_dir)
    elif input_path.is_dir():
        pdf_files = sorted(list(input_path.glob("*.pdf")))
        if not pdf_files:
            logger.error(f"No PDF files found in folder: {input_path}")
            sys.exit(1)

        group_output_dir = out_dir / input_path.stem
        group_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(pdf_files)} PDFs from folder: {input_path}")
        for pdf in pdf_files:
            process_single_pdf(pdf, group_output_dir)
    else:
        logger.error(f"Unsupported path type: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
