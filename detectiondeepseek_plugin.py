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
from pathlib import Path
from typing import Dict, List, Any, Optional

from docling_deepseek_ocr import DeepseekOcrOptions


# Load .env so DEEPSEEK_OCR_* and OLLAMA_* are available when running this script directly.
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except Exception:
    # If python-dotenv is not installed or .env is missing, continue with process env only.
    pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_deepseek_pipeline_options() -> PdfPipelineOptions:
    """PdfPipelineOptions using DeepSeek-OCR via the Docling plugin."""
    return PdfPipelineOptions(
        do_ocr=True,
        ocr_options=DeepseekOcrOptions(),
        layout_options=LayoutOptions(create_orphan_clusters=True),
        allow_external_plugins=True,
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"[\d.,%-]+", normalize(text)))


def is_header_like(row):
    non_numeric = sum(not is_numeric(c) for c in row if normalize(c))
    return non_numeric / max(len(row), 1) > 0.7


def is_group_header(row):
    if len(row) < 3:
        return False
    return (
        re.fullmatch(r"\d+\.", row[0].strip())
        and normalize(row[-1]) == ""
        and normalize(row[1]) != ""
    )


def is_child_row(row):
    return normalize(row[0]) == "" and normalize(row[-1]) != ""


def blank_repeated_adjacent_columns(row):
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
        return [row[: rightmost_filled + 1] for row in rows]
    return rows


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

        for row in group:
            if is_separator_row(row):
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


def process_markdown(text: str, preserve_page_breaks: bool = True) -> str:
    PAGE_BREAK_MARKER = "<!-- page break -->"
    lines = text.splitlines()
    output: List[str] = []
    buffer: List[str] = []
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


def extract_page_mapping_from_markdown(markdown_with_breaks: str) -> Dict[str, Any]:
    logger.info("Extracting page boundaries from markdown with page break markers...")
    PAGE_BREAK_MARKER = "<!-- page break -->"

    lines = markdown_with_breaks.splitlines()
    line_to_page: List[int] = []
    page_boundaries: List[tuple[int, int, int]] = []
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
    logger.info("Extracted page mapping: %s pages, %s boundaries", total_pages, len(page_boundaries))
    return {
        "line_to_page": line_to_page,
        "page_boundaries": page_boundaries,
        "total_pages": total_pages,
    }


def create_approximate_page_mapping(markdown_lines: List[str]) -> Dict[str, Any]:
    logger.info("Creating approximate page mapping (50 lines per page)")
    lines_per_page = 50
    line_to_page: List[int] = []
    page_boundaries: List[tuple[int, int, int]] = []
    total_pages = (len(markdown_lines) // lines_per_page) + 1

    for line_idx in range(len(markdown_lines)):
        page_num = (line_idx // lines_per_page) + 1
        line_to_page.append(page_num)
        if line_idx == 0 or line_to_page[line_idx - 1] != page_num:
            if line_idx > 0:
                page_boundaries.append((line_idx - lines_per_page, line_idx - 1, page_num - 1))
            page_boundaries.append(
                (line_idx, min(line_idx + lines_per_page - 1, len(markdown_lines) - 1), page_num)
            )

    return {
        "line_to_page": line_to_page,
        "page_boundaries": page_boundaries,
        "total_pages": total_pages,
    }


def process_single_pdf(pdf_path: Path, base_output_dir: Path, target_dir: Optional[Path] = None) -> None:
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        return

    logger.info("Processing file: %s", pdf_path)
    logger.info("File type: %s", pdf_path.suffix.upper() if pdf_path.suffix else "Unknown")

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

    logger.info("Initializing Docling converter with DeepSeek-OCR plugin")

    pipeline_options = get_deepseek_pipeline_options()
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }
    converter = DocumentConverter(format_options=format_options)

    logger.info("Converting document: %s", pdf_path.name)
    doc = converter.convert(str(pdf_path))

    confidence = getattr(doc, "confidence", None)
    if confidence is not None:
        try:
            if hasattr(confidence, "model_dump_json"):
                confidence_data = json.loads(confidence.model_dump_json())
            elif hasattr(confidence, "dict"):
                confidence_data = confidence.dict()
            else:
                confidence_data = json.loads(json.dumps(confidence, default=lambda o: o.__dict__))
            with open(confidence_path, "w", encoding="utf-8") as f:
                json.dump(confidence_data, f, indent=2, ensure_ascii=False)
            logger.info("Confidence report saved to: %s", confidence_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save confidence report: %s", exc)

    logger.info("Applying hierarchical postprocessor to fix heading levels...")
    try:
        ResultPostprocessor(doc, source=str(pdf_path)).process()
        logger.info("Successfully applied hierarchical postprocessor")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hierarchical postprocessor failed: %s", exc)
        try:
            ResultPostprocessor(doc).process()
            logger.info("Applied hierarchical postprocessor (without explicit source)")
        except Exception as exc2:  # noqa: BLE001
            logger.warning("Hierarchical postprocessor failed even without source: %s", exc2)

    logger.info("Exporting structured Markdown with page break markers")
    try:
        markdown = doc.document.export_to_markdown(page_break_placeholder="<!-- page break -->")
        logger.info("Successfully exported markdown with page break markers")
    except TypeError as exc:  # older docling without page_break_placeholder
        logger.warning("page_break_placeholder not supported: %s", exc)
        markdown = doc.document.export_to_markdown()

    logger.info("Fixing markdown table formatting (preserving page breaks)")
    fixed_markdown = process_markdown(markdown or "", preserve_page_breaks=True)

    page_mapping = extract_page_mapping_from_markdown(fixed_markdown)
    if not page_mapping or page_mapping.get("total_pages", 0) == 0:
        logger.warning("No page break markers found, using approximate mapping")
        markdown_lines = fixed_markdown.splitlines()
        page_mapping = create_approximate_page_mapping(markdown_lines)

    md_path.write_text(fixed_markdown, encoding="utf-8")
    with open(page_mapping_path, "w", encoding="utf-8") as f:
        json.dump(page_mapping, f, indent=2, ensure_ascii=False)

    logger.info("DONE")
    logger.info("Output directory: %s", doc_output_dir)
    logger.info("Fixed markdown saved to: %s", md_path)
    logger.info("Page mapping saved to: %s", page_mapping_path)
    logger.info("Total pages detected: %s", page_mapping["total_pages"])


def main() -> None:
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
            logger.error("No PDF files found in folder: %s", input_path)
            sys.exit(1)

        group_output_dir = out_dir / input_path.stem
        group_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing %s PDFs from folder: %s", len(pdf_files), input_path)
        for pdf in pdf_files:
            process_single_pdf(pdf, group_output_dir)
    else:
        logger.error("Unsupported path type: %s", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()

