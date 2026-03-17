from __future__ import annotations

from docling_deepseek_ocr.model import DeepseekOcrModel


def ocr_engines() -> dict[str, list[type[DeepseekOcrModel]]]:
    """Register the DeepSeek-OCR engine with Docling's plugin system."""
    return {"ocr_engines": [DeepseekOcrModel]}

