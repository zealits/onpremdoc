from __future__ import annotations

import os
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions
from pydantic import ConfigDict, Field


class DeepseekOcrOptions(OcrOptions):
    """Options for the DeepSeek-OCR engine used through the Docling plugin.

    This configuration mirrors the GLM-OCR plugin style, but targets DeepSeek:

    - Two backends are supported:
      - ``mode=\"api\"`` (default): call the DeepSeek OCR HTTP API.
      - ``mode=\"ollama\"``: call a local Ollama server running a DeepSeek OCR model.

    All fields fall back to environment variables so you can configure the engine
    purely via ``.env`` / container env without code changes.
    """

    # Unique identifier for this OCR engine kind.
    kind: ClassVar[Literal["deepseek-ocr"]] = "deepseek-ocr"

    # Backend selection.
    mode: Literal["api", "ollama"] = Field(
        default_factory=lambda: os.environ.get("DEEPSEEK_OCR_MODE", "api").lower(),  # type: ignore[arg-type]
    )

    # ---- API backend (cloud / HTTP service) ----
    # Two flavours supported:
    # - DeepSeek OCR HTTP API (DEEPSEEK_OCR_BASE_URL / DEEPSEEK_OCR_API_KEY)
    # - Hugging Face Inference API for DeepSeek-OCR (HF_OCR_ENDPOINT / HF_TOKEN)
    api_base_url: str = Field(
        default_factory=lambda: os.environ.get("DEEPSEEK_OCR_BASE_URL", "https://api.deepsee-ocr.ai").rstrip("/")
    )
    api_key: str | None = Field(
        default_factory=lambda: os.environ.get("DEEPSEEK_OCR_API_KEY") or None,
    )
    hf_api_url: str = Field(
        default_factory=lambda: os.environ.get("HF_OCR_ENDPOINT", "").strip(),
    )
    hf_token: str | None = Field(
        default_factory=lambda: os.environ.get("HF_TOKEN") or None,
    )

    # ---- Ollama backend (local HTTP server with DeepSeek vision model) ----
    ollama_base_url: str = Field(
        default_factory=lambda: (
            os.environ.get("DEEPSEEK_OCR_OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
    )
    ollama_model: str = Field(
        default_factory=lambda: os.environ.get("DEEPSEEK_OCR_OLLAMA_MODEL", "").strip(),
    )

    # ---- Generic OCR request behaviour ----
    lang: list[str] = Field(
        default_factory=lambda: os.environ.get("DEEPSEEK_OCR_LANG", "en").split(","),
    )
    prompt: str = Field(
        default_factory=lambda: os.environ.get("DEEPSEEK_OCR_PROMPT", "Free OCR."),
    )
    timeout: float = Field(
        default_factory=lambda: float(os.environ.get("DEEPSEEK_OCR_TIMEOUT", "120")),
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("DEEPSEEK_OCR_MAX_TOKENS", "4096")),
    )

    # ---- Image rendering / batching ----
    scale: float = Field(
        default_factory=lambda: float(os.environ.get("DEEPSEEK_OCR_SCALE", "3.0")),
    )
    max_image_pixels: int = Field(
        default_factory=lambda: int(os.environ.get("DEEPSEEK_OCR_MAX_IMAGE_PIXELS", "4500000")),
    )
    max_concurrent_requests: int = Field(
        default_factory=lambda: int(os.environ.get("DEEPSEEK_OCR_MAX_CONCURRENT_REQUESTS", "8")),
    )

    # ---- Retry behaviour ----
    max_retries: int = Field(
        default_factory=lambda: int(os.environ.get("DEEPSEEK_OCR_MAX_RETRIES", "3")),
    )
    retry_backoff_factor: float = Field(
        default_factory=lambda: float(os.environ.get("DEEPSEEK_OCR_RETRY_BACKOFF_FACTOR", "2.0")),
    )

    model_config = ConfigDict(extra="forbid")

