from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import TYPE_CHECKING, Final

import httpx
from docling.datamodel.base_models import DoclingComponentType, ErrorItem
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling_deepseek_ocr.options import DeepseekOcrOptions

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import Page
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import OcrOptions


logger = logging.getLogger(__name__)

_HTTP_CLIENT_ERROR_MIN: Final = 400
_HTTP_SERVER_ERROR_MIN: Final = 500


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_base64_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


class DeepseekOcrModel(BaseOcrModel):
    """Docling OCR engine that delegates recognition to DeepSeek-OCR.

    The model follows the same pattern as ``GlmOcrRemoteModel``:
    Docling decides which regions to OCR, this class renders crops and sends
    them to a DeepSeek backend (API or Ollama), and returns ``TextCell`` items.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Path | None,
        options: DeepseekOcrOptions,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: DeepseekOcrOptions
        self._local = threading.local()

        if self.enabled:
            logger.info(
                "DeepseekOcrModel initialised: mode=%s api_base_url=%s ollama_base_url=%s",
                self.options.mode,
                self.options.api_base_url,
                self.options.ollama_base_url,
            )

    def _get_client(self) -> httpx.Client:
        """Return a thread-local HTTP client, created lazily on first use."""
        if not self.enabled:
            msg = "DeepseekOcrModel is not enabled"
            raise RuntimeError(msg)
        if not hasattr(self._local, "client"):
            limits = httpx.Limits(
                max_connections=self.options.max_concurrent_requests,
                max_keepalive_connections=self.options.max_concurrent_requests,
            )
            headers: dict[str, str] = {}
            if self.options.mode == "api" and self.options.api_key:
                headers["Authorization"] = f"Bearer {self.options.api_key}"
            self._local.client = httpx.Client(
                timeout=self.options.timeout,
                limits=limits,
                headers=headers,
            )
        return self._local.client

    # --------- DeepSeek backends ---------

    def _recognise_crop_via_api(self, image: Image.Image) -> str:
        """Send a PNG image to a remote OCR HTTP API.

        Priority:
        1. Hugging Face Inference API (HF_OCR_ENDPOINT + HF_TOKEN) if configured.
        2. DeepSeek OCR HTTP API (DEEPSEEK_OCR_BASE_URL + DEEPSEEK_OCR_API_KEY).
        """
        png_bytes = _pil_to_png_bytes(image)

        # ---- Hugging Face Inference API path ----
        if self.options.hf_api_url:
            if not self.options.hf_token:
                msg = (
                    "HF_OCR_ENDPOINT is set but HF_TOKEN is missing. "
                    "Set HF_TOKEN or clear HF_OCR_ENDPOINT to use the DeepSeek API instead."
                )
                raise RuntimeError(msg)

            url = self.options.hf_api_url
            headers = {
                "Authorization": f"Bearer {self.options.hf_token}",
                "Accept": "application/json",
            }
            # Most HF text-generation / OCR models accept raw bytes; if your endpoint
            # expects a different schema, adjust here.
            resp = self._get_client().post(url, headers=headers, content=png_bytes)
            if resp.status_code >= _HTTP_CLIENT_ERROR_MIN:
                logger.error(
                    "HF DeepSeek-OCR API returned HTTP %d for %s: %s",
                    resp.status_code,
                    url,
                    resp.text[:500],
                )
            resp.raise_for_status()
            out = resp.json()
            # Try common keys used by HF Inference; adjust if your endpoint differs.
            text = (
                out.get("text")
                or out.get("generated_text")
                or (out[0].get("generated_text") if isinstance(out, list) and out and isinstance(out[0], dict) else "")
            )
            return (text or "").strip()

        # ---- DeepSeek OCR HTTP API path ----
        if not self.options.api_key:
            msg = (
                "DEEPSEEK_OCR_API_KEY is not set. Configure it via environment variables, "
                "set HF_OCR_ENDPOINT/HF_TOKEN for Hugging Face Inference, "
                "or switch DeepseekOcrOptions.mode to 'ollama'."
            )
            raise RuntimeError(msg)

        url = f"{self.options.api_base_url}/v1/ocr"
        files = {"file": ("crop.png", png_bytes, "image/png")}
        data: dict[str, str] = {}
        if self.options.prompt:
            data["prompt"] = self.options.prompt
        if self.options.lang:
            data["language"] = ",".join(self.options.lang)

        resp = self._get_client().post(url, files=files, data=data or None)
        if resp.status_code >= _HTTP_CLIENT_ERROR_MIN:
            logger.error(
                "DeepSeek OCR API returned HTTP %d for %s: %s",
                resp.status_code,
                url,
                resp.text[:500],
            )
        resp.raise_for_status()
        out = resp.json()
        return (out.get("text") or "").strip()

    def _recognise_crop_via_ollama(self, image: Image.Image) -> str:
        """Send an image to a local Ollama server running a DeepSeek OCR model."""
        if not self.options.ollama_model:
            msg = (
                "DEEPSEEK_OCR_OLLAMA_MODEL is not set. Configure it via environment variables "
                "or switch DeepseekOcrOptions.mode to 'api'."
            )
            raise RuntimeError(msg)

        data_uri = _pil_to_base64_uri(image, fmt="PNG")
        raw_b64 = data_uri.split(",", 1)[-1]

        # 1) Try /api/generate
        url_gen = f"{self.options.ollama_base_url}/api/generate"
        payload_gen = {
            "model": self.options.ollama_model,
            "prompt": self.options.prompt or "Free OCR.",
            "images": [raw_b64],
            "stream": False,
        }
        resp = self._get_client().post(url_gen, json=payload_gen)
        if resp.is_success:
            data = resp.json()
            return (data.get("response") or "").strip()

        # 2) Fallback: /api/chat
        url_chat = f"{self.options.ollama_base_url}/api/chat"
        payload_chat = {
            "model": self.options.ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": self.options.prompt or "Free OCR.",
                    "images": [raw_b64],
                }
            ],
            "stream": False,
        }
        resp = self._get_client().post(url_chat, json=payload_chat)
        if resp.status_code >= _HTTP_CLIENT_ERROR_MIN:
            logger.error(
                "Ollama DeepSeek OCR returned HTTP %d for %s: %s",
                resp.status_code,
                url_chat,
                resp.text[:500],
            )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()

    def _recognise_crop(self, image: Image.Image) -> str:
        """Dispatch OCR to the configured DeepSeek backend."""
        if self.options.mode == "ollama":
            return self._recognise_crop_via_ollama(image)
        return self._recognise_crop_via_api(image)

    def _recognise_crop_with_retry(self, image: Image.Image) -> str:
        """Run OCR with basic retry/backoff for transient errors."""
        max_retries = self.options.max_retries
        backoff = self.options.retry_backoff_factor

        for attempt in range(max_retries + 1):
            try:
                return self._recognise_crop(image)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < _HTTP_SERVER_ERROR_MIN:
                    # 4xx — deterministic client error; do not retry
                    raise
                if attempt == max_retries:
                    logger.exception("DeepSeek OCR call failed after %d retries", max_retries)
                    raise
                logger.debug(
                    "HTTP %d from DeepSeek OCR on attempt %d",
                    exc.response.status_code,
                    attempt + 1,
                )
            except httpx.HTTPError as exc:
                if attempt == max_retries:
                    logger.exception("DeepSeek OCR call failed after %d retries", max_retries)
                    raise
                logger.debug("Network error on attempt %d: %s", attempt + 1, exc)

            sleep_time = backoff**attempt
            logger.warning(
                "DeepSeek OCR call failed (attempt %d/%d). Retrying in %.1f seconds...",
                attempt + 1,
                max_retries + 1,
                sleep_time,
            )
            time.sleep(sleep_time)
        return ""

    # --------- Crop extraction & processing ---------

    def _collect_crops(
        self,
        page: Page,
        ocr_rects: list[BoundingBox],
    ) -> list[tuple[int, BoundingBox, Image.Image | None]]:
        """Extract crop images for all OCR regions sequentially."""
        backend = page._backend  # noqa: SLF001
        if backend is None:
            return []

        crop_data: list[tuple[int, BoundingBox, Image.Image | None]] = []
        for cell_idx, ocr_rect in enumerate(ocr_rects):
            if ocr_rect.area() == 0:
                crop_data.append((cell_idx, ocr_rect, None))
                continue

            crop_w = ocr_rect.r - ocr_rect.l
            crop_h = ocr_rect.b - ocr_rect.t
            native_pixels = crop_w * crop_h
            if native_pixels > 0:
                max_safe_scale = (self.options.max_image_pixels / native_pixels) ** 0.5
                actual_scale = min(self.options.scale, max_safe_scale)
            else:
                actual_scale = self.options.scale

            if actual_scale < self.options.scale:
                logger.debug(
                    "Crop (%dx%d page-units) would exceed max_image_pixels=%d at scale=%.1f; "
                    "reducing to scale=%.2f",
                    int(crop_w),
                    int(crop_h),
                    self.options.max_image_pixels,
                    self.options.scale,
                    actual_scale,
                )

            high_res_image = backend.get_page_image(
                scale=actual_scale,
                cropbox=ocr_rect,
            )
            crop_data.append((cell_idx, ocr_rect, high_res_image))
        return crop_data

    def _process_crop(
        self,
        cell_idx: int,
        ocr_rect: BoundingBox,
        image: Image.Image | None,
    ) -> tuple[TextCell | None, str | None]:
        """OCR a single crop and convert it to a TextCell."""
        if image is None:
            return None, None

        try:
            text = self._recognise_crop_with_retry(image)
        except httpx.HTTPStatusError as exc:
            msg = (
                f"DeepSeek OCR crop index={cell_idx} rejected: "
                f"HTTP {exc.response.status_code} — {exc.response.text[:200]}"
            )
            logger.exception(msg)
            return None, msg
        except httpx.HTTPError as exc:
            msg = f"DeepSeek OCR crop index={cell_idx} failed after retries: {exc}"
            logger.exception(msg)
            return None, msg

        if not text.strip():
            return None, None

        rect = BoundingRectangle.from_bounding_box(
            BoundingBox.from_tuple(
                coord=(ocr_rect.l, ocr_rect.t, ocr_rect.r, ocr_rect.b),
                origin=CoordOrigin.TOPLEFT,
            )
        )
        cell = TextCell(
            index=cell_idx,
            text=text,
            orig=text,
            from_ocr=True,
            confidence=1.0,
            rect=rect,
        )
        return cell, None

    # --------- Main entry point ---------

    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Iterable[Page]:
        """Run DeepSeek OCR on each page crop and yield updated pages."""
        if not self.enabled:
            yield from page_batch
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        for page in page_batch:
            if page._backend is None or not page._backend.is_valid():  # noqa: SLF001
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: list[TextCell] = []

                crop_data = self._collect_crops(page, ocr_rects)

                with ThreadPoolExecutor(max_workers=self.options.max_concurrent_requests) as executor:
                    futures = [
                        executor.submit(self._process_crop, cell_idx, ocr_rect, image)
                        for cell_idx, ocr_rect, image in crop_data
                    ]

                    for future in as_completed(futures):
                        cell, error_msg = future.result()
                        if cell is not None:
                            all_ocr_cells.append(cell)
                        if error_msg is not None:
                            conv_res.errors.append(
                                ErrorItem(
                                    component_type=DoclingComponentType.MODEL,
                                    module_name=__name__,
                                    error_message=error_msg,
                                )
                            )

                all_ocr_cells.sort(key=lambda c: c.index)
                self.post_process_cells(all_ocr_cells, page)

            yield page

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        """Return the options class for this OCR engine."""
        return DeepseekOcrOptions

