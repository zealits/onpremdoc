import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResultPostprocessor:
    """
    Lightweight stub implementation used when the original hierarchical
    postprocessor module is not available.

    The real implementation is expected to adjust heading levels and section
    hierarchy in the Docling document based on TOC / bookmarks. For now we
    simply keep the document unchanged so the rest of the pipeline can run.
    """

    def __init__(self, doc: Any, source: Optional[str] = None) -> None:
        self.doc = doc
        self.source = source

    def process(self) -> None:
        """
        No-op placeholder. Logs that hierarchical postprocessing is skipped
        but does not modify the underlying document.
        """
        if self.source:
            logger.info(
                "ResultPostprocessor stub active – skipping hierarchical "
                "postprocessing for source %s",
                self.source,
            )
        else:
            logger.info(
                "ResultPostprocessor stub active – skipping hierarchical "
                "postprocessing (no source provided)",
            )
