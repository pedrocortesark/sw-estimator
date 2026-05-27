"""PDF parser — text extraction with strategy selection.

Intermediate representation
---------------------------
``list[ParsedPage]``

Each ``ParsedPage`` corresponds to one page in the PDF.  The ``text`` field is
the extracted text for that page; empty pages are filtered out.

Strategy selection
------------------
Two strategies are supported:

``"fast"`` (default)
    Uses ``pypdf`` for direct text extraction.  Fast, zero ML dependencies,
    suitable for digitally-generated PDFs without complex table layouts.

``"hi_res"``
    Uses ``unstructured`` with ``strategy="hi_res"``.  Employs computer-vision
    models to detect tables, headers, and column layouts.  Required for:

    * PDFs that are scanned images (OCR needed).
    * PDFs where tables carry information that ``pypdf`` mis-orders.

    Costs: ~10× slower than ``fast``; requires a large optional dependency
    bundle (``unstructured[pdf]``).  Use only when explicitly configured in
    ``data_catalog.yaml``.

The recommended operational rule: configure ``strategy: fast`` for
digitally-generated contracts and proposals; set ``strategy: hi_res`` only
when the catalog entry flags the source as scanned or table-heavy.

Dependencies
------------
* ``pypdf`` — already in ``pyproject.toml``.
* ``unstructured[pdf]`` — optional; install when ``hi_res`` is needed::

    uv add "unstructured[pdf]"
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

Strategy = Literal["fast", "hi_res"]


@dataclass
class ParsedPage:
    """A single page extracted from a PDF."""

    page_number: int
    """1-based page number."""

    text: str
    """Extracted text content of this page."""


def parse(
    raw_bytes: bytes,
    *,
    strategy: Strategy = "fast",
    source_hint: str = "",
) -> list[ParsedPage]:
    """Parse a PDF file into a list of per-page text blocks.

    Args:
        raw_bytes:   Raw bytes of the PDF file.
        strategy:    ``"fast"`` (pypdf) or ``"hi_res"`` (unstructured).
        source_hint: Filename or path, used only in error messages.

    Returns:
        List of :class:`ParsedPage` instances (empty pages excluded).

    Raises:
        ValueError:  If the bytes cannot be opened as a valid PDF.
        ImportError: If ``unstructured[pdf]`` is not installed and
                     ``strategy="hi_res"`` is requested.
    """
    if strategy == "hi_res":
        return _parse_hi_res(raw_bytes, source_hint=source_hint)
    return _parse_fast(raw_bytes, source_hint=source_hint)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _parse_fast(raw_bytes: bytes, *, source_hint: str = "") -> list[ParsedPage]:
    """Extract text with ``pypdf`` — fast, no ML, digital PDFs only."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pypdf is required for PDF parsing.") from exc

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{source_hint}': {exc}") from exc

    pages: list[ParsedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(ParsedPage(page_number=i, text=text))

    return pages


def _parse_hi_res(raw_bytes: bytes, *, source_hint: str = "") -> list[ParsedPage]:
    """Extract text with ``unstructured`` hi_res strategy — slow, ML-backed.

    Merges unstructured ``Element`` objects back into per-page buckets so the
    normalizer receives the same ``list[ParsedPage]`` shape regardless of
    strategy.
    """
    try:
        from unstructured.partition.pdf import partition_pdf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "unstructured[pdf] is required for hi_res PDF parsing.  "
            "Install with: uv add 'unstructured[pdf]'"
        ) from exc

    with _tmp_pdf(raw_bytes) as tmp_path:
        try:
            elements = partition_pdf(filename=str(tmp_path), strategy="hi_res")
        except Exception as exc:
            raise ValueError(
                f"unstructured failed to parse '{source_hint}': {exc}"
            ) from exc

    # Group elements by page number
    pages_dict: dict[int, list[str]] = {}
    for elem in elements:
        pn = getattr(getattr(elem, "metadata", None), "page_number", None) or 1
        pages_dict.setdefault(pn, []).append(str(elem))

    return [
        ParsedPage(page_number=pn, text="\n\n".join(texts))
        for pn, texts in sorted(pages_dict.items())
        if any(t.strip() for t in texts)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tmp_pdf(raw_bytes: bytes):
    """Context manager: write bytes to a temp file and yield its path."""
    import tempfile  # noqa: PLC0415
    from contextlib import contextmanager  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    @contextmanager
    def _ctx():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(raw_bytes)
            tmp = Path(f.name)
        try:
            yield tmp
        finally:
            tmp.unlink(missing_ok=True)

    return _ctx()
