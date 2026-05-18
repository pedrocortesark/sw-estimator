"""Local document text extraction — Camino B.

Extracts plain text from PDF and Word files so that attachment content can
be injected into the LLM prompt as ordinary text.  No provider API is called;
extraction happens in-process using open-source libraries.

Why local extraction over provider Files API (Camino A)
--------------------------------------------------------
- Provider-agnostic: extracted text is a plain string, usable by any LLM
  already wired through LiteLLM — no extra per-provider upload logic needed.
- Word (.docx) is not natively supported by any Files API today.
- Cost control: only the text we explicitly include is billed as tokens.
- Prepares the ground for RAG chunking in a future module: the text is
  already in memory and ready to be split/embedded.
"""

from __future__ import annotations

import io

from src.core.exceptions import EstimatorError

# Lazy imports — these libraries are only needed when a file is actually
# uploaded, so we defer the import to avoid penalising startup time.


class UnsupportedFileTypeError(EstimatorError):
    """Raised when an uploaded file has an extension we cannot extract."""


# ---------------------------------------------------------------------------
# Separators used when concatenating extracted content to the transcript
# ---------------------------------------------------------------------------

ATTACHMENT_HEADER = "--- attachment: {filename} ---"
ATTACHMENT_FOOTER = "--- end of {filename} ---"


def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from a PDF or Word document.

    Args:
        filename: Original filename including extension (used to detect type).
        content:  Raw bytes of the uploaded file.

    Returns:
        Extracted text as a single string.  Pages/paragraphs are joined with
        newlines.  Whitespace is normalised but content is otherwise unchanged.

    Raises:
        UnsupportedFileTypeError: If the file extension is not .pdf or .docx.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return _extract_pdf(content)
    if lower.endswith(".docx"):
        return _extract_docx(content)

    suffix = filename.rsplit(".", 1)[-1] if "." in filename else "(none)"
    raise UnsupportedFileTypeError(
        f"Unsupported attachment type '.{suffix}'. Accepted: .pdf, .docx"
    )


def build_attachment_block(filename: str, text: str) -> str:
    """Wrap extracted text in clearly delimited block for the LLM prompt.

    The header/footer markers let the model understand that the content
    comes from a specific attached document, not from the user transcript.
    """
    header = ATTACHMENT_HEADER.format(filename=filename)
    footer = ATTACHMENT_FOOTER.format(filename=filename)
    return f"{header}\n{text.strip()}\n{footer}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    return "\n".join(pages)


def _extract_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
