"""Canonical normalizer — converts parser IR to Document instances.

This module is the thin "glue" between parser-specific intermediate
representations and the canonical :class:`~src.ingest.models.Document`
contract.

Each public function accepts:

* The parser's intermediate representation (IR).
* A :class:`~src.ingest.catalog.CatalogEntry` describing the source.
* The physical location of the file (path or URL).

And returns a ``list[Document]`` ready for downstream chunking / embedding.

Granularity decisions
---------------------
* **JSON** — one ``Document`` per ``ParsedBlock`` (one per array record).
* **TXT** — one ``Document`` per ``Turn`` (one per speaker utterance).
  When the whole file is a single raw turn, one ``Document`` is returned.
* **XLSX** — one ``Document`` per ``ParsedTable`` (one per worksheet).
* **DOCX** — one ``Document`` per ``ParsedSection`` (one per heading-delimited
  section).  Empty sections are filtered out.
* **PDF** — one ``Document`` per ``ParsedPage`` (one per non-empty page).
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from typing import TYPE_CHECKING

from src.ingest.models import Document, DocumentMetadata
from src.ingest.parsers.docx_parser import ParsedSection
from src.ingest.parsers.json_parser import ParsedBlock
from src.ingest.parsers.pdf_parser import ParsedPage
from src.ingest.parsers.txt_parser import Turn
from src.ingest.parsers.xlsx_parser import ParsedTable

if TYPE_CHECKING:
    from src.ingest.catalog import CatalogSource


def from_json_blocks(
    blocks: list[ParsedBlock],
    entry: "CatalogSource",
    source_location: str,
) -> list[Document]:
    """Normalize JSON parser output."""
    ingested_at = _now()
    return [
        Document(
            content=block.content,
            metadata=DocumentMetadata(
                source_name=entry.source_name,
                source_location=source_location,
                ingested_at=ingested_at,
                document_id=_stable_id(source_location, block.block_id),
                section_title=block.raw.get("title") or block.raw.get("name"),
                extra={
                    k: v
                    for k, v in block.raw.items()
                    if k not in ("title", "name", "content", "text")
                },
            ),
        )
        for block in blocks
    ]


def from_turns(
    turns: list[Turn],
    entry: "CatalogSource",
    source_location: str,
) -> list[Document]:
    """Normalize TXT parser output (speaker turns)."""
    ingested_at = _now()
    return [
        Document(
            content=_turn_content(turn),
            metadata=DocumentMetadata(
                source_name=entry.source_name,
                source_location=source_location,
                ingested_at=ingested_at,
                document_id=_stable_id(source_location, str(i)),
                extra=_turn_extra(turn),
            ),
        )
        for i, turn in enumerate(turns)
        if turn.text.strip()
    ]


def from_xlsx_tables(
    tables: list[ParsedTable],
    entry: "CatalogSource",
    source_location: str,
) -> list[Document]:
    """Normalize XLSX parser output."""
    ingested_at = _now()
    return [
        Document(
            content=table.markdown,
            metadata=DocumentMetadata(
                source_name=entry.source_name,
                source_location=source_location,
                ingested_at=ingested_at,
                document_id=_stable_id(source_location, table.sheet_name),
                section_title=table.sheet_name,
                extra={"row_count": table.row_count, "col_count": table.col_count},
            ),
        )
        for table in tables
        if table.markdown.strip()
    ]


def from_docx_sections(
    sections: list[ParsedSection],
    entry: "CatalogSource",
    source_location: str,
) -> list[Document]:
    """Normalize DOCX parser output."""
    ingested_at = _now()
    docs: list[Document] = []
    for i, section in enumerate(sections):
        if not section.body.strip() and not section.heading:
            continue
        content = (
            f"{'#' * section.heading_level} {section.heading}\n\n{section.body}".strip()
            if section.heading
            else section.body
        )
        extra = dict(section.extra)
        if section.heading_level:
            extra["heading_level"] = section.heading_level

        docs.append(
            Document(
                content=content,
                metadata=DocumentMetadata(
                    source_name=entry.source_name,
                    source_location=source_location,
                    ingested_at=ingested_at,
                    document_id=_stable_id(source_location, str(i)),
                    document_title=section.extra.get("title"),
                    document_author=section.extra.get("author"),
                    section_title=section.heading,
                    extra=extra,
                ),
            )
        )
    return docs


def from_pdf_pages(
    pages: list[ParsedPage],
    entry: "CatalogSource",
    source_location: str,
) -> list[Document]:
    """Normalize PDF parser output."""
    ingested_at = _now()
    return [
        Document(
            content=page.text,
            metadata=DocumentMetadata(
                source_name=entry.source_name,
                source_location=source_location,
                ingested_at=ingested_at,
                document_id=_stable_id(source_location, str(page.page_number)),
                page_number=page.page_number,
            ),
        )
        for page in pages
        if page.text.strip()
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _stable_id(source_location: str, fragment: str) -> str:
    """Short, stable identifier for a document fragment within a source."""
    raw = f"{source_location}::{fragment}"
    return sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]  # noqa: S324


def _turn_content(turn: Turn) -> str:
    if turn.speaker and turn.timestamp:
        return f"[{turn.timestamp}] {turn.speaker}: {turn.text}"
    if turn.speaker:
        return f"{turn.speaker}: {turn.text}"
    return turn.text


def _turn_extra(turn: Turn) -> dict:
    extra: dict = {}
    if turn.speaker:
        extra["speaker"] = turn.speaker
    if turn.timestamp:
        extra["timestamp"] = turn.timestamp
    return extra
