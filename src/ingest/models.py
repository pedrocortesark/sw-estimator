"""Canonical output contract for the ingest subsystem.

Every parser, regardless of input format, must ultimately produce instances of
:class:`Document`.  Downstream modules (chunking, embedding, retrieval) operate
exclusively on ``Document`` objects and have no knowledge of the source format.

Design rationale
----------------
*Homogeneity*: the chunking stage processes ``content`` and propagates
``metadata`` without needing to know whether the document came from a PDF scan
or a structured JSON file.

*Traceability by construction*: every document carries ``source_name``
(matches an entry in ``data_catalog.yaml``), ``source_location`` (physical
path or URL), and, when the format allows it, ``page_number`` / ``section_title``
so that citations in the final answer are precise.

*Escape hatch*: ``extra`` is an open dict for parser-specific metadata that
does not fit the canonical schema (DOCX template fields, revision authors, …).
Those values are preserved but not required by any downstream module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata propagated with every document through the pipeline.

    The first three fields come from the data catalog and are mandatory for
    every document, regardless of source format.  The rest are populated by the
    parser when the format allows it.
    """

    # ---- catalog-level (always present) ------------------------------------
    source_name: str
    """Matches an entry in ``data_catalog.yaml``."""

    source_location: str
    """Original physical path or URL of the source file."""

    ingested_at: datetime
    """UTC timestamp set by the normalizer at extraction time."""

    # ---- document-level (populated by the parser when available) -----------
    document_id: str
    """Stable identifier within the source (e.g. filename stem or hash)."""

    document_title: Optional[str] = None
    document_created_at: Optional[datetime] = None
    document_author: Optional[str] = None

    page_number: Optional[int] = None
    """For paginated formats (PDF, paged DOCX)."""

    section_title: Optional[str] = None
    """For structured formats (DOCX headings, JSON section keys)."""

    contains_pii: bool = False

    extra: dict = Field(default_factory=dict)
    """Open dict for parser-specific metadata that does not fit the schema."""


class Document(BaseModel):
    """The canonical output of the ingest subsystem.

    Every parser, regardless of input format, must produce instances of this
    class.  Downstream chunking, embedding, and retrieval operate exclusively on
    ``Document`` objects.
    """

    content: str
    metadata: DocumentMetadata
