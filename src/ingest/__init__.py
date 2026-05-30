"""Ingest subsystem — public API.

The ingest subsystem converts raw source files (JSON, TXT, XLSX, DOCX, PDF)
into canonical :class:`Document` objects consumed by chunking, embedding, and
retrieval.

Quick start::

    from src.ingest import ingest, ingest_file, Document, DocumentMetadata

    # Ingest all files registered under "meeting_transcripts" in the catalog
    docs: list[Document] = ingest("meeting_transcripts")

    # Ingest a single file (incremental / triggered ingestion)
    docs = ingest_file("data/contracts/acme-2024.pdf", source_name="signed_contracts")

Architecture summary
--------------------
Three layers, each with a single responsibility:

``loaders/``
    Physical access — returns raw ``bytes`` from disk, HTTP, or Drive.

``parsers/``
    Format extraction — converts bytes to a format-specific intermediate
    representation (IR) whose granularity matches the document's structure
    (page, section, speaker turn, table row, …).

``normalizers/``
    Contract conversion — maps the IR to canonical :class:`Document` instances,
    injecting catalog metadata (``source_name``, ``source_location``,
    ``ingested_at``) and parser-provided metadata (``page_number``,
    ``section_title``, ``speaker``, …).

See :mod:`src.ingest.orchestrator` for the end-to-end pipeline.
See :mod:`src.ingest.catalog` for the ``data_catalog.yaml`` schema.
"""

from src.ingest.catalog import CatalogSource, load_catalog
from src.ingest.documents.models import Document, DocumentMetadata
from src.ingest.orchestrator import ingest_source

__all__ = [
    "Document",
    "DocumentMetadata",
    "CatalogSource",
    "load_catalog",
    "ingest_source",
]
