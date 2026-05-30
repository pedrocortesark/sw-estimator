"""Ingest orchestrator — glues loaders, parsers, and normalizers.

The orchestrator is the single public entry point for the ingest subsystem.
Callers provide a ``source_name`` (matching a ``data_catalog.yaml`` entry) and
receive back a flat list of canonical :class:`~src.ingest.models.Document`
objects ready for chunking and embedding.

Execution flow
--------------
For each file found under ``CatalogEntry.location``:

1. **Load** — a loader retrieves raw bytes (filesystem or HTTP).
2. **Parse** — the format-appropriate parser produces an intermediate
   representation (IR).
3. **Normalize** — the canonical normalizer converts the IR to
   ``list[Document]``, propagating catalog metadata.

All three steps are format-agnostic from the caller's perspective.

Error handling
--------------
By default, a single file failure raises immediately (``fail_fast=True``).
Pass ``fail_fast=False`` to log the error and continue with the remaining
files — useful for batch jobs where partial ingestion is preferable to a
complete abort.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from src.ingest.catalog import CatalogSource, load_catalog
from src.ingest.models import Document

logger = logging.getLogger(__name__)


def ingest(
    source_name: str,
    *,
    catalog_path: str | Path = "data_catalog.yaml",
    base_dir: str | Path | None = None,
    fail_fast: bool = True,
) -> list[Document]:
    """Ingest all files for a named source from the catalog.

    Args:
        source_name:  Name of the source to ingest (must exist in catalog).
        catalog_path: Path to ``data_catalog.yaml``.
        base_dir:     Base directory for resolving relative locations in the
                      catalog.  Defaults to the directory of ``catalog_path``.
        fail_fast:    If ``True`` (default), raise on the first file error.
                      If ``False``, log errors and continue.

    Returns:
        Flat list of :class:`~src.ingest.models.Document` objects from all
        files in the source.

    Raises:
        KeyError:  If ``source_name`` is not found in the catalog.
        Exception: Any loader / parser error when ``fail_fast=True``.
    """
    catalog_path = Path(catalog_path)
    base = Path(base_dir) if base_dir else catalog_path.parent

    catalog = load_catalog(catalog_path)
    if source_name not in catalog:
        raise KeyError(
            f"Source '{source_name}' not found in catalog '{catalog_path}'.  "
            f"Available: {sorted(catalog)}"
        )

    entry = catalog[source_name]
    documents: list[Document] = []

    for file_path in _iter_files(entry, base):
        try:
            docs = _ingest_file(file_path, entry)
            documents.extend(docs)
            logger.debug("ingested %d documents from %s", len(docs), file_path)
        except Exception:
            if fail_fast:
                raise
            logger.exception("Failed to ingest '%s' — skipping.", file_path)

    logger.info(
        "ingest complete: source=%s files_scanned=%d documents_produced=%d",
        source_name,
        sum(1 for _ in _iter_files(entry, base)),
        len(documents),
    )
    return documents


def ingest_file(
    file_path: str | Path,
    *,
    catalog_path: str | Path = "data_catalog.yaml",
    source_name: str,
) -> list[Document]:
    """Ingest a single file using the catalog entry for ``source_name``.

    Useful for incremental ingestion (e.g. processing a newly arrived file
    without re-scanning the whole directory).

    Args:
        file_path:    Absolute or relative path to the file.
        catalog_path: Path to ``data_catalog.yaml``.
        source_name:  Catalog entry whose format / strategy settings apply.

    Returns:
        List of :class:`~src.ingest.models.Document` objects from this file.
    """
    catalog = load_catalog(catalog_path)
    if source_name not in catalog:
        raise KeyError(f"Source '{source_name}' not found in catalog '{catalog_path}'.")
    return _ingest_file(Path(file_path), catalog[source_name])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_files(entry: CatalogSource, base: Path) -> Iterator[Path]:
    """Yield file paths for all files matching the catalog entry's location.

    If ``location`` is a directory, yields all files whose suffix matches the
    format.  If it is a single file path, yields just that path.  HTTP URLs
    are yielded as-is (as Path objects with a fake suffix — the loader handles
    the scheme).
    """
    loc = entry.location
    if loc.startswith("http://") or loc.startswith("https://"):
        yield Path(loc)
        return

    resolved = base / loc if not Path(loc).is_absolute() else Path(loc)

    if resolved.is_file():
        yield resolved
        return

    if resolved.is_dir():
        suffix = f".{entry.format}"
        for f in sorted(resolved.iterdir()):
            if f.is_file() and f.suffix.lower() == suffix:
                yield f
        return

    logger.warning("Location '%s' does not exist — no files to ingest.", resolved)


def _ingest_file(file_path: Path, entry: CatalogSource) -> list[Document]:
    """Run the full load → parse → normalize pipeline for one file."""
    from src.ingest import loaders  # noqa: PLC0415
    from src.ingest.normalizers import canonical  # noqa: PLC0415

    loc_str = str(file_path)

    # --- Load ---------------------------------------------------------------
    if loc_str.startswith("http://") or loc_str.startswith("https://"):
        raw_bytes = loaders.http.load(loc_str)
    else:
        raw_bytes = loaders.filesystem.load(file_path)

    # --- Parse --------------------------------------------------------------
    fmt = entry.format

    if fmt == "json":
        from src.ingest.parsers import json_parser  # noqa: PLC0415

        ir = json_parser.parse(raw_bytes, source_hint=loc_str)
        return canonical.from_json_blocks(ir, entry, loc_str)

    if fmt == "txt":
        from src.ingest.parsers import txt_parser  # noqa: PLC0415

        ir = txt_parser.parse(raw_bytes)
        return canonical.from_turns(ir, entry, loc_str)

    if fmt == "xlsx":
        from src.ingest.parsers import xlsx_parser  # noqa: PLC0415

        ir = xlsx_parser.parse(raw_bytes, source_hint=loc_str)
        return canonical.from_xlsx_tables(ir, entry, loc_str)

    if fmt == "docx":
        from src.ingest.parsers import docx_parser  # noqa: PLC0415

        ir = docx_parser.parse(raw_bytes)
        return canonical.from_docx_sections(ir, entry, loc_str)

    if fmt == "pdf":
        from src.ingest.parsers import pdf_parser  # noqa: PLC0415

        ir = pdf_parser.parse(raw_bytes, strategy=entry.strategy, source_hint=loc_str)
        return canonical.from_pdf_pages(ir, entry, loc_str)

    raise ValueError(f"Unsupported format '{fmt}' for source '{entry.source_name}'.")
