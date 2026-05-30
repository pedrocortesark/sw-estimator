"""Data catalog loader.

The catalog is a YAML file (``data_catalog.yaml``) that declares every
ingestion source in the project.  Each entry describes:

* *what* the source is (``source_name``, ``description``),
* *where* it lives (``location`` — a local path or URL),
* *how* to parse it (``format``, optional ``strategy`` for PDF).

The catalog is the single place where source-level configuration lives.
Parsers and normalizers are source-agnostic; the orchestrator reads the catalog
to decide which loader / parser to invoke.

Example ``data_catalog.yaml``::

    sources:
      - source_name: historical_budgets
        location: data/budgets/
        format: json
        description: Historical project budget records (one JSON file per project)

      - source_name: meeting_transcripts
        location: data/transcripts/
        format: txt
        description: Client meeting transcriptions (timestamped format from 2024)

      - source_name: signed_contracts
        location: data/contracts/
        format: pdf
        strategy: fast
        description: Signed contracts in digitally-generated PDF format

      - source_name: proposal_templates
        location: data/proposals/
        format: docx
        description: Proposal Word templates with Alcance / Entregables / Cronograma sections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Format = Literal["json", "txt", "xlsx", "docx", "pdf"]
PdfStrategy = Literal["fast", "hi_res"]


@dataclass
class CatalogEntry:
    """A single source registered in ``data_catalog.yaml``."""

    source_name: str
    """Stable identifier — matches ``DocumentMetadata.source_name``."""

    location: str
    """Local directory / file path or URL.  Relative paths are resolved from
    the directory containing ``data_catalog.yaml``."""

    format: Format
    """File format for all files in this source."""

    description: str = ""
    """Human-readable description of the source."""

    strategy: PdfStrategy = "fast"
    """PDF extraction strategy.  Ignored for non-PDF formats."""

    extra: dict = field(default_factory=dict)
    """Additional source-level configuration (e.g. auth hints, encoding)."""


def load_catalog(path: str | Path) -> dict[str, CatalogEntry]:
    """Load and validate a ``data_catalog.yaml`` file.

    Args:
        path: Path to the YAML catalog file.

    Returns:
        Dict mapping ``source_name`` → :class:`CatalogEntry`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If the YAML is malformed or a required field is
                           missing.
        ImportError:       If ``PyYAML`` is not installed.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load data_catalog.yaml.  Install with: uv add pyyaml"
        ) from exc

    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"data_catalog.yaml not found at: {catalog_path}")

    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict) or "sources" not in raw:
        raise ValueError(
            f"'{catalog_path}' must be a YAML mapping with a top-level 'sources' list."
        )

    catalog: dict[str, CatalogEntry] = {}
    for item in raw["sources"]:
        if not isinstance(item, dict):
            raise ValueError(
                f"Each entry in 'sources' must be a mapping; got: {item!r}"
            )
        try:
            entry = CatalogEntry(
                source_name=item["source_name"],
                location=item["location"],
                format=item["format"],
                description=item.get("description", ""),
                strategy=item.get("strategy", "fast"),
                extra={
                    k: v
                    for k, v in item.items()
                    if k
                    not in (
                        "source_name",
                        "location",
                        "format",
                        "description",
                        "strategy",
                    )
                },
            )
        except KeyError as exc:
            raise ValueError(
                f"Catalog entry missing required field {exc}: {item}"
            ) from exc

        if entry.source_name in catalog:
            raise ValueError(
                f"Duplicate source_name '{entry.source_name}' in catalog '{catalog_path}'."
            )
        catalog[entry.source_name] = entry

    return catalog
