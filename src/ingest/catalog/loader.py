"""Load and validate the YAML catalog.

Reads a YAML file with ``yaml.safe_load``, validates it against
:class:`DataCatalog`. The two-line ``load_catalog`` is intentionally trivial —
all the heavy lifting (decision rules, duplicate detection) lives in the
model, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from src.ingest.catalog.models import CatalogDecision, DataCatalog


def load_catalog(path: str | Path) -> DataCatalog:
    """Parse ``path`` as YAML and validate as a :class:`DataCatalog`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DataCatalog.model_validate(raw)


def generate_audit_report(catalog: DataCatalog) -> str:
    """Generate a Markdown audit report for the given catalog.

    Sections:
    - Summary counters (total / included / review / excluded).
    - Included sources with ``is_rag_ready`` flag.
    - Sources in review with their ``decision_reason``.
    - Excluded sources with their ``decision_reason``.

    Args:
        catalog: A validated :class:`DataCatalog` instance.

    Returns:
        A Markdown string ready to print or write to a file.
    """
    included = [s for s in catalog.sources if s.decision is CatalogDecision.INCLUDE]
    review = [s for s in catalog.sources if s.decision is CatalogDecision.REVIEW]
    excluded = [s for s in catalog.sources if s.decision is CatalogDecision.EXCLUDE]

    lines: list[str] = []

    # Header
    lines += [
        f"# Data Catalog Audit Report",
        f"",
        f"**Version:** {catalog.version}",
        f"",
    ]
    if catalog.description:
        lines += [catalog.description.strip(), ""]

    # Summary counters
    lines += [
        "## Summary",
        "",
        f"| Status   | Count |",
        f"|----------|-------|",
        f"| Total    | {len(catalog.sources):>5} |",
        f"| Included | {len(included):>5} |",
        f"| Review   | {len(review):>5} |",
        f"| Excluded | {len(excluded):>5} |",
        "",
    ]

    # Included sources
    lines += ["## Included Sources", ""]
    if included:
        for src in included:
            rag_flag = "✅ rag_ready" if src.quality.is_rag_ready else "⚠️  not_rag_ready"
            lines += [
                f"### `{src.name}` ({src.format})  {rag_flag}",
                "",
            ]
            if src.description:
                lines += [src.description.strip(), ""]
            q = src.quality
            lines += [
                f"- **Quality:** completeness={q.completeness} consistency={q.consistency}"
                f" actuality={q.actuality} reliability={q.reliability}",
                f"- **Sensitivity:** {src.sensitivity.access_level}"
                + (f" — PII: {', '.join(src.sensitivity.pii_flags)}" if src.sensitivity.has_pii else ""),
                "",
            ]
    else:
        lines += ["_(none)_", ""]

    # Review sources
    lines += ["## Sources Under Review", ""]
    if review:
        for src in review:
            lines += [
                f"### `{src.name}` ({src.format})",
                "",
                f"> **Reason:** {src.decision_reason.strip()}",
                "",
            ]
    else:
        lines += ["_(none)_", ""]

    # Excluded sources
    lines += ["## Excluded Sources", ""]
    if excluded:
        for src in excluded:
            lines += [
                f"### `{src.name}` ({src.format})",
                "",
                f"> **Reason:** {src.decision_reason.strip()}",
                "",
            ]
    else:
        lines += ["_(none)_", ""]

    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    """CLI: ``python -m src.ingest.catalog.loader data/catalog/catalog.yaml``."""
    if len(argv) != 2:
        print("usage: python -m src.ingest.catalog.loader <path.yaml>")
        return 1
    catalog = load_catalog(argv[1])
    included = catalog.included_sources()
    print(f"Catalog version: {catalog.version}")
    print(f"Sources total:  {len(catalog.sources)}")
    print(f"Sources included: {len(included)}")
    for src in included:
        print(f"  - {src.name} ({src.format}) — {src.description!r}")
    print()
    print(generate_audit_report(catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
