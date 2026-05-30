"""DOCX parser — section-aware Word document parser.

Intermediate representation
---------------------------
``list[ParsedSection]``

Each ``ParsedSection`` corresponds to one structural section of the document,
delimited by a Heading-style paragraph.  The body is the concatenated text of
all non-heading paragraphs and table cells between two consecutive headings.

Rationale for section-level granularity
-----------------------------------------
Modern Word proposal templates (Alcance, Entregables, Cronograma, …) encode
their structure in Heading styles.  Emitting one ``Document`` per section lets
the retriever fetch the *relevant section*, not the whole 40-page proposal.
``section_title`` in the metadata makes citations human-readable.

Tables
------
Table cells are flattened to markdown and appended to the current section's
body.  Cell content is joined with ``|`` so the LLM can parse structure.

Dependency: ``python-docx`` is already in ``pyproject.toml``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field


@dataclass
class ParsedSection:
    """A structural section extracted from a DOCX file."""

    heading: str | None
    """The heading text that opens this section, or ``None`` for the preamble."""

    heading_level: int
    """Heading level (1–9, matching Word Heading 1 … Heading 9), or 0 for no heading."""

    body: str
    """Concatenated paragraph and table content under this heading."""

    extra: dict = field(default_factory=dict)
    """Additional per-document metadata (author from core properties, etc.)."""


def parse(raw_bytes: bytes) -> list[ParsedSection]:
    """Parse a DOCX file into a list of sections.

    Each section spans from one heading to the next (exclusive).  A synthetic
    preamble section with ``heading=None`` collects any content before the
    first heading.

    Args:
        raw_bytes: Raw bytes of the ``.docx`` file.

    Returns:
        List of :class:`ParsedSection` instances.  At least one section is
        always returned (the whole document as a single section when no
        headings are found).

    Raises:
        ValueError: If the bytes cannot be opened as a valid DOCX file.
    """
    try:
        from docx import Document  # noqa: PLC0415
        from docx.oxml.ns import qn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("python-docx is required for DOCX parsing.") from exc

    try:
        doc = Document(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot open DOCX file: {exc}") from exc

    extra = _extract_core_properties(doc)

    sections: list[ParsedSection] = []
    current_heading: str | None = None
    current_level: int = 0
    current_lines: list[str] = []

    for block in _iter_blocks(doc):
        if block["type"] == "heading":
            # Flush previous section
            _flush(sections, current_heading, current_level, current_lines, extra)
            current_heading = block["text"]
            current_level = block["level"]
            current_lines = []
        else:
            text = block["text"].strip()
            if text:
                current_lines.append(text)

    # Flush last section
    _flush(sections, current_heading, current_level, current_lines, extra)

    return sections or [
        ParsedSection(heading=None, heading_level=0, body="", extra=extra)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flush(
    sections: list[ParsedSection],
    heading: str | None,
    level: int,
    lines: list[str],
    extra: dict,
) -> None:
    body = "\n\n".join(lines).strip()
    if heading or body:
        sections.append(
            ParsedSection(
                heading=heading,
                heading_level=level,
                body=body,
                extra=extra,
            )
        )


def _iter_blocks(doc: object):  # type: ignore[return]
    """Yield heading and paragraph blocks from the document body, including tables."""
    from docx.oxml.ns import qn  # noqa: PLC0415

    for element in doc.element.body:  # type: ignore[union-attr]
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            # Access the paragraph via the document paragraphs list is slow;
            # parse the XML directly for style and text.
            style_elem = element.find(f".//{{{_w}}}pStyle")
            style_name = (
                style_elem.get(f"{{{_w}}}val", "") if style_elem is not None else ""
            )
            text = "".join(node.text or "" for node in element.iter(f"{{{_w}}}t"))
            level = _heading_level(style_name)
            yield {
                "type": "heading" if level else "paragraph",
                "text": text,
                "level": level,
            }

        elif tag == "tbl":
            yield {"type": "paragraph", "text": _table_to_markdown(element), "level": 0}


_w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _heading_level(style_name: str) -> int:
    """Return 1–9 for Heading styles, 0 otherwise."""
    s = style_name.lower()
    if s.startswith("heading"):
        try:
            return int(s.replace("heading", "").strip())
        except ValueError:
            return 1
    # Title / Subtitle treated as level 1
    if s in ("title", "subtitle"):
        return 1
    return 0


def _table_to_markdown(tbl_element: object) -> str:
    """Convert a ``<w:tbl>`` XML element to a markdown table string."""
    rows: list[list[str]] = []
    for tr in tbl_element.findall(f".//{{{_w}}}tr"):  # type: ignore[union-attr]
        cells: list[str] = []
        for tc in tr.findall(f".//{{{_w}}}tc"):
            cell_text = (
                "".join(node.text or "" for node in tc.iter(f"{{{_w}}}t"))
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    padded = [r + [""] * (col_count - len(r)) for r in rows]

    lines = [
        "| " + " | ".join(padded[0]) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _extract_core_properties(doc: object) -> dict:
    """Extract author and created date from DOCX core properties if available."""
    extra: dict = {}
    try:
        props = doc.core_properties  # type: ignore[union-attr]
        if props.author:
            extra["author"] = props.author
        if props.created:
            extra["created"] = props.created.isoformat()
        if props.title:
            extra["title"] = props.title
    except Exception:  # noqa: BLE001
        pass
    return extra
