"""XLSX parser — converts spreadsheet tables to markdown.

Intermediate representation
---------------------------
``list[ParsedTable]``

Each ``ParsedTable`` corresponds to one contiguous rectangular table found in
a worksheet.  The ``markdown`` field is a GitHub-Flavored Markdown table
suitable for embedding.  Sheet metadata (name, position) is preserved so the
normalizer can populate ``section_title``.

Design decisions
----------------
* Uses ``openpyxl`` in read-only mode (lazy cell evaluation, no recalculation).
  This avoids triggering formula recalculation, which would require Excel or a
  compatible engine.
* Only the primary data region of each sheet is extracted.  Cells outside the
  used range, floating text boxes, charts, and conditional-format-only
  annotations are deliberately ignored.
* Hidden sheets are skipped by default (``skip_hidden=True``).  Pass
  ``skip_hidden=False`` to include them.

Dependency: ``openpyxl`` is not in the current ``pyproject.toml``.  Add it
before using this parser::

    uv add openpyxl
"""

from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class ParsedTable:
    """A single table extracted from one worksheet."""

    sheet_name: str
    """Name of the Excel worksheet (tab label)."""

    table_index: int
    """Zero-based index of this table within the sheet (usually 0)."""

    markdown: str
    """GitHub-Flavored Markdown table string."""

    row_count: int
    col_count: int


def parse(
    raw_bytes: bytes,
    *,
    source_hint: str = "",
    skip_hidden: bool = True,
) -> list[ParsedTable]:
    """Parse an XLSX file into a list of markdown tables.

    Args:
        raw_bytes:   Raw bytes of the ``.xlsx`` file.
        source_hint: Filename or path, used only in error messages.
        skip_hidden: Whether to ignore hidden worksheets (default ``True``).

    Returns:
        List of :class:`ParsedTable` instances, one per non-empty worksheet.

    Raises:
        ImportError:  If ``openpyxl`` is not installed.
        ValueError:   If the bytes cannot be opened as a valid workbook.
    """
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required for XLSX parsing.  Add it with: uv add openpyxl"
        ) from exc

    try:
        wb = openpyxl.load_workbook(
            io.BytesIO(raw_bytes), read_only=True, data_only=True
        )
    except Exception as exc:
        raise ValueError(f"Cannot open XLSX workbook '{source_hint}': {exc}") from exc

    tables: list[ParsedTable] = []
    for sheet in wb.worksheets:
        if skip_hidden and sheet.sheet_state != "visible":
            continue

        rows = _read_rows(sheet)
        if not rows:
            continue

        markdown = _rows_to_markdown(rows)
        tables.append(
            ParsedTable(
                sheet_name=sheet.title,
                table_index=len(tables),
                markdown=markdown,
                row_count=len(rows),
                col_count=max(len(r) for r in rows) if rows else 0,
            )
        )

    wb.close()
    return tables


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_rows(sheet: object) -> list[list[str]]:
    """Extract all non-empty rows from a worksheet as string lists."""
    rows: list[list[str]] = []
    for row in sheet.iter_rows(values_only=True):  # type: ignore[union-attr]
        str_row = [str(cell) if cell is not None else "" for cell in row]
        # Skip entirely empty rows
        if any(cell.strip() for cell in str_row):
            rows.append(str_row)
    return rows


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Convert a 2-D list of strings to a GFM markdown table.

    The first row is treated as the header.  If there is only one row,
    a header-only table is emitted.
    """
    if not rows:
        return ""

    # Normalise column count
    col_count = max(len(r) for r in rows)
    padded = [r + [""] * (col_count - len(r)) for r in rows]

    def _row_line(cells: list[str]) -> str:
        escaped = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
        return "| " + " | ".join(escaped) + " |"

    lines = [_row_line(padded[0]), "| " + " | ".join(["---"] * col_count) + " |"]
    for row in padded[1:]:
        lines.append(_row_line(row))

    return "\n".join(lines)
