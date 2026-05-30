"""JSON parser — converts structured JSON files to markdown blocks.

Intermediate representation
---------------------------
``list[ParsedBlock]``

Each ``ParsedBlock`` corresponds to one top-level record (when the JSON is an
array) or to the whole document (when the JSON is an object).  The ``content``
field is a markdown rendering of the record, not a raw ``json.dumps()`` string.

Rationale for markdown rendering over raw JSON
----------------------------------------------
Dumping the JSON verbatim mixes keys (structural noise) with values
(semantic signal), producing embeddings that are hard to retrieve.  Rendering
to markdown promotes important keys to headings and exposes values as readable
prose, dramatically improving retrieval precision.

The parser knows the *shape* of the data (a list of budget records, for
example); the normalizer is shape-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedBlock:
    """A single markdown-rendered block extracted from a JSON file."""

    block_id: str
    """Stable identifier: array index as string, or document key path."""

    content: str
    """Markdown representation of this block, ready for embedding."""

    raw: dict[str, Any] = field(default_factory=dict)
    """The original dict so the normalizer can promote fields to metadata."""


def parse(raw_bytes: bytes, *, source_hint: str = "") -> list[ParsedBlock]:
    """Parse a JSON file into a list of markdown-rendered blocks.

    Supports two root shapes:

    * **Array of objects** — each element becomes one ``ParsedBlock``.
      ``block_id`` is the zero-based index.
    * **Single object** — the whole document becomes one ``ParsedBlock``.
      ``block_id`` is ``"root"``.

    Args:
        raw_bytes:   Raw bytes of the JSON file.
        source_hint: Filename or path, used only in error messages.

    Returns:
        List of :class:`ParsedBlock` instances.

    Raises:
        ValueError: If the bytes cannot be decoded as valid JSON.
    """
    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot decode JSON from '{source_hint}': {exc}") from exc

    if isinstance(data, list):
        return [
            ParsedBlock(
                block_id=str(i),
                content=_render_record(record, index=i),
                raw=record if isinstance(record, dict) else {},
            )
            for i, record in enumerate(data)
        ]

    # Single object
    return [
        ParsedBlock(
            block_id="root",
            content=_render_record(data, index=None),
            raw=data if isinstance(data, dict) else {},
        )
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_record(record: Any, *, index: int | None) -> str:
    """Render a JSON value to a readable markdown string.

    Dicts become a list of ``**key**: value`` lines; scalars and other types
    fall back to a code block with ``json.dumps``.
    """
    if not isinstance(record, dict):
        return f"```json\n{json.dumps(record, ensure_ascii=False, indent=2)}\n```"

    header = f"## Record {index}" if index is not None else "## Document"
    lines = [header, ""]
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            # Nested structure: render as indented JSON block
            rendered = json.dumps(value, ensure_ascii=False, indent=2)
            lines.append(f"**{key}**:\n```json\n{rendered}\n```")
        else:
            lines.append(f"**{key}**: {value}")

    return "\n".join(lines)
