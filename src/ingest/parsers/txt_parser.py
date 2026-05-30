"""TXT parser — transcription-aware plain-text parser.

Intermediate representation
---------------------------
``list[Turn]``

Each ``Turn`` is one speaker-turn in a conversation.  The parser detects the
transcript format and emits structured turns rather than raw text, so that:

* The embedding space contains coherent utterances, not arbitrary line breaks.
* Metadata carries ``speaker`` and ``timestamp`` for citation purposes.
* The retriever can answer "who said X" and "when was X discussed".

Supported transcript formats
-----------------------------
1. **Timestamped** (automatic transcription service, 2024 onwards)::

       [10:42:15] Alice: We need to add OAuth before launch.
       [10:42:30] Bob: Agreed, two weeks should be enough.

2. **Legacy** (heterogeneous pre-2024 formats — speaker may be omitted)::

       Alice: We need to add OAuth before launch.
       Agreed, two weeks should be enough.

3. **Raw text** (no speaker detection possible) — the whole file becomes a
   single ``Turn`` with ``speaker=None`` and ``timestamp=None``.

Detection is done by scanning the first 20 lines for the timestamped pattern.
If ≥ 50 % of non-empty lines match, the file is classified as timestamped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# [hh:mm:ss] Speaker: text  (timestamp is optional sub-seconds)
_TIMESTAMPED_RE = re.compile(r"^\[(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+([^:]+):\s+(.*)")
# Speaker: text  (legacy — speaker name has no brackets or timestamp prefix)
_LEGACY_SPEAKER_RE = re.compile(r"^([A-Za-zÀ-ÖØ-öø-ÿ][^:]{0,39}):\s+(.*)")

_DETECTION_LINES = 20
_TIMESTAMPED_THRESHOLD = 0.50


@dataclass
class Turn:
    """A single speaker-turn extracted from a transcript."""

    speaker: str | None
    """Speaker name, or ``None`` if not detected."""

    timestamp: str | None
    """``hh:mm:ss`` string from the transcript header, or ``None``."""

    text: str
    """The utterance content."""


def parse(raw_bytes: bytes) -> list[Turn]:
    """Parse a transcript file into a list of speaker turns.

    Args:
        raw_bytes: Raw bytes of the TXT file (UTF-8 or latin-1).

    Returns:
        List of :class:`Turn` instances.  At minimum one turn is returned
        (the whole file as a single ``Turn`` with no speaker or timestamp).
    """
    text = _decode(raw_bytes)
    lines = text.splitlines()

    fmt = _detect_format(lines)

    if fmt == "timestamped":
        return _parse_timestamped(lines)
    if fmt == "legacy":
        return _parse_legacy(lines)
    return _parse_raw(text)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format(lines: list[str]) -> str:
    sample = [ln for ln in lines[:_DETECTION_LINES] if ln.strip()]
    if not sample:
        return "raw"

    ts_hits = sum(1 for ln in sample if _TIMESTAMPED_RE.match(ln))
    if ts_hits / len(sample) >= _TIMESTAMPED_THRESHOLD:
        return "timestamped"

    legacy_hits = sum(1 for ln in sample if _LEGACY_SPEAKER_RE.match(ln))
    if legacy_hits / len(sample) >= _TIMESTAMPED_THRESHOLD:
        return "legacy"

    return "raw"


# ---------------------------------------------------------------------------
# Per-format parsers
# ---------------------------------------------------------------------------


def _parse_timestamped(lines: list[str]) -> list[Turn]:
    turns: list[Turn] = []
    current: Turn | None = None

    for line in lines:
        m = _TIMESTAMPED_RE.match(line)
        if m:
            if current is not None:
                turns.append(current)
            current = Turn(
                timestamp=m.group(1), speaker=m.group(2).strip(), text=m.group(3)
            )
        elif current is not None and line.strip():
            # Continuation line — append to current turn
            current = Turn(
                timestamp=current.timestamp,
                speaker=current.speaker,
                text=f"{current.text} {line.strip()}",
            )

    if current is not None:
        turns.append(current)

    return turns or _parse_raw("\n".join(lines))


def _parse_legacy(lines: list[str]) -> list[Turn]:
    turns: list[Turn] = []
    current: Turn | None = None

    for line in lines:
        if not line.strip():
            continue
        m = _LEGACY_SPEAKER_RE.match(line)
        if m:
            if current is not None:
                turns.append(current)
            current = Turn(timestamp=None, speaker=m.group(1).strip(), text=m.group(2))
        elif current is not None:
            current = Turn(
                timestamp=None,
                speaker=current.speaker,
                text=f"{current.text} {line.strip()}",
            )
        else:
            turns.append(Turn(timestamp=None, speaker=None, text=line.strip()))

    if current is not None:
        turns.append(current)

    return turns or _parse_raw("\n".join(lines))


def _parse_raw(text: str) -> list[Turn]:
    return [Turn(timestamp=None, speaker=None, text=text.strip())]


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
