"""Filesystem loader — reads raw bytes from a local path.

Loaders are responsible only for *physical access* to a source.  They have
no knowledge of the file's content or format; they just deliver bytes to the
parser layer.
"""

from __future__ import annotations

from pathlib import Path


def load(path: str | Path) -> bytes:
    """Return the raw bytes of a local file.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        Raw file bytes.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path points to a directory.
    """
    return Path(path).read_bytes()
