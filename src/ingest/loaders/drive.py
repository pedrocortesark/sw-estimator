"""Google Drive loader — stub implementation.

Full implementation requires ``google-auth`` and ``google-api-python-client``,
which are not in the current dependency set.  Add them to ``pyproject.toml``
before enabling this loader.

Expected interface once implemented::

    load(file_id: str, *, credentials_path: str | Path) -> bytes
"""

from __future__ import annotations


def load(file_id: str, *, credentials_path: str | None = None) -> bytes:  # noqa: ARG001
    """Download a file from Google Drive by its file ID.

    Not yet implemented.  Raises ``NotImplementedError`` until the
    ``google-auth`` / ``google-api-python-client`` dependencies are added and
    the OAuth flow is configured.

    Args:
        file_id:          Google Drive file ID (the ``id`` field in the API).
        credentials_path: Path to a service-account JSON key file.

    Raises:
        NotImplementedError: Always, until this loader is implemented.
    """
    raise NotImplementedError(
        "Google Drive loader is not yet implemented.  "
        "Add 'google-auth' and 'google-api-python-client' to pyproject.toml "
        "and implement the OAuth / service-account flow in this module."
    )
