"""HTTP/HTTPS loader — fetches raw bytes from a URL.

Uses ``httpx`` (already a project dependency) with a conservative timeout.
Authentication is intentionally out of scope here; callers that need OAuth or
API-key flows should pre-build the URL with the token embedded, or subclass
this loader for their specific scheme.
"""

from __future__ import annotations

import httpx

_DEFAULT_TIMEOUT = 30.0  # seconds


def load(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> bytes:
    """Download a remote file and return its raw bytes.

    Args:
        url:     HTTP or HTTPS URL of the resource.
        timeout: Request timeout in seconds (default 30 s).

    Returns:
        Raw response body as bytes.

    Raises:
        httpx.HTTPStatusError:   On 4xx / 5xx responses.
        httpx.TimeoutException:  If the server does not respond in time.
    """
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.content
