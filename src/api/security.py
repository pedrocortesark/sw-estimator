"""API-key authentication for the Session 9 routers.

Two independent keys protect the two routers: a ``RETRIEVAL_API_KEY`` holder
cannot call the estimate endpoint and vice versa. Keys are compared with
``secrets.compare_digest`` (constant-time) — never ``==``, which leaks length
and prefix information through timing.

Session 15 adds service-to-service authentication: ``SERVICE_TOKEN`` protects
endpoints from unauthorized callers within the Docker network. When set, the
``X-Service-Token`` header is required. When empty, authentication is disabled
(for local development without a business-backend).
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from src.core.config import get_settings

_API_KEY_HEADER = "X-API-Key"
_SERVICE_TOKEN_HEADER = "X-Service-Token"


def _verify(provided: str | None, expected: str | None) -> None:
    """Raise 401 unless ``provided`` matches the configured ``expected`` key."""
    if not expected or not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )


async def require_retrieval_key(
    x_api_key: str | None = Header(default=None, alias=_API_KEY_HEADER),
) -> None:
    """FastAPI dependency guarding ``POST /v1/retrieval/search``."""
    _verify(x_api_key, get_settings().RETRIEVAL_API_KEY)


async def require_estimate_key(
    x_api_key: str | None = Header(default=None, alias=_API_KEY_HEADER),
) -> None:
    """FastAPI dependency guarding ``POST /v1/estimate/from-transcript``."""
    _verify(x_api_key, get_settings().ESTIMATE_API_KEY)


async def require_service_token(
    x_service_token: str | None = Header(default=None, alias=_SERVICE_TOKEN_HEADER),
) -> None:
    """FastAPI dependency for service-to-service authentication (Session 15).

    When ``SERVICE_TOKEN`` is configured, callers must provide the matching
    ``X-Service-Token`` header. When empty, authentication is disabled (dev mode).
    """
    settings = get_settings()
    if not settings.service_token:
        return  # Auth disabled — development mode
    if not x_service_token or not secrets.compare_digest(x_service_token, settings.service_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing service token.",
            headers={"WWW-Authenticate": _SERVICE_TOKEN_HEADER},
        )
