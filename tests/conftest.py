"""Shared pytest fixtures for the sw-estimator test suite."""

import asyncio
import time

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import create_app

# ---------------------------------------------------------------------------
# LLM rate-limit throttle
# ---------------------------------------------------------------------------
# Anthropic free tier: 5 req/min → 1 call every 12 s minimum.
# This fixture is autouse for any test tagged hard | soft | judge so the
# live-LLM suites don't need to manage delays themselves.
_LLM_MARKERS = {"hard", "soft", "judge", "golden"}
# 5 req/min → 12 s minimum, but the 4 000 output-token/min limit combined
# with instructor's immediate retries requires a wider gap.  20 s gives ~3
# calls/min, well below both limits.
_MIN_INTERVAL_S: float = 20.0
_last_llm_call: list[float] = [0.0]  # mutable cell shared across fixtures


@pytest_asyncio.fixture(autouse=True)
async def _llm_throttle(request):
    """Sleep between live-LLM tests to respect the 5 req/min rate limit."""
    markers = {m.name for m in request.node.iter_markers()}
    if not markers & _LLM_MARKERS:
        yield
        return

    elapsed = time.monotonic() - _last_llm_call[0]
    wait = _MIN_INTERVAL_S - elapsed
    if wait > 0:
        await asyncio.sleep(wait)

    yield

    _last_llm_call[0] = time.monotonic()


@pytest.fixture
def test_app():
    """Return a fresh FastAPI app instance.

    Exposing the app separately from the HTTP client lets individual tests
    set ``test_app.dependency_overrides`` before issuing requests, which is
    the idiomatic FastAPI way to inject fake services without monkey-patching.
    """
    return create_app()


@pytest_asyncio.fixture
async def client(test_app) -> AsyncClient:
    """AsyncClient wired directly to the FastAPI app (no network needed).

    ASGITransport lets httpx call the app in-process, bypassing the TCP stack.
    This makes tests fast and deterministic.

    Usage in tests:
        async def test_something(client: AsyncClient):
            response = await client.get("/health")
    """
    async with AsyncClient(
        transport=ASGITransport(app=test_app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        yield ac
