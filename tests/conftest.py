"""Shared pytest fixtures for the sw-estimator test suite."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import create_app


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

