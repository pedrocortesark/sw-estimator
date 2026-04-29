"""Shared pytest fixtures for the sw-estimator test suite."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import create_app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """AsyncClient wired directly to the FastAPI app (no network needed).

    ASGITransport lets httpx call the app in-process, bypassing the TCP stack.
    This makes tests fast and deterministic.

    Usage in tests:
        async def test_something(client: AsyncClient):
            response = await client.get("/health")
    """
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        yield ac
