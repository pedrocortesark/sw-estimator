"""Tests for GET /health."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    """The health endpoint must always return HTTP 200."""
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_shape(client: AsyncClient):
    """The response must contain the three expected keys."""
    response = await client.get("/health")
    body = response.json()

    assert "status" in body
    assert "env" in body
    assert "llm_models" in body


@pytest.mark.asyncio
async def test_health_status_value(client: AsyncClient):
    """The 'status' field must be 'ok'."""
    response = await client.get("/health")
    assert response.json()["status"] == "ok"
