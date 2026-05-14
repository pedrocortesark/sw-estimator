"""Tests for POST /api/v1/estimate."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from src.schemas.estimation import (
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    UsageCost,
)

# A valid payload that passes all Pydantic validations
VALID_PAYLOAD = {
    "description": (
        "The client wants a web app where users can upload CSV files "
        "and visualise the data as interactive charts. The backend should "
        "store the files in S3 and expose a REST API."
    ),
    "project_type": ProjectType.WEB_SAAS.value,
    "detail_level": DetailLevel.MEDIUM.value,
    "output_format": OutputFormat.PHASES_TABLE.value,
}

# The fake response our mock will return — deterministic, instant, free
MOCK_RESPONSE = EstimationResponse(
    text="## Estimation\n\n| Task | Hours |\n|---|---|\n| Backend | 40 |\n\n**Total: 40 hours**",
    prompt_version="v1",
    provider_used="anthropic",
    model_used="claude-3-5-haiku-20241022",
    usage=UsageCost(
        input_tokens=500,
        output_tokens=200,
        total_tokens=700,
        cost_usd=0.000150,
    ),
)


@pytest.mark.asyncio
async def test_estimate_returns_200_with_valid_payload(client: AsyncClient):
    """A valid payload must return HTTP 200 and a well-shaped response."""
    with patch(
        "src.routers.estimation.generate_estimation",
        new_callable=AsyncMock,
        return_value=MOCK_RESPONSE,
    ):
        response = await client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert "text" in body
    assert "prompt_version" in body
    assert "provider_used" in body
    assert "model_used" in body


@pytest.mark.asyncio
async def test_estimate_response_content(client: AsyncClient):
    """The response body must match what the mocked service returns."""
    with patch(
        "src.routers.estimation.generate_estimation",
        new_callable=AsyncMock,
        return_value=MOCK_RESPONSE,
    ):
        response = await client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    body = response.json()
    assert body["provider_used"] == "anthropic"
    assert body["model_used"] == "claude-3-5-haiku-20241022"
    assert body["prompt_version"] == "v1"
    assert "40 hours" in body["text"]


@pytest.mark.asyncio
async def test_estimate_returns_422_when_body_missing(client: AsyncClient):
    """FastAPI must return 422 when the request body is empty."""
    response = await client.post("/api/v1/estimate", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_422_when_description_too_short(client: AsyncClient):
    """Descriptions shorter than 20 characters must be rejected with 422."""
    payload = {**VALID_PAYLOAD, "description": "too short"}
    response = await client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_422_with_invalid_project_type(client: AsyncClient):
    """An unknown project_type enum value must return 422."""
    payload = {**VALID_PAYLOAD, "project_type": "blockchain_nft"}
    response = await client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_422_with_invalid_detail_level(client: AsyncClient):
    """An unknown detail_level enum value must return 422."""
    payload = {**VALID_PAYLOAD, "detail_level": "ultra"}
    response = await client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_422_with_invalid_output_format(client: AsyncClient):
    """An unknown output_format enum value must return 422."""
    payload = {**VALID_PAYLOAD, "output_format": "powerpoint"}
    response = await client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_500_on_unexpected_error(client: AsyncClient):
    """If the LLM service raises an unexpected exception, the router must return 500."""
    with patch(
        "src.routers.estimation.generate_estimation",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM network timeout"),
    ):
        response = await client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 500
