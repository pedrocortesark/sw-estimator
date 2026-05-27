"""Tests for POST /api/v1/estimate."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.dependencies import get_estimation_service
from src.main import create_app
from src.schemas.estimation import (
    DetailLevel,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
    Task,
    TeamMember,
    UsageCost,
)

# A valid payload that passes all Pydantic validations
VALID_PAYLOAD = {
    "transcript": (
        "The client wants a web app where users can upload CSV files "
        "and visualise the data as interactive charts. The backend should "
        "store the files in S3 and expose a REST API."
    ),
    "project_type": ProjectType.WEB_SAAS.value,
    "detail_level": DetailLevel.MEDIUM.value,
    "output_format": OutputFormat.PHASES_TABLE.value,
}

_MOCK_RESULT = EstimationResult(
    executive_summary="Simple dashboard — 40 hours, two developers.",
    phases=[
        Phase(
            name="Backend Development",
            tasks=[Task(name="REST API", hours=40.0, cost_usd=4000.0)],
            total_hours=40.0,
            total_cost_usd=4000.0,
        )
    ],
    total_hours=40.0,
    total_cost_usd=4000.0,
    team_composition=[TeamMember(role="Backend Engineer", count=1, dedication="100%")],
    duration_weeks=2.0,
)

# The fake response our mock will return — deterministic, instant, free
MOCK_RESPONSE = EstimationResponse(
    estimation=_MOCK_RESULT,
    provider_used="anthropic",
    model_used="claude-3-5-haiku-20241022",
    usage=UsageCost(
        input_tokens=500,
        output_tokens=200,
        total_tokens=700,
        cost_usd=0.000150,
    ),
)


class _MockService:
    async def estimate(self, request, project_metadata=None, prompt_version=None):
        return MOCK_RESPONSE


class _FailingService:
    async def estimate(self, request, project_metadata=None, prompt_version=None):
        raise RuntimeError("LLM network timeout")


@pytest_asyncio.fixture
async def api_client():
    """AsyncClient with MockService injected via dependency_overrides."""
    app = create_app()
    app.dependency_overrides[get_estimation_service] = lambda: _MockService()
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_estimate_returns_200_with_valid_payload(api_client: AsyncClient):
    """A valid payload must return HTTP 200 and a well-shaped response."""
    response = await api_client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert "estimation" in body
    assert "prompt_version" in body
    assert "provider_used" in body
    assert "model_used" in body


@pytest.mark.asyncio
async def test_estimate_response_content(api_client: AsyncClient):
    """The response body must match what the mocked service returns."""
    response = await api_client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    body = response.json()
    assert body["provider_used"] == "anthropic"
    assert body["model_used"] == "claude-3-5-haiku-20241022"
    assert body["prompt_version"] == "v1"
    assert body["estimation"]["total_hours"] == 40.0


@pytest.mark.asyncio
async def test_estimate_returns_422_when_body_missing(client: AsyncClient):
    """FastAPI must return 422 when the request body is empty."""
    response = await client.post("/api/v1/estimate", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_422_when_transcript_too_short(client: AsyncClient):
    """Transcripts shorter than 20 characters must be rejected with 422."""
    payload = {**VALID_PAYLOAD, "transcript": "too short"}
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
async def test_estimate_returns_500_on_unexpected_error(client: AsyncClient, test_app):
    """If the LLM service raises an unexpected exception, the router must return 500."""
    test_app.dependency_overrides[get_estimation_service] = lambda: _FailingService()
    try:
        response = await client.post("/api/v1/estimate", json=VALID_PAYLOAD)
    finally:
        test_app.dependency_overrides.pop(get_estimation_service, None)

    assert response.status_code == 500
