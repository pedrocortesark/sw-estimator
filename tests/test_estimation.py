"""Tests for POST /api/v1/estimate."""

import pytest
from httpx import AsyncClient

from src.dependencies import get_estimation_service
from src.schemas.estimation import (
    EstimationResponse,
    EstimationResult,
    Phase,
    Task,
    TeamMember,
    UsageCost,
)

# A realistic transcript that passes the min_length=20 validation rule
VALID_TRANSCRIPT = (
    "The client wants a web app where users can upload CSV files "
    "and visualise the data as interactive charts. The backend should "
    "store the files in S3 and expose a REST API."
)

# Minimal but schema-valid EstimationResult
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

_MOCK_RESPONSE = EstimationResponse(
    estimation=_MOCK_RESULT,
    provider_used="anthropic",
    model_used="claude-3-5-haiku-20241022",
    usage=UsageCost(input_tokens=500, output_tokens=200, total_tokens=700, cost_usd=0.000150),
)


class _MockService:
    async def estimate(self, request):
        return _MOCK_RESPONSE


class _FailingService:
    async def estimate(self, request):
        raise RuntimeError("LLM network timeout")


@pytest.mark.asyncio
async def test_estimate_returns_200_with_valid_transcript(client: AsyncClient, test_app):
    """A valid transcript must return HTTP 200 and a well-shaped response."""
    test_app.dependency_overrides[get_estimation_service] = lambda: _MockService()
    try:
        response = await client.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )
    finally:
        test_app.dependency_overrides.pop(get_estimation_service, None)

    assert response.status_code == 200
    body = response.json()
    assert "estimation" in body
    assert "provider_used" in body
    assert "model_used" in body


@pytest.mark.asyncio
async def test_estimate_response_content(client: AsyncClient, test_app):
    """The response body must match what the mocked service returns."""
    test_app.dependency_overrides[get_estimation_service] = lambda: _MockService()
    try:
        response = await client.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )
    finally:
        test_app.dependency_overrides.pop(get_estimation_service, None)

    body = response.json()
    assert body["provider_used"] == "anthropic"
    assert body["model_used"] == "claude-3-5-haiku-20241022"
    assert body["estimation"]["total_hours"] == 40.0


@pytest.mark.asyncio
async def test_estimate_returns_422_when_transcript_missing(client: AsyncClient):
    """FastAPI must return 422 Unprocessable Entity when 'transcript' is absent."""
    response = await client.post("/api/v1/estimate", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_422_when_transcript_too_short(client: AsyncClient):
    """Transcripts shorter than 20 characters must be rejected with 422."""
    response = await client.post(
        "/api/v1/estimate",
        json={"transcript": "too short"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_estimate_returns_400_for_unknown_provider(client: AsyncClient, test_app):
    """Requesting an unsupported provider name must return HTTP 400."""
    from src.core.exceptions import UnknownProviderError

    class _BadProviderService:
        async def estimate(self, request):
            raise UnknownProviderError(f"Unsupported provider: '{request.provider}'.")

    test_app.dependency_overrides[get_estimation_service] = lambda: _BadProviderService()
    try:
        response = await client.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT, "provider": "grok"},
        )
    finally:
        test_app.dependency_overrides.pop(get_estimation_service, None)

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_estimate_returns_500_on_unexpected_error(client: AsyncClient, test_app):
    """If the service raises an unexpected exception, the router must return 500."""
    test_app.dependency_overrides[get_estimation_service] = lambda: _FailingService()
    try:
        response = await client.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )
    finally:
        test_app.dependency_overrides.pop(get_estimation_service, None)

    assert response.status_code == 500

