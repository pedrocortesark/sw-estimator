"""Tests for POST /api/v1/estimate."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from src.schemas.estimation import EstimationResponse, UsageCost

# A realistic transcript that passes the min_length=20 validation rule
VALID_TRANSCRIPT = (
    "The client wants a web app where users can upload CSV files "
    "and visualise the data as interactive charts. The backend should "
    "store the files in S3 and expose a REST API."
)

# The fake response our mock will return — deterministic, instant, free
MOCK_RESPONSE = EstimationResponse(
    estimation="## Estimation\n\n| Task | Hours |\n|---|---|\n| Backend | 40 |\n\n**Total: 40 hours**",
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
async def test_estimate_returns_200_with_valid_transcript(client: AsyncClient):
    """A valid transcript must return HTTP 200 and a well-shaped response.

    We mock 'generate_estimation' so the test never calls Anthropic/OpenAI.
    The mock replaces the function only for the duration of this test,
    then restores the original automatically.
    """
    with patch(
        "src.routers.estimation.generate_estimation",
        new_callable=AsyncMock,
        return_value=MOCK_RESPONSE,
    ):
        response = await client.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )

    assert response.status_code == 200
    body = response.json()
    assert "estimation" in body
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
        response = await client.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )

    body = response.json()
    assert body["provider_used"] == "anthropic"
    assert body["model_used"] == "claude-3-5-haiku-20241022"
    assert "40 hours" in body["estimation"]


@pytest.mark.asyncio
async def test_estimate_returns_422_when_transcript_missing(client: AsyncClient):
    """FastAPI must return 422 Unprocessable Entity when 'transcript' is absent.

    This validation is handled automatically by Pydantic — no mock needed
    because the request never reaches our business logic.
    """
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
async def test_estimate_returns_400_for_unknown_provider(client: AsyncClient):
    """Requesting an unsupported provider name must return HTTP 400.

    The service raises UnknownProviderError, which is caught by the global
    exception handler in main.py and converted to a 400 response.
    Here we let the real service run — the error is triggered before any LLM call.
    """
    response = await client.post(
        "/api/v1/estimate",
        json={"transcript": VALID_TRANSCRIPT, "provider": "grok"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_estimate_returns_500_on_unexpected_error(client: AsyncClient):
    """If the LLM service raises an unexpected exception, the router must return 500."""
    with patch(
        "src.routers.estimation.generate_estimation",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM network timeout"),
    ):
        response = await client.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )

    assert response.status_code == 500
