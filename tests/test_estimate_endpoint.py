"""End-to-end integration tests for POST /api/v1/estimate.

All tests use a ``FakeEstimationService`` injected via FastAPI's
``dependency_overrides`` mechanism — no LLM or Redis connection required.

Two notes on field names vs. the user spec
------------------------------------------
* The request field is ``transcript`` (not ``description``) — this is the
  canonical name in ``EstimationRequest``.
* Cost fields are ``total_cost_usd`` / ``cost_usd`` (not ``*_eur``) — these
  match the Pydantic schema validated by Instructor on every real LLM call.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.dependencies import get_estimation_service
from src.guardrails.input import InputGuardrailViolation
from src.main import create_app
from src.schemas.estimation import (
    EstimationResponse,
    EstimationResult,
    Phase,
    Task,
    TeamMember,
    UsageCost,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_TRANSCRIPT = (
    "A small B2B SaaS to manage employee equipment loans across office teams, "
    "with role-based access and Slack notifications."
)

# Three-phase valid EstimationResult — arithmetic is correct within ±5 %
_PHASE_1 = Phase(
    name="Discovery & Architecture",
    tasks=[Task(name="Requirements workshop", hours=8.0, cost_usd=500.00)],
    total_hours=8.0,
    total_cost_usd=500.00,
)
_PHASE_2 = Phase(
    name="Backend Development",
    tasks=[Task(name="API & data model", hours=32.0, cost_usd=2000.00)],
    total_hours=32.0,
    total_cost_usd=2000.00,
)
_PHASE_3 = Phase(
    name="QA & Deployment",
    tasks=[Task(name="Testing and CI/CD", hours=16.0, cost_usd=1000.00)],
    total_hours=16.0,
    total_cost_usd=1000.00,
)

_FAKE_RESULT = EstimationResult(
    executive_summary="Equipment loan SaaS — 56 hours, two-person team.",
    phases=[_PHASE_1, _PHASE_2, _PHASE_3],
    total_hours=56.0,
    total_cost_usd=3500.00,
    team_composition=[
        TeamMember(role="Backend Engineer", count=1, dedication="100%"),
        TeamMember(role="Frontend Engineer", count=1, dedication="100%"),
    ],
    duration_weeks=4.0,
    confidence_pct=70.0,
)

_FAKE_RESPONSE = EstimationResponse(
    estimation=_FAKE_RESULT,
    provider_used="openai",
    model_used="gpt-4o-mini",
    usage=UsageCost(
        input_tokens=800, output_tokens=400, total_tokens=1200, cost_usd=0.001
    ),
    cached=False,
    prompt_version="v1",
)

# ---------------------------------------------------------------------------
# FakeEstimationService
# ---------------------------------------------------------------------------


class FakeEstimationService:
    """Drop-in replacement for EstimationService — records calls, returns canned data."""

    def __init__(self) -> None:
        self.calls: list = []

    async def estimate(self, request):
        self.calls.append(request)
        return _FAKE_RESPONSE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_svc() -> FakeEstimationService:
    return FakeEstimationService()


@pytest_asyncio.fixture
async def endpoint_client(fake_svc):
    """AsyncClient + app with FakeEstimationService injected."""
    _app = create_app()
    _app.dependency_overrides[get_estimation_service] = lambda: fake_svc
    async with AsyncClient(
        transport=ASGITransport(app=_app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        yield ac
    _app.dependency_overrides.pop(get_estimation_service, None)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_payload_returns_200(endpoint_client):
    resp = await endpoint_client.post(
        "/api/v1/estimate",
        json={"transcript": VALID_TRANSCRIPT},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_response_has_required_fields(endpoint_client):
    resp = await endpoint_client.post(
        "/api/v1/estimate",
        json={"transcript": VALID_TRANSCRIPT},
    )
    body = resp.json()
    assert "estimation" in body
    assert "phases" in body["estimation"]
    assert "total_cost_usd" in body["estimation"]
    assert "cached" in body
    assert "prompt_version" in body


@pytest.mark.asyncio
async def test_total_cost_equals_sum_of_phases(endpoint_client):
    """total_cost_usd must equal the sum of phase costs (arithmetic rule)."""
    resp = await endpoint_client.post(
        "/api/v1/estimate",
        json={"transcript": VALID_TRANSCRIPT},
    )
    body = resp.json()["estimation"]
    phase_sum = sum(p["total_cost_usd"] for p in body["phases"])
    assert abs(body["total_cost_usd"] - phase_sum) / max(phase_sum, 1) < 0.05


@pytest.mark.asyncio
async def test_cached_is_false_on_fresh_call(endpoint_client):
    resp = await endpoint_client.post(
        "/api/v1/estimate",
        json={"transcript": VALID_TRANSCRIPT},
    )
    assert resp.json()["cached"] is False


@pytest.mark.asyncio
async def test_prompt_version_present(endpoint_client):
    resp = await endpoint_client.post(
        "/api/v1/estimate",
        json={"transcript": VALID_TRANSCRIPT},
    )
    assert resp.json()["prompt_version"] == "v1"


@pytest.mark.asyncio
async def test_service_receives_transcript(endpoint_client, fake_svc):
    """The endpoint must forward the transcript to the service unchanged."""
    await endpoint_client.post(
        "/api/v1/estimate",
        json={"transcript": VALID_TRANSCRIPT},
    )
    assert len(fake_svc.calls) == 1
    assert fake_svc.calls[0].transcript == VALID_TRANSCRIPT


@pytest.mark.asyncio
async def test_optional_fields_forwarded_to_service(fake_svc):
    """project_type, detail_level, output_format must be forwarded to the service."""
    _app = create_app()
    _app.dependency_overrides[get_estimation_service] = lambda: fake_svc
    async with AsyncClient(
        transport=ASGITransport(app=_app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        await ac.post(
            "/api/v1/estimate",
            json={
                "transcript": VALID_TRANSCRIPT,
                "project_type": "web_saas",
                "detail_level": "medium",
                "output_format": "phases_table",
            },
        )
    req = fake_svc.calls[0]
    assert req.project_type == "web_saas"
    assert req.detail_level == "medium"
    assert req.output_format == "phases_table"


# ---------------------------------------------------------------------------
# Validation error tests (422 from Pydantic — no service needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_transcript_returns_422(endpoint_client):
    resp = await endpoint_client.post("/api/v1/estimate", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_transcript_too_short_returns_422(endpoint_client):
    resp = await endpoint_client.post(
        "/api/v1/estimate",
        json={"transcript": "too short"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_guardrail_violation_returns_400():
    """InputGuardrailViolation must be translated to HTTP 400 with reason."""

    class _GuardrailService:
        async def estimate(self, request):
            raise InputGuardrailViolation(
                message="Prompt injection attempt detected.",
                reason="prompt_injection",
            )

    _app = create_app()
    _app.dependency_overrides[get_estimation_service] = lambda: _GuardrailService()
    async with AsyncClient(
        transport=ASGITransport(app=_app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        resp = await ac.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )

    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "prompt_injection"
    assert "detail" in body


# ---------------------------------------------------------------------------
# Upstream error tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unexpected_service_error_returns_500():
    """Unhandled exceptions must become HTTP 500."""

    class _BrokenService:
        async def estimate(self, request):
            raise Exception("upstream error")

    _app = create_app()
    _app.dependency_overrides[get_estimation_service] = lambda: _BrokenService()
    async with AsyncClient(
        transport=ASGITransport(app=_app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        resp = await ac.post(
            "/api/v1/estimate",
            json={"transcript": VALID_TRANSCRIPT},
        )

    assert resp.status_code == 500
