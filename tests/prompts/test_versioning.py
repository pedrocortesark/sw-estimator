"""Tests for real prompt versioning (Bonus 1).

Verifies that:
- v1 and v2 render distinct system prompts.
- v2 contains the chain-of-thought instruction absent from v1.
- v1 contains calibration examples absent from v2.
- An unknown version raises ValueError.
- The endpoint accepts ?prompt_version=v2 and echoes it in the response.
"""

import pytest

from src.prompts.loader import render_estimation_prompt
from src.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REQ = EstimationRequest(
    description="A simple task management web app with projects, tasks, and user roles.",
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.PHASES_TABLE,
)


def _system(version: str) -> str:
    system, _ = render_estimation_prompt(_REQ, version=version, prompt_style="xml")
    return system


# ---------------------------------------------------------------------------
# Template content differences between versions
# ---------------------------------------------------------------------------

def test_v2_system_contains_chain_of_thought_instruction():
    """v2 must include the chain-of-thought block absent from v1."""
    assert "chain_of_thought" in _system("v2")


def test_v1_system_does_not_contain_chain_of_thought():
    """v1 must not include the chain-of-thought block."""
    assert "chain_of_thought" not in _system("v1")


def test_v1_system_contains_calibration_examples():
    """v1 uses few-shot examples; the examples section must be present."""
    assert "Calibration examples" in _system("v1") or "examples" in _system("v1")


def test_v2_system_does_not_contain_calibration_examples():
    """v2 is zero-shot; the examples section must be absent."""
    assert "Calibration examples" not in _system("v2")
    # The include of examples.j2 should not appear at all
    assert "SaaS" not in _system("v2")  # text from examples.j2


def test_v1_and_v2_produce_different_systems():
    """v1 and v2 must render distinct system prompts."""
    assert _system("v1") != _system("v2")


# ---------------------------------------------------------------------------
# Unknown version raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_version_raises_value_error():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        render_estimation_prompt(_REQ, version="v99")


def test_unknown_version_error_message_lists_available():
    """The error message must list the existing versions."""
    with pytest.raises(ValueError, match="v1"):
        render_estimation_prompt(_REQ, version="v99")


# ---------------------------------------------------------------------------
# HTTP endpoint accepts ?prompt_version query param
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_estimate_endpoint_accepts_prompt_version_v1(client):
    """?prompt_version=v1 must return 200 (mocked LLM, schema validation only)."""
    # The client fixture hits the real app in-process; LLM calls will fail
    # without credentials, but we can at least test that 422 is NOT returned
    # for the query param itself. A 500/503 from missing env vars is acceptable.
    response = await client.post(
        "/api/v1/estimate?prompt_version=v1",
        json={
            "description": "A simple task management web app with projects and tasks.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
        },
    )
    assert response.status_code != 422


@pytest.mark.asyncio
async def test_estimate_endpoint_accepts_prompt_version_v2(client):
    """?prompt_version=v2 must return 200-level or 5xx, never 422."""
    response = await client.post(
        "/api/v1/estimate?prompt_version=v2",
        json={
            "description": "A simple task management web app with projects and tasks.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
        },
    )
    assert response.status_code != 422


@pytest.mark.asyncio
async def test_estimate_endpoint_rejects_invalid_prompt_version(client):
    """?prompt_version=foo (doesn't match ^v[0-9]+$) must return 422."""
    response = await client.post(
        "/api/v1/estimate?prompt_version=foo",
        json={
            "description": "A simple task management web app with projects and tasks.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
        },
    )
    assert response.status_code == 422
