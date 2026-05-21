"""Family 1 — Hard deterministic tests.

These tests verify structural and numerical properties of the estimation
output that hold regardless of what the LLM says.  The LLM response is
treated as an opaque input; the assertions are computable without any
additional model call.

What this family catches
------------------------
- Schema regression: a prompt change that drops a required field is caught
  immediately because Instructor raises a validation error (or the test
  fails on the missing attribute).
- Arithmetic invariants: total_hours and total_cost_usd must equal the sum of
  their phases (already enforced by the Pydantic validators, but retested
  here so CI fails loudly if someone weakens the validators).
- Sanity bounds: hours and cost must be positive and below a hard ceiling.
  These bounds are wider than any expected range in the golden dataset —
  they catch hallucinated values (e.g. 0 h, or 9 999 999 h), not judgment
  calls about scope.
- Non-empty strings: names, summaries, and role labels must not be blank.

Running
-------
Run in isolation (no live LLM needed if using mocks, but these tests use
the real service to catch regressions on real outputs):

    pytest tests/test_hard_properties.py -v -m hard

Skip during offline development:

    pytest tests/ -m "not hard"
"""

from __future__ import annotations

import os

import pytest

from tests.golden_dataset import _GOLDENS

# ---------------------------------------------------------------------------
# Absolute sanity bounds — wider than any golden expected range
# ---------------------------------------------------------------------------
_MIN_HOURS: float = 1.0
_MAX_HOURS: float = 100_000.0
_MIN_COST: float = 0.0
_MAX_COST: float = 10_000_000.0
_MIN_DURATION_WEEKS: float = 0.1
_MIN_PHASES: int = 1
_MIN_TASKS_PER_PHASE: int = 1
_MIN_TEAM_MEMBERS: int = 1


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def estimation_service():
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No LLM API key found — set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
    from src.services.estimation import EstimationService

    return EstimationService(cache=None, openai_client=None)


# ---------------------------------------------------------------------------
# Parametric test — one test per golden case
# ---------------------------------------------------------------------------


@pytest.mark.hard
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    _GOLDENS,
    ids=[f"{g.additional_metadata['category']}_{i}" for i, g in enumerate(_GOLDENS)],
)
async def test_hard_properties(golden, estimation_service):
    """Assert all hard structural and numerical invariants for one golden case."""
    from src.schemas.estimation import (
        EstimationRequest,
        EstimationResponse,
        EstimationResult,
    )

    category: str = golden.additional_metadata["category"]
    request = EstimationRequest(transcript=golden.input)
    try:
        response = await estimation_service.estimate(request)
    except Exception as exc:
        if "rate_limit" in str(exc).lower():
            pytest.skip(
                f"[{category}] Anthropic rate limit hit — re-run with a wider "
                "throttle or a higher-tier API key."
            )
        raise

    est: EstimationResult = response.estimation

    # ------------------------------------------------------------------ #
    # H1 — Schema: top-level response type                                #
    # ------------------------------------------------------------------ #
    assert isinstance(response, EstimationResponse), (
        f"[{category}] Response is not an EstimationResponse: {type(response)}"
    )

    # ------------------------------------------------------------------ #
    # H2 — Schema: required top-level fields are present and non-empty   #
    # ------------------------------------------------------------------ #
    assert est.executive_summary.strip(), (
        f"[{category}] executive_summary must not be blank"
    )
    assert est.phases, f"[{category}] phases list must not be empty"
    assert est.team_composition, f"[{category}] team_composition must not be empty"
    assert response.provider_used.strip(), (
        f"[{category}] provider_used must not be blank"
    )
    assert response.model_used.strip(), f"[{category}] model_used must not be blank"

    # ------------------------------------------------------------------ #
    # H3 — Sanity bounds: total hours and cost                           #
    # ------------------------------------------------------------------ #
    assert est.total_hours >= _MIN_HOURS, (
        f"[{category}] total_hours={est.total_hours} is below minimum {_MIN_HOURS}"
    )
    assert est.total_hours <= _MAX_HOURS, (
        f"[{category}] total_hours={est.total_hours} exceeds ceiling {_MAX_HOURS}"
    )
    assert est.total_cost_usd >= _MIN_COST, (
        f"[{category}] total_cost_usd={est.total_cost_usd} is negative"
    )
    assert est.total_cost_usd <= _MAX_COST, (
        f"[{category}] total_cost_usd={est.total_cost_usd} exceeds ceiling {_MAX_COST}"
    )

    # ------------------------------------------------------------------ #
    # H4 — Sanity bounds: calendar duration                              #
    # ------------------------------------------------------------------ #
    assert est.duration_weeks >= _MIN_DURATION_WEEKS, (
        f"[{category}] duration_weeks={est.duration_weeks} is below minimum {_MIN_DURATION_WEEKS}"
    )

    # ------------------------------------------------------------------ #
    # H5 — Arithmetic invariants: grand totals match phase subtotals     #
    # (Pydantic validators already enforce this; we retest to make the   #
    # CI failure surface here, not inside the service call above.)       #
    # ------------------------------------------------------------------ #
    computed_hours = sum(p.total_hours for p in est.phases)
    computed_cost = sum(p.total_cost_usd for p in est.phases)
    tolerance = 0.05

    hours_drift = abs(est.total_hours - computed_hours) / max(computed_hours, 1.0)
    assert hours_drift <= tolerance, (
        f"[{category}] total_hours={est.total_hours:.1f} differs from "
        f"sum of phases={computed_hours:.1f} by {hours_drift:.1%}"
    )
    cost_drift = abs(est.total_cost_usd - computed_cost) / max(computed_cost, 1.0)
    assert cost_drift <= tolerance, (
        f"[{category}] total_cost_usd={est.total_cost_usd:.2f} differs from "
        f"sum of phases={computed_cost:.2f} by {cost_drift:.1%}"
    )

    # ------------------------------------------------------------------ #
    # H6 — Structure: phases have at least one task with non-blank names #
    # ------------------------------------------------------------------ #
    assert len(est.phases) >= _MIN_PHASES, (
        f"[{category}] Expected at least {_MIN_PHASES} phase(s), got {len(est.phases)}"
    )
    for phase in est.phases:
        assert phase.name.strip(), f"[{category}] Phase has a blank name"
        assert len(phase.tasks) >= _MIN_TASKS_PER_PHASE, (
            f"[{category}] Phase '{phase.name}' has no tasks"
        )
        for task in phase.tasks:
            assert task.name.strip(), (
                f"[{category}] Phase '{phase.name}' contains a task with a blank name"
            )
            assert task.hours > 0, (
                f"[{category}] Task '{task.name}' has non-positive hours: {task.hours}"
            )
            assert task.cost_usd >= 0, (
                f"[{category}] Task '{task.name}' has negative cost: {task.cost_usd}"
            )

    # ------------------------------------------------------------------ #
    # H7 — Structure: team composition roles are non-blank               #
    # ------------------------------------------------------------------ #
    assert len(est.team_composition) >= _MIN_TEAM_MEMBERS, (
        f"[{category}] Expected at least {_MIN_TEAM_MEMBERS} team member(s)"
    )
    for member in est.team_composition:
        assert member.role.strip(), (
            f"[{category}] team_composition contains a member with a blank role"
        )
        assert member.count >= 1, (
            f"[{category}] Role '{member.role}' has count={member.count} (must be ≥ 1)"
        )

    # ------------------------------------------------------------------ #
    # H8 — Confidence: must be within [0, 100]                          #
    # (also enforced by Pydantic ge/le constraints — kept as explicit    #
    # assertion for CI clarity)                                          #
    # ------------------------------------------------------------------ #
    assert 0.0 <= est.confidence_pct <= 100.0, (
        f"[{category}] confidence_pct={est.confidence_pct} outside [0, 100]"
    )

    # ------------------------------------------------------------------ #
    # H9 — Usage: token counts and cost are non-negative                 #
    # ------------------------------------------------------------------ #
    usage = response.usage
    assert usage.input_tokens >= 0
    assert usage.output_tokens >= 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens, (
        f"[{category}] usage total_tokens mismatch: "
        f"{usage.total_tokens} != {usage.input_tokens} + {usage.output_tokens}"
    )
    assert usage.cost_usd >= 0
