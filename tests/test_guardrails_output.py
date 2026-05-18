"""Tests for the output guardrail (Layer 5).

``model_construct`` primer
--------------------------
``EstimationResult.model_construct(**fields)`` creates a Pydantic model instance
by *setting fields directly* on the object, bypassing every validator —
``__init__``, ``@field_validator``, and ``@model_validator`` included.

Why we need it here:
    Some tests need an ``EstimationResult`` where ``confidence_pct < 30`` but
    the ``executive_summary`` does *not* start with ``OUT_OF_SCOPE_PREFIX``.
    If we used the normal constructor, ``check_confidence_prefix`` would raise a
    ``ValueError`` before the guardrail could ever see the object — the scenario
    we want to test would be unreachable.

``model_construct`` is the right tool whenever we need a model in a state that
*should not exist in production* but *can exist in practice* (e.g. data
deserialized from an old cache entry, a provider that bypassed Instructor, or an
edge case at exactly the threshold boundary).

Never use ``model_construct`` in production code that builds objects for clients.
"""

from __future__ import annotations

import pytest

from src.guardrails.output import enforce_scope_response
from src.schemas.estimation import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
    Task,
    TeamMember,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE = LOW_CONFIDENCE_THRESHOLD + 10.0  # safely above threshold
_LOW_CONFIDENCE = LOW_CONFIDENCE_THRESHOLD - 15.0   # safely below threshold


def _valid_phase() -> Phase:
    """Minimal Phase that satisfies all validators."""
    task = Task(name="Implement feature", hours=8.0, cost_usd=500.00)
    return Phase(name="Backend", tasks=[task], total_hours=8.0, total_cost_usd=500.00)


def _valid_result(*, confidence_pct: float = _HIGH_CONFIDENCE) -> EstimationResult:
    """Fully valid EstimationResult built through the normal constructor."""
    phase = _valid_phase()
    return EstimationResult(
        executive_summary="The project is a simple CRUD application.",
        phases=[phase],
        total_hours=8.0,
        total_cost_usd=500.00,
        team_composition=[TeamMember(role="Backend Engineer", count=1, dedication="100%")],
        duration_weeks=2.0,
        confidence_pct=confidence_pct,
    )


def _out_of_scope_result(*, confidence_pct: float = _LOW_CONFIDENCE) -> EstimationResult:
    """Valid result where the summary already carries the required prefix."""
    phase = _valid_phase()
    return EstimationResult(
        executive_summary=f"{OUT_OF_SCOPE_PREFIX} too vague to estimate.",
        phases=[phase],
        total_hours=8.0,
        total_cost_usd=500.00,
        team_composition=[TeamMember(role="Backend Engineer", count=1, dedication="100%")],
        duration_weeks=2.0,
        confidence_pct=confidence_pct,
    )


def _low_confidence_no_prefix(*, confidence_pct: float = _LOW_CONFIDENCE) -> EstimationResult:
    """EstimationResult with low confidence but missing the required prefix.

    Uses ``model_construct`` to bypass ``check_confidence_prefix``, producing
    the kind of object the guardrail must detect and rewrite.
    """
    return EstimationResult.model_construct(
        executive_summary="The project looks straightforward.",
        phases=[_valid_phase()],
        total_hours=8.0,
        total_cost_usd=500.00,
        team_composition=[TeamMember(role="Backend Engineer", count=1, dedication="100%")],
        duration_weeks=2.0,
        confidence_pct=confidence_pct,
    )


# ---------------------------------------------------------------------------
# Tests: high-confidence pass-through
# ---------------------------------------------------------------------------


def test_high_confidence_returns_same_instance() -> None:
    """Results above the threshold must pass through untouched (same object)."""
    result = _valid_result(confidence_pct=_HIGH_CONFIDENCE)
    out = enforce_scope_response(result)
    assert out is result


def test_high_confidence_at_exact_threshold_passes() -> None:
    """confidence_pct == LOW_CONFIDENCE_THRESHOLD is treated as 'good enough'."""
    result = _valid_result(confidence_pct=LOW_CONFIDENCE_THRESHOLD)
    out = enforce_scope_response(result)
    assert out is result


# ---------------------------------------------------------------------------
# Tests: low confidence with correct prefix — pass-through
# ---------------------------------------------------------------------------


def test_low_confidence_with_prefix_returns_same_instance() -> None:
    """If the summary already starts with OUT_OF_SCOPE_PREFIX, no rewrite."""
    result = _out_of_scope_result(confidence_pct=_LOW_CONFIDENCE)
    out = enforce_scope_response(result)
    assert out is result


def test_low_confidence_with_prefix_at_various_levels() -> None:
    """Parametric check: any confidence below threshold + prefix → pass-through."""
    for pct in (0.0, 1.0, 10.0, LOW_CONFIDENCE_THRESHOLD - 0.001):
        result = _out_of_scope_result(confidence_pct=pct)
        assert enforce_scope_response(result) is result


# ---------------------------------------------------------------------------
# Tests: low confidence without prefix — rewrite
# ---------------------------------------------------------------------------


def test_rewrite_summary_starts_with_prefix() -> None:
    result = _low_confidence_no_prefix()
    out = enforce_scope_response(result)
    assert out.executive_summary.startswith(OUT_OF_SCOPE_PREFIX)


def test_rewrite_contains_original_rationale() -> None:
    """The original summary text should appear inside the rewritten summary."""
    original_summary = "The project looks straightforward."
    result = _low_confidence_no_prefix()
    out = enforce_scope_response(result)
    assert original_summary in out.executive_summary


def test_rewrite_total_cost_is_zero() -> None:
    result = _low_confidence_no_prefix()
    out = enforce_scope_response(result)
    assert out.total_cost_usd == 0.0


def test_rewrite_duration_weeks_is_one() -> None:
    result = _low_confidence_no_prefix()
    out = enforce_scope_response(result)
    assert out.duration_weeks == 1.0


def test_rewrite_has_single_phase() -> None:
    result = _low_confidence_no_prefix()
    out = enforce_scope_response(result)
    assert len(out.phases) == 1


def test_rewrite_phase_name_is_not_estimated() -> None:
    result = _low_confidence_no_prefix()
    out = enforce_scope_response(result)
    assert out.phases[0].name == "Not estimated"


def test_rewrite_preserves_confidence_pct() -> None:
    """The rewritten result must carry the original (low) confidence value."""
    result = _low_confidence_no_prefix(confidence_pct=5.0)
    out = enforce_scope_response(result)
    assert out.confidence_pct == 5.0


def test_rewrite_summary_truncated_to_max_length() -> None:
    """A very long original summary must not produce a summary over 1200 chars."""
    long_rationale = "x" * 2000
    result = EstimationResult.model_construct(
        executive_summary=long_rationale,
        phases=[_valid_phase()],
        total_hours=8.0,
        total_cost_usd=500.00,
        team_composition=[],
        duration_weeks=1.0,
        confidence_pct=_LOW_CONFIDENCE,
    )
    out = enforce_scope_response(result)
    assert len(out.executive_summary) <= 1200


# ---------------------------------------------------------------------------
# Test: rewritten result passes full schema validation (round-trip)
# ---------------------------------------------------------------------------


def test_rewrite_produces_valid_model() -> None:
    """The rewritten result must survive model_validate(model_dump()) — the
    guardrail must always return a schema-conformant object."""
    result = _low_confidence_no_prefix()
    out = enforce_scope_response(result)
    # This calls all validators including check_grand_totals and check_confidence_prefix.
    validated = EstimationResult.model_validate(out.model_dump())
    assert validated.executive_summary == out.executive_summary
    assert validated.confidence_pct == out.confidence_pct
