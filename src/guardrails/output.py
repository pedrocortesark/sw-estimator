"""Output guardrails — validate and sanitise the LLM response.

This is the last line of defence before the response is serialised and sent to
the client.  Unlike the input guardrail (which raises exceptions), this layer
uses a *filter policy*: it never raises, always returns a well-formed
``EstimationResult``.

The single rule enforced here mirrors ``EstimationResult.check_confidence_prefix``
but catches the edge cases that slip past it:

* ``confidence_pct == LOW_CONFIDENCE_THRESHOLD`` exactly (the validator uses
  strict ``<``, so this value is technically valid).
* The threshold was changed at runtime via a feature flag after the schema was
  compiled, so old validator bytecode uses a stale value.
* A provider bypassed Instructor and returned a pre-constructed object.
"""

from __future__ import annotations

from src.core.logging import logger
from src.schemas.estimation import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
    Task,
)

# Maximum length of the rewritten executive_summary.
_MAX_SUMMARY_LEN = 1200
# How many characters of the original rationale to embed.
_MAX_ORIGINAL_LEN = 400


def enforce_scope_response(result: EstimationResult) -> EstimationResult:
    """Enforce the low-confidence → out-of-scope declaration rule.

    Args:
        result: ``EstimationResult`` produced by the LLM (already validated by
                Instructor).

    Returns:
        * The *same* instance (``is`` identity preserved) when no rewrite is
          needed — avoids unnecessary object churn.
        * A *new* ``EstimationResult`` with a sanitised summary and a single
          placeholder phase when confidence is below threshold and the summary
          lacks the required prefix.
    """
    # Fast path 1: confidence is high enough — nothing to do.
    if result.confidence_pct >= LOW_CONFIDENCE_THRESHOLD:
        return result

    # Fast path 2: already correctly marked — nothing to do.
    if result.executive_summary.startswith(OUT_OF_SCOPE_PREFIX):
        return result

    # Rewrite: prefix the summary and replace phases with a single placeholder.
    logger.warning(
        "output_guardrail_rewrite",
        confidence_pct=result.confidence_pct,
        original_summary_chars=len(result.executive_summary),
    )

    new_summary = (
        f"{OUT_OF_SCOPE_PREFIX} not enough information to estimate confidently. "
        f"Original model rationale: {result.executive_summary[:_MAX_ORIGINAL_LEN]}"
    )[:_MAX_SUMMARY_LEN]

    placeholder_phase = Phase(
        name="Not estimated",
        tasks=[Task(name="Not enough information to size this project", hours=0.0, cost_usd=0.0)],
        total_hours=0.0,
        total_cost_usd=0.0,
    )

    return EstimationResult(
        executive_summary=new_summary,
        phases=[placeholder_phase],
        total_hours=0.0,
        total_cost_usd=0.0,
        team_composition=[],
        duration_weeks=1.0,
        confidence_pct=result.confidence_pct,
    )

