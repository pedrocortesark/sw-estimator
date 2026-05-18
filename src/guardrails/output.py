"""Output guardrails — validate and sanitise the LLM response."""

from __future__ import annotations

from src.schemas.estimation import EstimationResult


def enforce_scope_response(result: EstimationResult) -> EstimationResult:
    """Enforce business rules on the structured LLM response.

    Returns the result unchanged until rules are implemented.
    """
    return result
