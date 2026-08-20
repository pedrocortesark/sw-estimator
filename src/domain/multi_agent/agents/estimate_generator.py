"""Estimate generator agent.

Generates the final estimate from budget matches.
This agent has access ONLY to the calculate_estimate tool.
"""

from __future__ import annotations

import asyncio

import logfire
import structlog

from src.core.config import get_settings
from src.domain.graph.schemas import ConsolidatedEstimate
from src.domain.multi_agent.state import BudgetMatch, EstimationState

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a senior software estimator. Consolidate the historical budget references "
    "into a single structured estimate expressed in engineer-days.\n"
    "Method:\n"
    "1. For each component, convert every reference from hours to engineer-days by "
    "DIVIDING by 8 (8 working hours per day).\n"
    "2. Put the ROUNDED MEDIAN of those per-reference day values in the component's "
    "`engineer_days` field as an integer — the field itself, not just the rationale.\n"
    "3. If a component has NO references, set its `engineer_days` to null and say so "
    "in its rationale.\n"
    "4. Set total_engineer_days to the exact SUM of the grounded components' "
    "engineer_days (treat null as 0 in the sum).\n"
    "5. Set confidence based on how well the references ground the estimate."
)


def _references_digest(matches: list[BudgetMatch]) -> str:
    """Create a compact digest of budget matches for the LLM."""
    if not matches:
        return "No historical budget references found."

    lines = ["Historical budget references (recorded in engineer-hours):"]
    for match in matches:
        component = match["component"]
        amount = match["amount"]
        budget_id = match.get("reference_budget_id", "unknown")
        distance = match["distance"]
        lines.append(
            f"- {component}: {amount:.0f}h (budget_id={budget_id}, distance={distance:.3f})"
        )
    return "\n".join(lines)


async def estimate_generator(state: EstimationState) -> dict:
    """Generate estimate from budget matches (calculate_estimate tool only)."""
    with logfire.span("agent: estimate_generator"):
        settings = get_settings()
        from src.dependencies import get_llm_wrapper

        wrapper = get_llm_wrapper()
        matches = state.get("budget_matches") or []
        user_message = _references_digest(matches)

        result, _meta = await wrapper.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
            response_model=ConsolidatedEstimate,
            model_override=settings.graph_generation_model,
        )

        estimate = result.model_dump()
        confidence = _calculate_confidence(matches, result)

        log.info(
            "agent_estimate_generator_done",
            total_engineer_days=result.total_engineer_days,
            confidence=confidence,
        )

        # Audit log entry
        audit_entry = {
            "agent": "estimate_generator",
            "tool": "calculate_estimate",
            "input_summary": f"{len(matches)} budget matches",
            "output_summary": f"total={result.total_engineer_days}d, confidence={confidence:.2f}",
        }

        return {
            "estimate": estimate,
            "confidence": confidence,
            "agent_actions": [audit_entry],
        }


def _calculate_confidence(matches: list[BudgetMatch], result: ConsolidatedEstimate) -> float:
    """Calculate a confidence score based on reference quality."""
    if not matches:
        return 0.0

    # Base confidence on number of references and their distances
    avg_distance = sum(m["distance"] for m in matches) / len(matches)

    # Closer references = higher confidence (distance is 0-1, lower is better)
    distance_confidence = max(0.0, 1.0 - avg_distance)

    # More references = higher confidence (diminishing returns after 5)
    reference_count_confidence = min(1.0, len(matches) / 5.0)

    # Weighted combination
    confidence = (distance_confidence * 0.6) + (reference_count_confidence * 0.4)

    # Penalize if total is None (no grounded components)
    if result.total_engineer_days is None:
        confidence *= 0.5

    return round(confidence, 3)
