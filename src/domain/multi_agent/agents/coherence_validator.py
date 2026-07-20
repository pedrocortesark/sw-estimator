"""Coherence validator agent.

Validates the estimate for coherence and consistency.
This agent has access ONLY to the validate_estimate tool.
"""

from __future__ import annotations

import logfire
import structlog

from src.domain.multi_agent.state import BudgetMatch, EstimationState, Validation

log = structlog.get_logger()

# References are historical engineer-HOURS; the estimate is in engineer-DAYS.
HOURS_PER_DAY = 8.0


def _validate_estimate(estimate: dict, matches: list[BudgetMatch]) -> Validation:
    """Deterministic validation of the estimate against its references.

    Checks:
    1. Each component's engineer-days against the plausible range from references
    2. Total engineer-days matches the sum of components
    3. Total is positive
    4. Confidence is reasonable
    """
    issues: list[str] = []
    components = estimate.get("components") or []
    component_sum = 0.0

    for component in components:
        name = component.get("name", "?")
        days = component.get("engineer_days")
        refs_hours = _references_for(name, matches)

        if days is None:
            if not refs_hours:
                issues.append(f"{name!r} has no historical reference (unbudgeted).")
            continue

        component_sum += days

        if not refs_hours:
            issues.append(f"{name!r} has no historical reference (unbudgeted).")
            continue

        refs_days = [h / HOURS_PER_DAY for h in refs_hours]
        low = min(refs_days) * 0.5
        high = max(refs_days) * 2.0

        if not (low <= days <= high):
            issues.append(
                f"{name!r} estimate {days}d is outside the plausible range "
                f"[{round(low, 1)}, {round(high, 1)}]d implied by its references."
            )

    total = estimate.get("total_engineer_days")
    if total is None:
        issues.append("Total engineer-days is missing.")
    else:
        if total <= 0:
            issues.append("Total engineer-days is non-positive.")
        if abs(component_sum - total) > 0.5:
            issues.append(
                f"Total {total}d does not match the sum of components ({round(component_sum, 1)}d)."
            )

    # Calculate validation confidence
    confidence = _calculate_validation_confidence(issues, matches)

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "confidence": confidence,
    }


def _references_for(component: str, matches: list[BudgetMatch]) -> list[float]:
    """Get the historical reference hours for a component."""
    return [m["amount"] for m in matches if m["component"] == component]


def _calculate_validation_confidence(issues: list[str], matches: list[BudgetMatch]) -> float:
    """Calculate a validation confidence score."""
    if not matches:
        return 0.0

    # Start with high confidence and deduct for issues
    base_confidence = 1.0

    # Deduct for each issue
    issue_penalty = len(issues) * 0.15
    confidence = max(0.0, base_confidence - issue_penalty)

    # Bonus for having many references
    reference_bonus = min(0.2, len(matches) * 0.05)
    confidence = min(1.0, confidence + reference_bonus)

    return round(confidence, 3)


async def coherence_validator(state: EstimationState) -> dict:
    """Validate the estimate for coherence (validate_estimate tool only)."""
    with logfire.span("agent: coherence_validator"):
        estimate = state.get("estimate") or {}
        matches = state.get("budget_matches") or []

        validation = _validate_estimate(estimate, matches)

        log.info(
            "agent_coherence_validator_done",
            is_valid=validation["is_valid"],
            issues=len(validation["issues"]),
            confidence=validation["confidence"],
        )

        # Audit log entry
        audit_entry = {
            "agent": "coherence_validator",
            "tool": "validate_estimate",
            "input_summary": f"estimate + {len(matches)} matches",
            "output_summary": f"valid={validation['is_valid']}, {len(validation['issues'])} issues",
        }

        return {
            "validation": validation,
            "agent_actions": [audit_entry],
        }
