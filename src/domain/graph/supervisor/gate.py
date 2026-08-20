"""Human-in-the-loop: pause when the estimate is not trustworthy enough to ship.

Session 13's gates pause UNCONDITIONALLY at fixed points in the flow. This one pauses
on a SIGNAL: the graph runs unattended when the numbers are well grounded, and stops
for a person exactly when they are not. That difference is the whole point — a human
gate that always fires is a form, not a control.

The three trigger conditions are the ones the exercise names:

1. confidence below the configured threshold,
2. the estimate falls outside the range its historical references imply,
3. the transcript has essentially no precedent in the budget corpus.

The split of responsibility matters and is deliberate: ``coherence_validator`` writes
FACTS (``confidence``, ``out_of_range``, ``grounded_components``); this module owns the
VERDICT. That is what lets the threshold move via configuration — or a second trigger
be added — without touching the validator.
"""

from __future__ import annotations

import logfire
import structlog
from langgraph.types import interrupt

from src.core.config import Settings, get_settings
from src.domain.graph.supervisor.state import SupervisorState

log = structlog.get_logger()

HOURS_PER_DAY = 8.0


def review_reasons(state: SupervisorState, settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    reasons: list[str] = []

    confidence = state.get("confidence")
    if confidence is not None and confidence < settings.supervisor_confidence_threshold:
        reasons.append(
            f"confidence {confidence:.2f} is below the "
            f"{settings.supervisor_confidence_threshold:.2f} threshold"
        )

    if state.get("out_of_range"):
        reasons.append(
            "at least one component falls outside the plausible range implied by its "
            "historical references"
        )

    total = len(state.get("components") or [])
    grounded = state.get("grounded_components") or 0
    if total and (grounded / total) < settings.supervisor_min_grounded_ratio:
        reasons.append(
            f"only {grounded}/{total} components have any precedent in the historical budgets"
        )

    if state.get("persist_requested"):
        reasons.append(
            "an irreversible save_estimate is queued; the human pause authorises the write"
        )

    return reasons


def needs_human_review(state: SupervisorState, settings: Settings | None = None) -> bool:
    return bool(review_reasons(state, settings))


def _apply_decision(state: SupervisorState, decision: dict) -> tuple[dict, str]:
    estimate = dict(state.get("estimate") or {})
    action = (decision or {}).get("decision") or (decision or {}).get("action") or "approve"

    if action == "reject":
        return estimate, "rejected"

    if action == "adjust":
        overrides = (decision or {}).get("estimate_overrides") or {}
        estimate = {**estimate, **overrides}
        components = estimate.get("components") or []
        if components:
            estimate["total_engineer_days"] = sum(
                int(c.get("engineer_days") or 0) for c in components
            )

    return estimate, "validated"


async def human_review_gate(state: SupervisorState) -> dict:
    reasons = review_reasons(state)

    if not reasons:
        with logfire.span("gate: human_review (auto-approved)"):
            log.info(
                "human_review_gate_skipped",
                confidence=state.get("confidence"),
                status=state.get("status"),
            )
            return {"needs_human_review": False, "review_reasons": []}

    settings = get_settings()
    decision = interrupt(
        {
            "gate": "low_confidence_review",
            "estimation_id": state.get("estimation_id"),
            "reasons": reasons,
            "confidence": state.get("confidence"),
            "threshold": settings.supervisor_confidence_threshold,
            "estimate": state.get("estimate"),
            "validation": state.get("validation"),
            "routing_history": state.get("routing_history") or [],
        }
    )

    with logfire.span("gate: human_review"):
        decision = decision or {}
        estimate, status = _apply_decision(state, decision)
        action = decision.get("decision") or decision.get("action") or "approve"
        log.info(
            "human_review_gate_resumed",
            action=action,
            status=status,
            reasons=len(reasons),
        )
        return {
            "estimate": estimate,
            "status": status,
            "human_decision": decision,
            "needs_human_review": True,
            "review_reasons": reasons,
            "agent_contributions": [
                {
                    "step": int(state.get("supervisor_steps") or 0),
                    "agent": "human",
                    "action": "review_decision",
                    "tool": None,
                    "outcome": "ok",
                    "summary": f"human {action}: {decision.get('note') or '—'}",
                    "args_digest": None,
                    "duration_ms": None,
                }
            ],
        }
