"""Human review gate for Session 14 multi-agent system.

Pauses the graph when the estimate confidence is below threshold,
waiting for human intervention (approve/adjust/reject).
"""

from __future__ import annotations

import logfire
import structlog
from langgraph.types import Command, interrupt

from src.core.config import get_settings
from src.domain.multi_agent.state import EstimationState

log = structlog.get_logger()


async def human_review_gate(state: EstimationState) -> Command:
    """Pause for human review when confidence is low.

    If confidence is above threshold, skip the gate.
    Otherwise, pause with interrupt() and wait for human decision.

    Human decision shape:
    {
        "action": "approve" | "adjust" | "reject",
        "adjusted_estimate": {...} | None,  # if action == "adjust"
        "rationale": "..."
    }
    """
    with logfire.span("agent: human_review_gate"):
        settings = get_settings()
        confidence_threshold = getattr(settings, "confidence_threshold", 0.7)
        confidence = state.get("confidence")

        # If confidence is acceptable, skip the gate
        if confidence is not None and confidence >= confidence_threshold:
            log.info(
                "human_review_gate_skip",
                confidence=confidence,
                threshold=confidence_threshold,
            )
            return Command(goto="finalize")

        # Pause for human review
        log.info(
            "human_review_gate_pause",
            confidence=confidence,
            threshold=confidence_threshold,
            estimate=state.get("estimate"),
        )

        # interrupt() pauses the graph and persists state
        # The graph will resume when Command(resume=...) is called
        decision = interrupt(
            {
                "reason": "low_confidence_estimate",
                "estimate": state.get("estimate"),
                "confidence": confidence,
                "validation": state.get("validation"),
                "message": "This estimate requires human review due to low confidence.",
            }
        )

        # Resume with human decision
        log.info(
            "human_review_gate_resume",
            decision=decision,
        )

        # Audit log entry
        audit_entry = {
            "agent": "human_review_gate",
            "tool": None,
            "input_summary": f"confidence={confidence:.2f}",
            "output_summary": f"human decision: {decision.get('action', 'unknown')}",
        }

        return Command(
            goto="finalize",
            update={
                "human_decision": decision,
                "agent_actions": [audit_entry],
            },
        )
