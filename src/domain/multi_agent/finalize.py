"""Finalize node for Session 14 multi-agent system.

Consolidates the final estimate and status after all agents have run
and any human review has been completed.
"""

from __future__ import annotations

import logfire
import structlog

from src.domain.multi_agent.state import EstimationState

log = structlog.get_logger()


def finalize(state: EstimationState) -> dict:
    """Consolidate the final estimate and status.

    Applies human adjustments if any, and sets the final status.
    """
    with logfire.span("agent: finalize"):
        estimate = state.get("estimate") or {}
        validation = state.get("validation") or {}
        human_decision = state.get("human_decision")

        # Apply human adjustments if any
        if human_decision:
            action = human_decision.get("action")
            if action == "adjust":
                adjusted = human_decision.get("adjusted_estimate")
                if adjusted:
                    estimate = {**estimate, **adjusted}
                    log.info("finalize_applied_human_adjustment")
            elif action == "reject":
                estimate["rejected"] = True
                estimate["rejection_rationale"] = human_decision.get("rationale", "")
                log.info("finalize_estimate_rejected_by_human")

        # Determine final status
        if estimate.get("rejected"):
            status = "rejected"
        elif validation.get("is_valid", False):
            status = "validated"
        else:
            status = "needs_review"

        log.info(
            "finalize_done",
            status=status,
            has_human_decision=human_decision is not None,
        )

        # Audit log entry
        audit_entry = {
            "agent": "finalize",
            "tool": None,
            "input_summary": f"estimate + validation + human_decision",
            "output_summary": f"final status: {status}",
        }

        return {
            "estimate": estimate,
            "status": status,
            "awaiting_review": False,
            "agent_actions": [audit_entry],
        }
