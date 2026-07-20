"""Supervisor for the Session 14 multi-agent system.

The supervisor is a hand-built routing node that decides the next specialist agent
based on the current state. It has NO tool privileges - it only routes.

Every routing decision is made explicit via Command(goto=...) so it appears in traces.
"""

from __future__ import annotations

from typing import Literal

import logfire
import structlog
from langgraph.types import Command

from src.core.config import get_settings
from src.domain.multi_agent.state import EstimationState

log = structlog.get_logger()

# Valid routing targets
AgentName = Literal[
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
    "human_review_gate",
    "finalize",
    "__end__",
]


def supervisor(
    state: EstimationState,
) -> Command[AgentName]:
    """Decide the next specialist agent based on the current state.

    The supervisor has NO tool privileges - it only routes.
    Every routing decision is explicit via Command(goto=...) for traceability.

    Routing logic:
    1. If no requirements yet → requirements_extractor
    2. If requirements but budget_searcher hasn't run → budget_searcher
    3. If budget_searcher ran (even with 0 matches) → estimate_generator
    4. If estimate but no validation → coherence_validator
    5. If validation done and confidence < threshold → human_review_gate
    6. If validation done and confidence >= threshold → finalize
    7. If human decision made → finalize
    """
    with logfire.span("agent: supervisor"):
        settings = get_settings()
        confidence_threshold = getattr(settings, "confidence_threshold", 0.7)

        # Determine current stage based on state
        has_requirements = bool(state.get("requirements"))
        has_budget_matches = bool(state.get("budget_matches"))
        has_estimate = state.get("estimate") is not None
        has_validation = state.get("validation") is not None
        has_human_decision = state.get("human_decision") is not None
        confidence = state.get("confidence")
        
        # Check if budget_searcher has already run by looking at agent_actions
        agent_actions = state.get("agent_actions", [])
        budget_searcher_ran = any(
            action.get("agent") == "budget_searcher" 
            for action in agent_actions
        )

        # Routing decision
        if not has_requirements:
            next_agent: AgentName = "requirements_extractor"
            reason = "no requirements extracted yet"
        elif has_requirements and not budget_searcher_ran:
            next_agent = "budget_searcher"
            reason = "budget_searcher hasn't run yet"
        elif budget_searcher_ran and not has_estimate:
            next_agent = "estimate_generator"
            reason = "budget_searcher completed, generating estimate"
        elif has_estimate and not has_validation:
            next_agent = "coherence_validator"
            reason = "estimate not validated yet"
        elif has_human_decision:
            next_agent = "finalize"
            reason = "human decision received, finalizing"
        elif confidence is not None and confidence < confidence_threshold:
            next_agent = "human_review_gate"
            reason = f"confidence {confidence:.2f} < threshold {confidence_threshold:.2f}"
        else:
            next_agent = "finalize"
            reason = f"validation complete, confidence {confidence:.2f} >= threshold"

        log.info(
            "supervisor_routing",
            next_agent=next_agent,
            reason=reason,
            has_requirements=has_requirements,
            has_budget_matches=has_budget_matches,
            budget_searcher_ran=budget_searcher_ran,
            has_estimate=has_estimate,
            has_validation=has_validation,
            confidence=confidence,
        )

        # Audit log entry
        audit_entry = {
            "agent": "supervisor",
            "tool": None,  # Supervisor has no tools
            "input_summary": f"stage: {reason}",
            "output_summary": f"routing to {next_agent}",
        }

        return Command(
            goto=next_agent,
            update={"agent_actions": [audit_entry]},
        )
