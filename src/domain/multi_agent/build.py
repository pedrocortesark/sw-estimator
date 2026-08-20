"""Build the Session 14 multi-agent graph.

Wires the supervisor and specialist agents into a StateGraph with explicit routing.
The supervisor decides the next agent based on state, and each agent has minimum
tool privileges.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.domain.multi_agent.state import EstimationState
from src.domain.multi_agent.supervisor import supervisor
from src.domain.multi_agent.agents import (
    requirements_extractor,
    budget_searcher,
    estimate_generator,
    coherence_validator,
)
from src.domain.multi_agent.human_review_gate import human_review_gate
from src.domain.multi_agent.finalize import finalize

log = structlog.get_logger()


def build_multi_agent_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the multi-agent estimation graph.

    The graph topology is supervisor/workers:
    - Supervisor routes to the next specialist based on state
    - Each specialist agent has minimum tool privileges
    - Human review gate pauses for intervention when confidence is low
    - Finalize consolidates the result

    Flow:
    START → supervisor → requirements_extractor → supervisor
                        → budget_searcher → supervisor
                        → estimate_generator → supervisor
                        → coherence_validator → supervisor
                        → human_review_gate (if low confidence) → supervisor
                        → finalize → END
    """
    builder = StateGraph(EstimationState)

    # Add the supervisor as the central router
    builder.add_node("supervisor", supervisor)

    # Add specialist agents
    builder.add_node("requirements_extractor", requirements_extractor)
    builder.add_node("budget_searcher", budget_searcher)
    builder.add_node("estimate_generator", estimate_generator)
    builder.add_node("coherence_validator", coherence_validator)

    # Add human-in-the-loop gate
    builder.add_node("human_review_gate", human_review_gate)

    # Add finalization node
    builder.add_node("finalize", finalize)

    # Wire the graph: all agents route back to supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor routes to specialists (via Command(goto=...))
    # No static edges needed - supervisor uses Command for dynamic routing

    # Specialists route back to supervisor for next decision
    builder.add_edge("requirements_extractor", "supervisor")
    builder.add_edge("budget_searcher", "supervisor")
    builder.add_edge("estimate_generator", "supervisor")
    builder.add_edge("coherence_validator", "supervisor")

    # Human review gate routes back to supervisor after decision
    builder.add_edge("human_review_gate", "supervisor")

    # Finalize goes to END
    builder.add_edge("finalize", END)

    # Compile with optional checkpointer for human-in-the-loop
    graph = builder.compile(checkpointer=checkpointer)

    log.info("multi_agent_graph_built", checkpointer=checkpointer is not None)

    return graph
