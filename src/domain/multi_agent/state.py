"""Typed state for the Session 14 multi-agent system.

Extends the Session 13 state with reducer accumulators for agent contributions
and fields for human-in-the-loop intervention.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict


class BudgetMatch(TypedDict):
    """A historical reference budget retrieved for a component."""

    component: str
    reference_budget_id: Optional[str]
    amount: float
    distance: float


class Validation(TypedDict):
    """Validation result from the coherence validator."""

    is_valid: bool
    issues: list[str]
    confidence: float


class AgentAction(TypedDict):
    """Audit log entry for an agent action."""

    agent: str
    tool: Optional[str]
    input_summary: str
    output_summary: str


class EstimationState(TypedDict, total=False):
    """The state threaded through the multi-agent graph.

    Reducer accumulators:
    - budget_matches: grows as budget_searcher finds references
    - agent_actions: audit log of all agent actions

    Human-in-the-loop:
    - human_decision: the human's decision when the graph pauses
    - awaiting_review: flag indicating the graph is paused for human review
    """

    # Input
    transcript: str
    estimation_id: str

    # Agent contributions (accumulators)
    requirements: list[str]
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    estimate: Optional[dict]
    validation: Optional[Validation]
    confidence: Optional[float]

    # Human-in-the-loop
    human_decision: Optional[dict]
    awaiting_review: bool

    # Audit log (accumulator)
    agent_actions: Annotated[list[AgentAction], operator.add]

    # Output
    status: str  # "validated" | "needs_review" | "awaiting_human_review"
