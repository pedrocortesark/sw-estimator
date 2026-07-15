"""Typed state for the LangGraph estimation graph (Session 13).

The state flows through five nodes:
1. extract_requirements - Extract requirements from transcript
2. classify_components - Group requirements into components
3. search_budgets - Find reference budgets for each component
4. generate_estimate - Generate estimation from budgets
5. validate_and_consolidate - Validate and set final status

Accumulator fields (with reducers) grow as the graph progresses.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

import operator


class Component(TypedDict):
    """A project component with its category."""

    name: str
    category: str
    requirements: list[str]


class BudgetMatch(TypedDict):
    """A reference budget match for a component."""

    component: str
    budget_id: str
    amount: float
    description: str
    relevance_score: float


class EstimationState(TypedDict):
    """State that flows through the estimation graph.

    Fields with Annotated[..., operator.add] are accumulators that grow
    as nodes append to them.
    """

    # Input
    transcript: str

    # Intermediate outputs
    requirements: list[str]
    components: list[Component]

    # Accumulator: grows as each component is searched
    budget_matches: Annotated[list[BudgetMatch], operator.add]

    # Final outputs
    estimate: Optional[dict[str, Any]]
    status: Optional[str]  # "validated" | "needs_review"

    # Accumulator: errors collected during execution
    errors: Annotated[list[str], operator.add]

    # Metadata
    estimation_id: str
