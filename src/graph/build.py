"""Build the LangGraph estimation graph (Session 13).

The graph has five nodes connected in sequence:
START → extract_requirements → classify_components → search_budgets
      → generate_estimate → validate_and_consolidate → END

In the live session, we'll add:
- Parallel execution of search_budgets (Send API)
- Conditional edge from validate_and_consolidate
- Human intervention with interrupt()
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    classify_components,
    extract_requirements,
    generate_estimate,
    search_budgets,
    validate_and_consolidate,
)
from src.graph.state import EstimationState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the estimation graph.

    Args:
        checkpointer: Optional checkpointer for persistence.
                     If None, the graph runs without persistence.

    Returns:
        Compiled graph ready for execution.
    """
    builder = StateGraph(EstimationState)

    # Add nodes
    builder.add_node("extract_requirements", extract_requirements)
    builder.add_node("classify_components", classify_components)
    builder.add_node("search_budgets", search_budgets)
    builder.add_node("generate_estimate", generate_estimate)
    builder.add_node("validate_and_consolidate", validate_and_consolidate)

    # Add edges (sequential for now)
    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")
    builder.add_edge("validate_and_consolidate", END)

    # Compile with optional checkpointer
    return builder.compile(checkpointer=checkpointer)
