"""LangGraph estimation graph (Session 13).

This module provides a LangGraph-based implementation of the estimation
pipeline, replacing the hand-written agent loop from Session 12.

The graph has five nodes:
1. extract_requirements - Extract requirements from transcript
2. classify_components - Group requirements into components
3. search_budgets - Find reference budgets for each component
4. generate_estimate - Generate estimation from budgets
5. validate_and_consolidate - Validate and set final status

Usage:
    from src.graph import build_graph, get_checkpointer

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)

    config = {"configurable": {"thread_id": estimation_id}}
    result = await graph.ainvoke({"transcript": transcript}, config)
"""

from src.graph.build import build_graph
from src.graph.state import BudgetMatch, Component, EstimationState

__all__ = [
    "build_graph",
    "EstimationState",
    "Component",
    "BudgetMatch",
]
