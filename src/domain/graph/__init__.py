"""Session 13 — the estimation flow as an explicit LangGraph StateGraph.

Where Session 12 drove the flow with a hand-written reason→act→observe loop, this
package re-expresses it as a graph: five typed nodes wired sequentially, a shared
TypedDict state with accumulator reducers, a Postgres checkpointer for
persistence, and Logfire spans for observability.

Architecturally the graph is CONDUCTOR territory (it composes generation/rag
retrieval + foundation/llm generation), so it lives under src/domain/
beside estimation_service.py — the only layer allowed to compose generation
siblings. Nodes self-wire their dependencies through local imports, so each node
stays a pure state -> partial update function.

The external contract is unchanged: transcript in, structured estimate + status
out. The business backend is oblivious to the graph underneath.

Usage:
    from src.domain.graph import build_graph

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)

    config = {"configurable": {"thread_id": estimation_id}}
    result = await graph.ainvoke({"transcript": transcript}, config)
"""

from src.domain.graph.build import build_graph
from src.domain.graph.state import BudgetMatch, Component, EstimationState

__all__ = [
    "build_graph",
    "EstimationState",
    "Component",
    "BudgetMatch",
]
