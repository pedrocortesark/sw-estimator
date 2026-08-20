"""Wire and compile the supervisor graph (Session 14).

The topology is a STAR, which is what a supervisor system should look like when you
draw it — not a line with more boxes::

            START
              │
              ▼
       ┌─▶ supervisor ──Command(goto)──┬──▶ requirements_extractor ──┐
       │                               ├──▶ budget_searcher ─────────┤
       │                               ├──▶ estimate_generator ──────┤
       └──────── static return ────────┼──▶ coherence_validator ─────┘
         edges                         │
                                       └──▶ human_review_gate ──▶ END

* **Dynamic edges** — the five ``supervisor → {4 agents, gate}`` hand-overs. These do
  not exist in the graph definition at all: ``Command(goto=...)`` draws them at
  runtime. That is the point of the session.
* **Static edges** — ``START → supervisor``, the four ``agent → supervisor`` return
  edges, and ``human_review_gate → END``. Six in total.

``END`` is reached through exactly one edge, whether the gate paused for a human or
fell straight through. One exit is much easier to reason about than two.

Note the gate returns a plain dict rather than a ``Command``. Mixing ``interrupt()``
with a ``Command`` return in one node is legal, but the resume path re-executes and
would have to reconstruct the same ``Command`` — one more thing to get wrong, for no
benefit when there is only one destination.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph

from src.domain.graph.supervisor.agents import (
    budget_searcher,
    coherence_validator,
    competitive_estimate_generator,
    estimate_generator,
    persistence_agent,
    requirements_extractor,
)
from src.domain.graph.supervisor.gate import human_review_gate
from src.domain.graph.supervisor.sandbox import verify_tool_grants
from src.domain.graph.supervisor.state import SupervisorState
from src.domain.graph.supervisor.supervisor import supervisor

log = structlog.get_logger()

AGENT_NODES = {
    "requirements_extractor": requirements_extractor,
    "budget_searcher": budget_searcher,
    "estimate_generator": estimate_generator,
    "coherence_validator": coherence_validator,
}


def build_supervisor_graph(
    checkpointer=None, *, competitive: bool = False, sandboxed: bool = False
):
    verify_tool_grants()

    agent_nodes = dict(AGENT_NODES)
    if competitive:
        agent_nodes["estimate_generator"] = competitive_estimate_generator

    builder = StateGraph(SupervisorState)

    builder.add_node(
        "supervisor",
        supervisor,
        destinations=(*AGENT_NODES, "human_review_gate"),
    )
    for name, fn in agent_nodes.items():
        builder.add_node(name, fn)
    builder.add_node("human_review_gate", human_review_gate)

    builder.add_edge(START, "supervisor")
    for name in agent_nodes:
        builder.add_edge(name, "supervisor")

    if sandboxed:
        builder.add_node("persistence_agent", persistence_agent)
        builder.add_edge("human_review_gate", "persistence_agent")
        builder.add_edge("persistence_agent", END)
    else:
        builder.add_edge("human_review_gate", END)

    return builder.compile(checkpointer=checkpointer)
