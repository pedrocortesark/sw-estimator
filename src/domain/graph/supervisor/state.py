"""The supervisor flow's shared state — Session 13's state, EXTENDED.

``SupervisorState`` SUBCLASSES ``EstimationState`` rather than redeclaring it. That is
the literal reading of "estado tipado extendido desde el de la S13", and ``TypedDict``
inheritance makes LangGraph fold the parent's reducers into the child's channel set,
so ``budget_matches`` (``operator.add``) and ``errors`` (``operator.add``) arrive
already correct — along with the ``Component``/``BudgetMatch`` shapes and the field
vocabulary of the five pre-exercise nodes this flow reorganises.

The cost, stated plainly: the Session 13 *live* channels (``structure``,
``task_hours``, ``gate1_decision``, …) come along for the ride and show up empty in
``snapshot.values``. They are ``total=False`` and no Session 14 node ever writes them,
so the cost is one unused channel each — cheaper than duplicating six field
definitions plus two reducer annotations, and invisible over HTTP because the router's
response model projects only the fields it cares about.

Two NEW accumulators carry what this session is about:

* ``agent_contributions`` — the audit trail (Level 3). One row per thing an agent did:
  a model call, a tool call, or a DENIED tool call. This is the accumulator the
  exercise asks for.
* ``routing_history`` — one row per supervisor decision, with the model's own reason.
  Routing that is not in the state is routing nobody can audit.

Both use a KEYED reducer rather than ``operator.add``, for the same reason
``merge_task_hours`` does in Session 13: ``interrupt()`` RE-EXECUTES the interrupted
node on resume, so a plain concat trail would grow a duplicate row on every human
pause. Keying by identity makes a re-emitted row replace rather than append.

Note that the keyed reducer is a SAFETY NET, not a licence: the gate still calls
``interrupt()`` before it writes anything (see ``gate.py``). Relying on the reducer to
repair a bad write order would hide the bug instead of fixing it.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Optional

from typing_extensions import TypedDict

from src.domain.graph.state import EstimationState


class AgentContribution(TypedDict, total=False):
    step: int
    agent: str
    action: str
    tool: Optional[str]
    outcome: str
    summary: str
    args_digest: Optional[str]
    duration_ms: Optional[int]


class RoutingRecord(TypedDict, total=False):
    step: int
    next_agent: str
    reason: str
    source: str
    decision_confidence: Optional[str]


def _keyed_append(
    existing: list[dict] | None,
    new: list[dict] | None,
    *,
    key: Callable[[dict], tuple],
) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for item in list(existing or []) + list(new or []):
        item_key = key(item)
        merged[item_key] = {**merged.get(item_key, {}), **item}
    return list(merged.values())


def _contribution_key(contribution: dict) -> tuple:
    return (
        contribution.get("step"),
        contribution.get("agent"),
        contribution.get("action"),
        contribution.get("args_digest"),
    )


def _routing_key(record: dict) -> tuple:
    return (record.get("step"),)


def append_contributions(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    return _keyed_append(existing, new, key=_contribution_key)


def append_routing(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    return _keyed_append(existing, new, key=_routing_key)


class SupervisorState(EstimationState, total=False):
    next_agent: Optional[str]
    route_reason: Optional[str]
    supervisor_steps: int
    routing_history: Annotated[list[RoutingRecord], append_routing]
    agent_contributions: Annotated[list[AgentContribution], append_contributions]
    component_anchors: list[dict]
    validation: Optional[dict]
    confidence: Optional[float]
    out_of_range: Optional[bool]
    grounded_components: Optional[int]
    needs_human_review: Optional[bool]
    review_reasons: list[str]
    human_decision: Optional[dict]
    proposals: Optional[list[dict]]
    divergence: Optional[dict]
    synthesis: Optional[dict]
    persist_requested: Optional[bool]
    saved: Optional[dict]


def privilege_violations(state: dict[str, Any]) -> list[dict]:
    return [
        contribution
        for contribution in (state.get("agent_contributions") or [])
        if contribution.get("outcome") == "denied"
    ]
