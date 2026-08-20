"""Failure mode #1 — a supervisor that never converges (routing loop).

SYMPTOM (live): the run bounces requirements_extractor → budget_searcher →
requirements_extractor → budget_searcher … and only stops when the step budget forces a
``finish``. ``routing_history`` shows the ping-pong and ends with a ``source == "limit"``
row.

CAUSE: cyclic return edges (every agent hands control back to the supervisor) plus a
router that may re-choose an agent that already ran. The legality guard is what forbids a
re-visit; drop it and the loop is inevitable — a scripted OR an LLM router will happily
oscillate.

FIX (str_replace on screen): flip ``guard`` on — restore the ``and not _already_ran(...)``
clause in ``_is_legal``. One clause turns the ping-pong into a convergent flow.

This is a standalone reproduction of the exact brakes in
``src/domain/graph/supervisor/supervisor.py`` (``_already_ran`` / ``_is_legal`` /
``_fallback_next`` / the step budget), boiled down so the loop is visible in a few lines.
"""

from __future__ import annotations

MAX_ROUTING_STEPS = 8

_ORDER = ["requirements_extractor", "budget_searcher"]


def _already_ran(agent: str, history: list[dict]) -> bool:
    return any(record["next_agent"] == agent for record in history)


def _inputs_ready(agent: str, history: list[dict]) -> bool:
    if agent == "requirements_extractor":
        return True
    if agent == "budget_searcher":
        return _already_ran("requirements_extractor", history)
    return False


def _is_legal(target: str, history: list[dict], *, guard: bool) -> bool:
    if target == "finish":
        return True
    if target not in _ORDER:
        return False
    if guard:
        return _inputs_ready(target, history) and not _already_ran(target, history)
    return _inputs_ready(target, history)


def _fallback_next(history: list[dict], *, guard: bool) -> str:
    for agent in _ORDER:
        if _is_legal(agent, history, guard=guard):
            return agent
    return "finish"


def run_router(
    route_script: list[str], *, guard: bool = False, max_steps: int = MAX_ROUTING_STEPS
) -> list[dict]:
    history: list[dict] = []
    for step in range(max_steps + 1):
        if step >= max_steps:
            history.append(
                {
                    "step": step,
                    "next_agent": "finish",
                    "source": "limit",
                    "reason": f"step budget of {max_steps} exhausted",
                }
            )
            break

        proposed = route_script[step % len(route_script)] if route_script else "finish"
        if _is_legal(proposed, history, guard=guard):
            target, source = proposed, "llm"
        else:
            target, source = _fallback_next(history, guard=guard), "fallback"

        history.append(
            {"step": step, "next_agent": target, "source": source, "reason": f"routed to {target}"}
        )
        if target == "finish":
            break
    return history
