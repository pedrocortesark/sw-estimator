"""The hand-built supervisor: the node that decides WHO acts next.

This is the frontier the session is about. In Session 13 the order was written in
``build.py`` at authoring time; here the MODEL reads the state and picks the next
specialist, and the graph obeys. That — not the node count, not the naming — is what
separates a workflow from an agentic system.

Built by hand with ``StateGraph`` + ``Command`` rather than ``create_supervisor``,
because every routing decision has to be visible, overridable and logged, and that
means the decision has to be OUR code. The model chooses; three deterministic brakes
decide whether the choice survives:

* **A step budget** (``supervisor_max_steps``). Cyclic return edges plus an LLM router
  is exactly how a graph ping-pongs forever. The counter is the hard ceiling on both
  loops and spend.
* **A legality guard** (``_is_legal``). Rejects a target whose inputs do not exist yet
  (searching budgets before components are classified) and re-visits of an agent whose
  output is already in the state.
* **A deterministic fallback** (``_fallback_next``). Used when the model raises,
  returns an illegal target, or is unavailable. The graph terminates correctly even
  with the LLM completely broken — which is also what keeps the tests network-free.
"""

from __future__ import annotations

import asyncio

import logfire
import structlog
from langgraph.types import Command

from src.core.config import get_settings
from src.domain.graph.schemas import SupervisorDecision
from src.domain.graph.supervisor.state import SupervisorState

log = structlog.get_logger()

_ORDER = [
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
]

_SYSTEM_PROMPT = """\
You are the SUPERVISOR of a software-estimation multi-agent system. You do not do any \
estimation work yourself: you read the current state and decide which specialist acts next.

The specialists and what each one produces:

- requirements_extractor: reads the raw transcript and produces `requirements` and \
`components`. It holds no tools. Needs: the transcript.
- budget_searcher: searches the historical budget corpus and produces `budget_matches` \
(reference hours per component). Needs: `components`.
- estimate_generator: turns references into a consolidated `estimate` in engineer-days. \
Needs: `budget_matches`.
- coherence_validator: runs guardrails over the estimate and produces `validation` plus a \
confidence signal. Needs: `estimate`.

Rules:
- Choose the ONE agent that can make progress right now, given what the state already holds.
- Never choose an agent whose inputs do not exist yet.
- Never re-run an agent whose output is already in the state.
- Choose "finish" once the estimate has been produced AND validated.
- Explain your choice in one line: what the state has, what is missing, why this agent.
"""


def _summarise(state: SupervisorState) -> str:
    components = state.get("components") or []
    matches = state.get("budget_matches") or []
    estimate = state.get("estimate")
    done = [record.get("next_agent") for record in (state.get("routing_history") or [])]

    component_names = ", ".join(c["name"] for c in components[:6]) or "—"
    grounded = len({m["component"] for m in matches})
    return "\n".join(
        [
            "Estimation state so far:",
            f"- transcript: {len(state.get('transcript') or '')} characters",
            f"- requirements: {len(state.get('requirements') or [])} extracted",
            f"- components: {len(components)} classified ({component_names})",
            f"- budget_matches: {len(matches)} references covering "
            f"{grounded}/{len(components)} components",
            f"- estimate: {'produced' if estimate else 'not produced yet'}",
            f"- validation: {'run' if state.get('validation') else 'not run yet'}",
            f"- agents already dispatched: {', '.join(done) or 'none'}",
            "",
            "Which agent must act next?",
        ]
    )


def _already_ran(agent: str, state: SupervisorState) -> bool:
    return any(record.get("next_agent") == agent for record in (state.get("routing_history") or []))


def _inputs_ready(agent: str, state: SupervisorState) -> bool:
    if agent == "requirements_extractor":
        return bool(state.get("transcript"))
    if agent == "budget_searcher":
        return bool(state.get("components"))
    if agent == "estimate_generator":
        return bool(state.get("components")) and _already_ran("budget_searcher", state)
    if agent == "coherence_validator":
        return bool(state.get("estimate"))
    return False


def _is_legal(target: str, state: SupervisorState) -> bool:
    if target == "finish":
        return True
    if target not in _ORDER:
        return False
    return _inputs_ready(target, state) and not _already_ran(target, state)


def _fallback_next(state: SupervisorState) -> str:
    for agent in _ORDER:
        if _is_legal(agent, state):
            return agent
    return "finish"


async def supervisor(
    state: SupervisorState,
) -> Command:
    settings = get_settings()
    step = int(state.get("supervisor_steps") or 0)

    if step >= settings.supervisor_max_steps:
        target, reason, source, decision_confidence = (
            "finish",
            f"step budget of {settings.supervisor_max_steps} exhausted; finishing",
            "limit",
            None,
        )
        log.warning("supervisor_step_budget_exhausted", step=step)
    else:
        with logfire.span("supervisor: route"):
            from src.dependencies import get_llm_wrapper

            decision_confidence = None
            try:
                decision, _meta = await asyncio.to_thread(
                    get_llm_wrapper().complete_structured,
                    system_prompt=_SYSTEM_PROMPT,
                    user_message=_summarise(state),
                    response_model=SupervisorDecision,
                    model_override=settings.supervisor_router_model,
                )
                target, reason, source = decision.next_agent, decision.reason, "llm"
                decision_confidence = decision.confidence
            except Exception as exc:  # noqa: BLE001
                target, reason, source = (
                    _fallback_next(state),
                    f"router unavailable ({type(exc).__name__}); "
                    "fell back to the dependency ladder",
                    "fallback",
                )
                log.error("supervisor_route_failed", error=str(exc)[:200], step=step)

            if not _is_legal(target, state):
                overridden, target = target, _fallback_next(state)
                log.warning(
                    "supervisor_route_overridden",
                    step=step,
                    proposed=overridden,
                    chosen=target,
                )
                reason = (
                    f"router proposed {overridden!r}, which is not legal in this state; "
                    f"overridden to {target!r}"
                )
                source = "fallback"

    goto = "human_review_gate" if target == "finish" else target
    log.info(
        "supervisor_route",
        step=step,
        next_agent=target,
        goto=goto,
        source=source,
        reason=reason[:200],
    )
    return Command(
        goto=goto,
        update={
            "next_agent": target,
            "route_reason": reason,
            "supervisor_steps": step + 1,
            "routing_history": [
                {
                    "step": step,
                    "next_agent": target,
                    "reason": reason,
                    "source": source,
                    "decision_confidence": decision_confidence,
                }
            ],
        },
    )
