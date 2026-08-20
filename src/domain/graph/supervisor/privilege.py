"""Minimum privilege over tools + the audit trail (Level 3).

"Each agent only accesses its own tools" is a sentence until something enforces it.
This module is the enforcement: a declarative table of what each agent may call, and a
guarded dispatcher that checks the table BEFORE executing and records every attempt —
allowed or denied — as a structlog event.

Why privilege matters here beyond security hygiene: the session's argument is that a
single decision space holding eight or twelve tools makes the model choose worse. Each
agent below sees at most ONE tool, so there is nothing to get wrong. The safety
property and the accuracy property come from the same split.

The audit shape is fixed on purpose. Every action emits an ``agent_action`` event with
the same key set, so a whole run replays from the log in order::

    docker compose logs estimator | jq -c \\
      'select(.event == "agent_action" and .estimation_id == "<id>")
       | [.step, .agent, .tool, .outcome, .result_summary]'

Denials additionally emit ``agent_privilege_denied`` at error level, so they surface
without needing to know the estimation id.
"""

from __future__ import annotations

import hashlib
import json
from time import perf_counter
from typing import Any

import structlog

from src.core.config import get_settings
from src.generation.agentic.agent_tools import ConsensusFn, RetrievalBackend, dispatch_tool

log = structlog.get_logger()

CALCULATE_TOOL = "derive_task_hours"

AGENT_PRIVILEGES: dict[str, frozenset[str]] = {
    "supervisor": frozenset(),
    "requirements_extractor": frozenset(),
    "budget_searcher": frozenset({"search_budgets"}),
    "estimate_generator": frozenset({CALCULATE_TOOL}),
    "coherence_validator": frozenset({"validate_estimate"}),
}


class PrivilegeViolation(RuntimeError):
    def __init__(self, agent: str, tool: str, allowed: frozenset[str]) -> None:
        self.agent = agent
        self.tool = tool
        self.allowed = allowed
        super().__init__(
            f"agent {agent!r} attempted tool {tool!r}; its declared privilege is "
            f"{sorted(allowed) or 'NO tools'}"
        )


def allowed_tools(agent: str) -> frozenset[str]:
    return AGENT_PRIVILEGES.get(agent, frozenset())


def assert_allowed(agent: str, tool: str) -> None:
    allowed = allowed_tools(agent)
    if tool not in allowed:
        raise PrivilegeViolation(agent, tool, allowed)


def _digest(args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _preview(args: dict[str, Any]) -> str:
    limit = get_settings().supervisor_audit_args_preview_chars
    return json.dumps(args, sort_keys=True, default=str)[:limit]


def record_model_action(
    agent: str,
    action: str,
    *,
    step: int,
    summary: str,
    estimation_id: str | None = None,
    duration_ms: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    log.info(
        "agent_action",
        estimation_id=estimation_id,
        step=step,
        agent=agent,
        tool=None,
        action=action,
        outcome="ok",
        allowed=sorted(allowed_tools(agent)),
        model=model,
        result_summary=summary[:200],
        duration_ms=duration_ms,
    )
    return {
        "step": step,
        "agent": agent,
        "action": action,
        "tool": None,
        "outcome": "ok",
        "summary": summary[:200],
        "args_digest": None,
        "duration_ms": duration_ms,
    }


async def guarded_dispatch(
    agent: str,
    tool: str,
    args: dict[str, Any],
    *,
    step: int,
    estimation_id: str | None = None,
    backend: RetrievalBackend | None = None,
    consensus_fn: ConsensusFn | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    started = perf_counter()
    digest = _digest(args)
    allowed = allowed_tools(agent)

    if tool not in allowed:
        violation = PrivilegeViolation(agent, tool, allowed)
        log.error(
            "agent_privilege_denied",
            estimation_id=estimation_id,
            step=step,
            agent=agent,
            tool=tool,
            allowed=sorted(allowed),
            args_digest=digest,
            args_preview=_preview(args),
        )
        contribution = {
            "step": step,
            "agent": agent,
            "action": f"tool:{tool}",
            "tool": tool,
            "outcome": "denied",
            "summary": str(violation),
            "args_digest": digest,
            "duration_ms": int((perf_counter() - started) * 1000),
        }
        if settings.supervisor_privilege_strict:
            raise violation
        return (
            {"ok": False, "error": "privilege_denied", "summary": str(violation)},
            contribution,
        )

    try:
        result = await dispatch_tool(tool, args, backend=backend, consensus_fn=consensus_fn)
        outcome = "ok"
    except Exception as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "error": type(exc).__name__,
            "summary": str(exc)[:200],
        }
        outcome = "error"

    duration_ms = int((perf_counter() - started) * 1000)
    summary = str(result.get("summary", ""))[:200]
    log.info(
        "agent_action",
        estimation_id=estimation_id,
        step=step,
        agent=agent,
        tool=tool,
        action=f"tool:{tool}",
        outcome=outcome,
        allowed=sorted(allowed),
        args_digest=digest,
        args_preview=_preview(args),
        result_summary=summary,
        duration_ms=duration_ms,
    )
    return result, {
        "step": step,
        "agent": agent,
        "action": f"tool:{tool}",
        "tool": tool,
        "outcome": outcome,
        "summary": summary,
        "args_digest": digest,
        "duration_ms": duration_ms,
    }
