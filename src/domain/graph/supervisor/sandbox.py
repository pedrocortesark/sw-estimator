"""Agent-level sandboxing (Session 14 LIVE): the three containment layers for WRITES.

The reference solution already enforces MINIMUM PRIVILEGE over read tools
(``privilege.AGENT_PRIVILEGES`` + ``guarded_dispatch``). This module extends that to the
one place it really bites — an agent that can WRITE — with three deterministic layers,
all in plain Python. There is NO process isolation, OS sandbox or secret management here:
that is Session 15. "Sandboxing" at this layer means privilege + argument validation +
audit, which is what actually stops an agent doing the wrong irreversible thing.

Layer 1 — GRANTS with a RISK dimension. ``AGENT_TOOL_GRANTS`` extends the privilege
    table with the write-capable ``persistence_agent``, and ``TOOL_RISK`` classifies
    every tool READ / WRITE / IRREVERSIBLE. ``verify_tool_grants()`` runs at graph-build
    time and FAILS THE STARTUP if an agent is granted a tool with no declared risk or no
    implementation — a misconfiguration can never reach runtime.

Layer 2 — ARGUMENT VALIDATION before execution. ``guard_action`` is pure, deterministic
    code: it checks the allowlist, validates the arguments, and — the load-bearing check
    for a multi-tenant system — verifies the action's ``estimation_id`` matches the run
    in progress, so one run can never write another run's estimate. An IRREVERSIBLE tool
    additionally requires a recorded human approval; without it the write is refused and
    the flow is expected to route through the human gate (see ``gate.review_reasons``).

Layer 3 — AUDIT of every intent WITH EFFECTS, including the denied ones.
    ``execute_guarded`` logs a structlog event for allowed, denied AND deferred writes,
    with the ``estimation_id`` and a REDACTED argument preview (the full SHA-256 digest
    is always logged, so a call's identity is provable without dumping sensitive data).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import structlog

from src.core.config import get_settings
from src.domain.graph.supervisor.privilege import AGENT_PRIVILEGES, _digest

log = structlog.get_logger()

SAVE_ESTIMATE_TOOL = "save_estimate"


class ToolRisk(enum.StrEnum):
    READ = "read"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


TOOL_RISK: dict[str, ToolRisk] = {
    "search_budgets": ToolRisk.READ,
    "derive_task_hours": ToolRisk.READ,
    "validate_estimate": ToolRisk.READ,
    SAVE_ESTIMATE_TOOL: ToolRisk.IRREVERSIBLE,
}


AGENT_TOOL_GRANTS: dict[str, frozenset[str]] = {
    **AGENT_PRIVILEGES,
    "persistence_agent": frozenset({SAVE_ESTIMATE_TOOL}),
}


class GrantVerificationError(RuntimeError):
    pass


def verify_tool_grants(known_tools: set[str] | None = None) -> None:
    known = known_tools if known_tools is not None else set(TOOL_RISK)
    for agent, tools in AGENT_TOOL_GRANTS.items():
        for tool in tools:
            if tool not in TOOL_RISK:
                raise GrantVerificationError(
                    f"agent {agent!r} is granted tool {tool!r}, which has no declared "
                    f"ToolRisk — every granted tool must be classified in TOOL_RISK"
                )
            if tool not in known:
                raise GrantVerificationError(
                    f"agent {agent!r} is granted tool {tool!r}, which is not a known "
                    f"tool ({sorted(known)})"
                )


@dataclass(frozen=True)
class ActionRequest:
    agent: str
    tool: str
    args: dict[str, Any]
    estimation_id: str | None
    step: int


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    requires_human_approval: bool = False
    reason: str = ""
    risk: ToolRisk | None = None
    redacted_args: dict[str, Any] = field(default_factory=dict)


_SENSITIVE_KEYS = {"transcript", "note", "content", "body", "reasoning"}


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in args.items():
        if key in _SENSITIVE_KEYS:
            redacted[key] = "«redacted»"
        elif isinstance(value, str) and len(value) > 80:
            redacted[key] = value[:77] + "…"
        else:
            redacted[key] = value
    return redacted


def _human_approved(state: dict[str, Any]) -> bool:
    decision = state.get("human_decision") or {}
    action = decision.get("decision") or decision.get("action")
    return action == "approve"


def guard_action(req: ActionRequest, state: dict[str, Any]) -> GuardDecision:
    redacted = _redact(req.args if isinstance(req.args, dict) else {})
    risk = TOOL_RISK.get(req.tool)

    granted = AGENT_TOOL_GRANTS.get(req.agent, frozenset())
    if req.tool not in granted:
        return GuardDecision(
            allowed=False,
            reason=f"agent {req.agent!r} is not granted tool {req.tool!r} "
            f"(granted: {sorted(granted) or 'none'})",
            risk=risk,
            redacted_args=redacted,
        )

    if not isinstance(req.args, dict):
        return GuardDecision(
            allowed=False, reason="arguments must be an object", risk=risk, redacted_args=redacted
        )

    run_id = state.get("estimation_id")
    if req.estimation_id != run_id:
        return GuardDecision(
            allowed=False,
            reason=f"action estimation_id {req.estimation_id!r} does not match the "
            f"current run {run_id!r}",
            risk=risk,
            redacted_args=redacted,
        )
    args_id = req.args.get("estimation_id")
    if args_id is not None and args_id != run_id:
        return GuardDecision(
            allowed=False,
            reason=f"argument estimation_id {args_id!r} does not match the current run {run_id!r}",
            risk=risk,
            redacted_args=redacted,
        )

    if req.tool == SAVE_ESTIMATE_TOOL and not req.args.get("estimate"):
        return GuardDecision(
            allowed=False,
            reason="save_estimate requires a non-empty 'estimate' payload",
            risk=risk,
            redacted_args=redacted,
        )

    if risk == ToolRisk.IRREVERSIBLE and not _human_approved(state):
        return GuardDecision(
            allowed=True,
            requires_human_approval=True,
            reason="irreversible action requires a human approval; route through the gate",
            risk=risk,
            redacted_args=redacted,
        )

    return GuardDecision(allowed=True, reason="ok", risk=risk, redacted_args=redacted)


SaveSink = Callable[[str | None, dict[str, Any]], dict[str, Any]]

PERSISTED: dict[str, dict[str, Any]] = {}


def _default_sink(estimation_id: str | None, estimate: dict[str, Any]) -> dict[str, Any]:
    record = {"estimation_id": estimation_id, "estimate": estimate}
    PERSISTED[estimation_id or "?"] = record
    log.info(
        "persistence_would_write",
        estimation_id=estimation_id,
        total_engineer_days=(estimate or {}).get("total_engineer_days"),
    )
    return {"ok": True, "stored": True}


async def execute_guarded(
    req: ActionRequest,
    state: dict[str, Any],
    *,
    sink: SaveSink | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    started = perf_counter()
    digest = _digest(req.args if isinstance(req.args, dict) else {})
    decision = guard_action(req, state)
    preview = str(decision.redacted_args)[: settings.supervisor_audit_args_preview_chars]

    def _contribution(outcome: str, summary: str) -> dict[str, Any]:
        return {
            "step": req.step,
            "agent": req.agent,
            "action": f"tool:{req.tool}",
            "tool": req.tool,
            "outcome": outcome,
            "summary": summary[:200],
            "args_digest": digest,
            "duration_ms": int((perf_counter() - started) * 1000),
        }

    if not decision.allowed:
        log.error(
            "agent_privilege_denied",
            estimation_id=req.estimation_id,
            step=req.step,
            agent=req.agent,
            tool=req.tool,
            risk=str(decision.risk),
            args_digest=digest,
            args_preview=preview,
            reason=decision.reason,
        )
        return (
            {"ok": False, "error": "denied", "summary": decision.reason},
            _contribution("denied", decision.reason),
        )

    if decision.requires_human_approval:
        log.warning(
            "agent_action_deferred",
            estimation_id=req.estimation_id,
            step=req.step,
            agent=req.agent,
            tool=req.tool,
            risk=str(decision.risk),
            args_digest=digest,
            args_preview=preview,
            reason=decision.reason,
        )
        return (
            {"ok": False, "error": "awaiting_human_approval", "summary": decision.reason},
            _contribution("deferred", decision.reason),
        )

    try:
        run_sink = sink or _default_sink
        result = run_sink(req.estimation_id, req.args.get("estimate") or {})
        outcome, summary = "ok", "estimate persisted (guarded, human-authorised)"
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": type(exc).__name__, "summary": str(exc)[:200]}
        outcome, summary = "error", str(exc)[:200]

    log.info(
        "agent_action",
        estimation_id=req.estimation_id,
        step=req.step,
        agent=req.agent,
        tool=req.tool,
        action=f"tool:{req.tool}",
        outcome=outcome,
        risk=str(decision.risk),
        args_digest=digest,
        args_preview=preview,
        result_summary=summary,
    )
    return result, _contribution(outcome, summary)
