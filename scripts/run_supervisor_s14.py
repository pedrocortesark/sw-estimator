#!/usr/bin/env python3
"""Session 14 — run the SUPERVISOR multi-agent flow end to end.

Drives the compiled supervisor graph (``src/domain/graph/supervisor``) and prints the
four things the exercise asks you to be able to see:

1. **ROUTING** — every supervisor decision, its reason, and whether the model's choice
   stood or was overridden by the legality guard.
2. **TOOL PRIVILEGE** — the declared allowlist per agent.
3. **AUDIT TRAIL** — every agent action (model or tool), including any DENIED one.
4. **HUMAN REVIEW** — whether the gate tripped, why, and what decision resumed it.

Unlike Session 13's runner, the pause here is CONDITIONAL: a well-grounded transcript
runs straight through and never stops. Use ``sample_transcript_edge_case.txt`` to see
the gate actually fire.

Run variants::

    # Deliverable run: real retrieval + Postgres checkpointer + Logfire
    docker compose exec estimator python scripts/run_supervisor_s14.py \\
        --out exercises/session-14/example_run_edge_case.txt

    # Offline smoke: no DB, canned retrieval (still needs OPENAI_API_KEY for the
    # router + the extraction/consolidation agents)
    uv run python scripts/run_supervisor_s14.py --memory --stub

    # Level 3 demo: make an agent reach for a tool it does not hold, and watch the
    # denial land in the audit trail without killing the run
    uv run python scripts/run_supervisor_s14.py --memory --stub --violate
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.types import Command  # noqa: E402

from src.domain.graph.supervisor.build import build_supervisor_graph  # noqa: E402
from src.domain.graph.supervisor.privilege import AGENT_PRIVILEGES  # noqa: E402

DEFAULT_TRANSCRIPT = REPO_ROOT / "exercises" / "session-14" / "edge_cases" / "low_confidence.txt"
STUB_PATH = REPO_ROOT / "exercises" / "session-12" / "reference_retrieval.py"


def _install_stub_backend() -> None:
    spec = importlib.util.spec_from_file_location("reference_retrieval", STUB_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _make(*_args, **_kwargs):
        async def _backend(query, sectors=None):
            return module.search_budgets_stub(query, {"sectors": sectors})

        return _backend

    from src.domain.graph.supervisor import agents

    agents.make_retrieval_backend = _make


def _install_violation_probe() -> None:
    from src.domain.graph.supervisor import agents
    from src.domain.graph.supervisor.privilege import guarded_dispatch

    original = agents.budget_searcher

    async def _probing_budget_searcher(state):
        _result, denial = await guarded_dispatch(
            "budget_searcher",
            "validate_estimate",
            {"components": [], "total_hours": 0},
            step=int(state.get("supervisor_steps") or 0),
            estimation_id=state.get("estimation_id"),
        )
        update = await original(state)
        return {
            **update,
            "agent_contributions": [denial, *(update.get("agent_contributions") or [])],
        }

    agents.budget_searcher = _probing_budget_searcher
    from src.domain.graph.supervisor import build as build_module

    build_module.AGENT_NODES["budget_searcher"] = _probing_budget_searcher


def _render_routing(state: dict) -> list[str]:
    lines = ["ROUTING (supervisor decisions)", "-" * 78]
    for row in state.get("routing_history") or []:
        lines.append(
            f"  {row['step'] + 1}. supervisor → {row['next_agent']:<24} "
            f"[{row.get('source', '?')}]  {row.get('reason', '')[:90]}"
        )
    if not (state.get("routing_history") or []):
        lines.append("  (no routing decisions recorded)")
    return lines


def _render_privilege() -> list[str]:
    lines = ["", "TOOL PRIVILEGE (declared allowlists)", "-" * 78]
    for agent, tools in AGENT_PRIVILEGES.items():
        rendered = ", ".join(sorted(tools)) if tools else "(no tools)"
        lines.append(f"  {agent:<24} : {rendered}")
    return lines


def _render_competition(state: dict) -> list[str]:
    proposals = state.get("proposals")
    if not proposals:
        return []
    lines = ["", "COMPETITION (conservative vs aggressive)", "-" * 78]
    for proposal in proposals:
        lines.append(
            f"  {proposal.get('stance', '?'):<14} {str(proposal.get('total_engineer_days')) + 'd':>8}  "
            f"risks: {', '.join(proposal.get('risks') or []) or '—'}"
        )
    divergence = state.get("divergence") or {}
    synthesis = state.get("synthesis") or {}
    lines.append(
        f"  divergence   : ratio {divergence.get('ratio')} ({divergence.get('level')}), "
        f"spread {divergence.get('spread')}d"
    )
    lines.append(
        f"  synthesized  : {synthesis.get('low')}..{synthesis.get('high')}d "
        f"(confidence {synthesis.get('confidence')})"
    )
    for question in synthesis.get("open_questions") or []:
        lines.append(f"    open question: {question}")
    return lines


def _render_audit(state: dict) -> list[str]:
    lines = ["", "AUDIT TRAIL (agent_contributions)", "-" * 78]
    for row in state.get("agent_contributions") or []:
        marker = {"ok": "ok", "denied": "DENIED", "error": "ERROR", "deferred": "DEFER"}.get(
            row.get("outcome", "?"), "?"
        )
        digest = row.get("args_digest") or "-"
        duration = row.get("duration_ms")
        timing = f"{duration}ms" if duration is not None else "-"
        lines.append(
            f"  [{marker:^6}] {row.get('agent', '?'):<24} {row.get('action', '?'):<28} "
            f"{row.get('summary', '')[:60]:<60} ({digest}, {timing})"
        )
    if not (state.get("agent_contributions") or []):
        lines.append("  (no actions recorded)")
    return lines


def _render_review(state: dict, decision: str | None) -> list[str]:
    lines = ["", "HUMAN REVIEW", "-" * 78]
    reasons = state.get("review_reasons") or []
    if state.get("needs_human_review"):
        lines.append("  triggered: YES")
        for reason in reasons:
            lines.append(f"    - {reason}")
        lines.append(f"  decision : {decision or '(not resumed)'}")
    else:
        confidence = state.get("confidence")
        confidence_text = f"{confidence:.2f}" if confidence is not None else "n/a"
        lines.append(f"  triggered: no (confidence {confidence_text}, all conditions clear)")
    return lines


def _render_estimate(state: dict) -> list[str]:
    lines = ["", "ESTIMATE", "-" * 78]
    estimate = state.get("estimate") or {}
    for component in estimate.get("components") or []:
        days = component.get("engineer_days")
        rendered = f"{days}d" if days is not None else "— (unbudgeted)"
        lines.append(f"  {component.get('name', '?'):<40} {rendered:>16}")
    lines.append(f"  {'TOTAL':<40} {str(estimate.get('total_engineer_days')) + 'd':>16}")
    estimate_range = estimate.get("range")
    if estimate_range:
        rendered_range = f"{estimate_range.get('low')}..{estimate_range.get('high')}d"
        lines.append(f"  {'RANGE (competition)':<40} {rendered_range:>16}")
    lines.append(
        f"  status = {state.get('status')} · model confidence = {estimate.get('confidence')}"
    )
    saved = state.get("saved")
    if saved is not None:
        outcome = "persisted" if saved.get("ok") else f"NOT persisted ({saved.get('error')})"
        lines.append(f"  persistence = {outcome}")
    errors = state.get("errors") or []
    if errors:
        lines.append("")
        lines.append("ISSUES")
        lines.append("-" * 78)
        for issue in errors:
            lines.append(f"  - {issue}")
    return lines


def _render(state: dict, decision: str | None) -> str:
    return "\n".join(
        [
            "=" * 78,
            "SESSION 14 — SUPERVISOR MULTI-AGENT RUN",
            "=" * 78,
            "",
            *_render_routing(state),
            *_render_privilege(),
            *_render_competition(state),
            *_render_audit(state),
            *_render_review(state, decision),
            *_render_estimate(state),
            "",
        ]
    )


async def _run_to_completion(graph, transcript: str, estimation_id: str, decision: str):
    config = {"configurable": {"thread_id": f"s14:{estimation_id}"}}
    await graph.ainvoke({"transcript": transcript, "estimation_id": estimation_id}, config)

    answered: str | None = None
    while True:
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            return snapshot.values, answered

        interrupts = getattr(snapshot, "interrupts", None) or ()
        if not interrupts:
            await graph.ainvoke(None, config)
            continue

        payload = interrupts[0].value or {}
        print(f"\n  ⏸ human gate '{payload.get('gate')}' — resuming with '{decision}'")
        for reason in payload.get("reasons") or []:
            print(f"      · {reason}")
        answered = decision
        await graph.ainvoke(
            Command(resume={"decision": decision, "note": "auto-approved by the S14 runner"}),
            config,
        )


async def _main_async(args: argparse.Namespace) -> int:
    from src.core.config import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        print(
            "OPENAI_API_KEY is required (the router and the LLM agents need it).", file=sys.stderr
        )
        return 1

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)
        return 1
    transcript = transcript_path.read_text(encoding="utf-8")

    if args.stub:
        _install_stub_backend()
    if args.violate:
        _install_violation_probe()

    estimation_id = args.estimation_id or f"s14-{transcript_path.stem}"
    print("=" * 78)
    print(f"transcript     : {transcript_path}")
    print(f"estimation_id  : {estimation_id}")
    print(f"router model   : {settings.supervisor_router_model}")
    print(f"threshold      : {settings.supervisor_confidence_threshold}")
    print(f"checkpointer   : {'MemorySaver' if args.memory else 'AsyncPostgresSaver'}")
    print(f"variant        : competitive={args.compete} · sandboxed={args.persist}")
    print("=" * 78)

    build_kwargs = {"competitive": args.compete, "sandboxed": args.persist}
    if args.memory:
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_supervisor_graph(MemorySaver(), **build_kwargs)
        state, decision = await _run_to_completion(graph, transcript, estimation_id, args.decision)
    else:
        from src.domain.graph.checkpointer import open_checkpointer

        async with open_checkpointer() as checkpointer:
            graph = build_supervisor_graph(checkpointer, **build_kwargs)
            state, decision = await _run_to_completion(
                graph, transcript, estimation_id, args.decision
            )

    rendered = _render(state, decision)
    print(rendered)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        print(f"\nWrote {out_path}")
    return 0


def main() -> int:
    from src.core.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transcript", default=str(DEFAULT_TRANSCRIPT))
    parser.add_argument("--estimation-id", default=None)
    parser.add_argument(
        "--decision",
        choices=["approve", "adjust", "reject"],
        default="approve",
        help="What the runner answers if the human gate trips.",
    )
    parser.add_argument(
        "--memory", action="store_true", help="Use a MemorySaver instead of Postgres."
    )
    parser.add_argument(
        "--stub", action="store_true", help="Use the offline retrieval stub (no DB)."
    )
    parser.add_argument(
        "--violate",
        action="store_true",
        help="Make an agent attempt an out-of-privilege tool (Level 3 demo).",
    )
    parser.add_argument(
        "--compete",
        action="store_true",
        help="Run the estimate step as a conservative-vs-aggressive competition (S14 live).",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Append the sandboxed persistence_agent; the human gate authorises the write.",
    )
    parser.add_argument("--out", default=None, help="Write the rendered run to this file.")
    args = parser.parse_args()

    if args.persist:
        import os

        os.environ["SUPERVISOR_PERSISTENCE_ENABLED"] = "true"
        get_settings.cache_clear()

    _ = settings
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
