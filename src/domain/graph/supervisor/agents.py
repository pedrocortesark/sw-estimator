"""The four specialised agents — pure ``state -> partial update`` functions.

Each one is a reorganisation of the Session 13 pre-exercise nodes (still in
``src/domain/graph/nodes.py``), not a rewrite: the prompts, the response models and
the deterministic guardrails are imported from there. What changed is the SHAPE —
five nodes wired in a fixed line become four agents that the supervisor dispatches,
each holding exactly the tools it needs and nothing more:

===========================  ===========================================
agent                        tools
===========================  ===========================================
``requirements_extractor``   (none — the model only)
``budget_searcher``          ``search_budgets``
``estimate_generator``       ``derive_task_hours`` (the "calculate" tool)
``coherence_validator``      ``validate_estimate``
===========================  ===========================================

Every tool call goes through ``guarded_dispatch``, never through ``dispatch_tool``
directly — that is what makes the privilege table load-bearing rather than
documentation. Each agent returns the contributions it collected in
``agent_contributions``, so the audit trail is assembled by the reducer rather than by
a side effect, and the agents stay pure.

``make_retrieval_backend`` and ``distance_weighted_consensus`` are imported at MODULE
level on purpose: that is the monkeypatch seam the network-free tests use, matching
the convention in ``agents/hours.py``.
"""

from __future__ import annotations

import asyncio
from time import perf_counter

import logfire
import structlog

from src.core.config import get_settings
from src.domain.graph.nodes import (
    HOURS_PER_DAY,
    _GENERATE_SYSTEM_PROMPT,
    _CLASSIFY_SYSTEM_PROMPT,
    _EXTRACT_SYSTEM_PROMPT,
    _max_tokens_for,
    _norm,
    _reasoning_effort_for,
    _references_for,
    _validate,
)
from src.domain.graph.schemas import (
    ComponentClassification,
    ConsolidatedEstimate,
    RequirementsExtraction,
)
from src.domain.graph.state import BudgetMatch, Component
from src.domain.graph.supervisor.competition import COMPETITION_GRAPH, compute_divergence
from src.domain.graph.supervisor.privilege import (
    CALCULATE_TOOL,
    guarded_dispatch,
    record_model_action,
)
from src.domain.graph.supervisor.sandbox import ActionRequest, execute_guarded
from src.domain.graph.supervisor.state import SupervisorState
from src.generation.rag.agent_retrieval import make_retrieval_backend
from src.generation.rag.task_hours import distance_weighted_consensus

log = structlog.get_logger()


def _step_of(state: SupervisorState) -> int:
    return int(state.get("supervisor_steps") or 0)


async def requirements_extractor(state: SupervisorState) -> dict:
    with logfire.span("agent: requirements_extractor"):
        settings = get_settings()
        from src.dependencies import get_llm_wrapper

        wrapper = get_llm_wrapper()
        step = _step_of(state)
        estimation_id = state.get("estimation_id")
        model = settings.graph_extraction_model

        started = perf_counter()
        extraction, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=_EXTRACT_SYSTEM_PROMPT,
            user_message=state["transcript"],
            response_model=RequirementsExtraction,
            model_override=model,
        )
        requirements = [r.strip() for r in extraction.requirements if r.strip()]
        contribution_extract = record_model_action(
            "requirements_extractor",
            "extract_requirements",
            step=step,
            estimation_id=estimation_id,
            model=model,
            summary=f"{len(requirements)} requirements extracted from the transcript",
            duration_ms=int((perf_counter() - started) * 1000),
        )

        started = perf_counter()
        classification, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=_CLASSIFY_SYSTEM_PROMPT,
            user_message="Requirements:\n" + "\n".join(f"- {r}" for r in requirements),
            response_model=ComponentClassification,
            model_override=model,
        )
        components: list[Component] = [
            {"name": c.name.strip(), "category": c.category.strip()}
            for c in classification.components
            if c.name.strip()
        ]
        contribution_classify = record_model_action(
            "requirements_extractor",
            "classify_components",
            step=step,
            estimation_id=estimation_id,
            model=model,
            summary=f"{len(components)} components: "
            + ", ".join(c["name"] for c in components[:5]),
            duration_ms=int((perf_counter() - started) * 1000),
        )

        log.info(
            "supervisor_agent_requirements_extractor",
            requirements=len(requirements),
            components=len(components),
        )
        return {
            "requirements": requirements,
            "components": components,
            "agent_contributions": [contribution_extract, contribution_classify],
        }


async def budget_searcher(state: SupervisorState) -> dict:
    with logfire.span("agent: budget_searcher"):
        settings = get_settings()
        step = _step_of(state)
        estimation_id = state.get("estimation_id")
        backend = make_retrieval_backend(
            settings.agent_search_top_k, settings.agent_search_distance_threshold
        )

        matches: list[BudgetMatch] = []
        contributions: list[dict] = []
        errors: list[str] = []

        for component in state.get("components") or []:
            result, contribution = await guarded_dispatch(
                "budget_searcher",
                "search_budgets",
                {
                    "query": f"{component['name']} ({component['category']})",
                    "filters": {
                        "sectors": None,
                        "component_type": component["category"],
                    },
                },
                step=step,
                estimation_id=estimation_id,
                backend=backend,
            )
            contributions.append(contribution)

            if not result.get("ok", True) and result.get("error"):
                errors.append(
                    f"budget search failed for {component['name']!r}: {result.get('summary')}"
                )
                continue

            for item in result.get("items") or []:
                hours = item.get("estimated_hours")
                if hours is None:
                    continue
                matches.append(
                    {
                        "component": component["name"],
                        "reference_budget_id": item.get("budget_id"),
                        "amount": float(hours),
                        "distance": float(item.get("distance") or 0.0),
                    }
                )

        log.info(
            "supervisor_agent_budget_searcher",
            components=len(state.get("components") or []),
            matches=len(matches),
        )
        update: dict = {"budget_matches": matches, "agent_contributions": contributions}
        if errors:
            update["errors"] = errors
        return update


def _reference_rows_for(component: str, matches: list[BudgetMatch]) -> list[BudgetMatch]:
    target = _norm(component)
    return [m for m in matches if _norm(m["component"]) == target]


async def estimate_generator(state: SupervisorState) -> dict:
    with logfire.span("agent: estimate_generator"):
        settings = get_settings()
        from src.dependencies import get_llm_wrapper

        wrapper = get_llm_wrapper()
        step = _step_of(state)
        estimation_id = state.get("estimation_id")
        components = state.get("components") or []
        matches = state.get("budget_matches") or []

        anchors: list[dict] = []
        contributions: list[dict] = []
        for component in components:
            rows = _reference_rows_for(component["name"], matches)
            if not rows:
                anchors.append({"name": component["name"], "has_match": False})
                continue
            result, contribution = await guarded_dispatch(
                "estimate_generator",
                CALCULATE_TOOL,
                {
                    "module": component["category"],
                    "task": component["name"],
                    "neighbors": [
                        {
                            "estimated_hours": int(row["amount"]),
                            "distance": float(row["distance"]),
                            "source_id": None,
                            "budget_id": row.get("reference_budget_id"),
                        }
                        for row in rows
                    ],
                },
                step=step,
                estimation_id=estimation_id,
                consensus_fn=distance_weighted_consensus,
            )
            contributions.append(contribution)
            anchors.append(
                {
                    "name": component["name"],
                    "estimated_hours": result.get("estimated_hours"),
                    "reliability": result.get("reliability"),
                    "dispersion": result.get("dispersion"),
                    "has_match": bool(result.get("has_match")),
                }
            )

        anchor_by_name = {a["name"]: a for a in anchors}
        lines: list[str] = []
        for component in components:
            refs = _references_for(component["name"], matches)
            ref_text = ", ".join(f"{h:.0f}h" for h in refs) if refs else "no references"
            anchor = anchor_by_name.get(component["name"]) or {}
            if anchor.get("has_match"):
                anchor_text = (
                    f" | consensus anchor = {anchor['estimated_hours']}h "
                    f"(reliability {anchor.get('reliability')})"
                )
            else:
                anchor_text = " | no consensus anchor (no historical analog)"
            lines.append(
                f"- {component['name']} [{component['category']}]: "
                f"references = {ref_text}{anchor_text}"
            )

        model = settings.graph_generation_model
        started = perf_counter()
        result, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=_GENERATE_SYSTEM_PROMPT,
            user_message="Components, their historical reference budgets and the "
            "deterministic consensus anchors:\n" + "\n".join(lines),
            response_model=ConsolidatedEstimate,
            model_override=model,
            max_tokens=_max_tokens_for(model),
            reasoning_effort=_reasoning_effort_for(model),
        )
        contributions.append(
            record_model_action(
                "estimate_generator",
                "consolidate_estimate",
                step=step,
                estimation_id=estimation_id,
                model=model,
                summary=f"total {result.total_engineer_days}d over "
                f"{len(result.components)} components (confidence {result.confidence})",
                duration_ms=int((perf_counter() - started) * 1000),
            )
        )

        log.info(
            "supervisor_agent_estimate_generator",
            components=len(result.components),
            total_engineer_days=result.total_engineer_days,
            anchored=sum(1 for a in anchors if a.get("has_match")),
        )
        return {
            "estimate": result.model_dump(),
            "component_anchors": anchors,
            "agent_contributions": contributions,
        }


def _competition_brief(state: SupervisorState, estimate: dict) -> str:
    components = state.get("components") or []
    matches = state.get("budget_matches") or []
    lines = ["Project components and their historical reference budgets (engineer-hours):"]
    for component in components:
        refs = _references_for(component["name"], matches)
        ref_text = ", ".join(f"{h:.0f}h" for h in refs) if refs else "no historical references"
        lines.append(f"- {component['name']} [{component['category']}]: {ref_text}")
    lines.append(
        f"\nGrounded consolidation total (engineer-days): {estimate.get('total_engineer_days')}"
    )
    return "\n".join(lines)


async def competitive_estimate_generator(state: SupervisorState) -> dict:
    base = await estimate_generator(state)
    with logfire.span("agent: competitive_estimate_generator"):
        step = _step_of(state)
        estimation_id = state.get("estimation_id")
        brief = _competition_brief(state, base["estimate"])
        sub = await COMPETITION_GRAPH.ainvoke({"brief": brief})
        proposals = sub.get("proposals") or []
        divergence = sub.get("divergence") or compute_divergence(proposals)
        synthesis = sub.get("synthesis") or {}

        estimate = dict(base["estimate"])
        if synthesis:
            estimate["range"] = {"low": synthesis.get("low"), "high": synthesis.get("high")}
            estimate["open_questions"] = synthesis.get("open_questions") or []

        contributions = list(base.get("agent_contributions") or [])
        for proposal in proposals:
            contributions.append(
                record_model_action(
                    "estimate_generator",
                    f"competition_{proposal.get('stance')}",
                    step=step,
                    estimation_id=estimation_id,
                    summary=f"{proposal.get('stance')} total = "
                    f"{proposal.get('total_engineer_days')}d",
                )
            )
        contributions.append(
            record_model_action(
                "estimate_generator",
                "competition_synthesis",
                step=step,
                estimation_id=estimation_id,
                summary=f"range {synthesis.get('low')}..{synthesis.get('high')}d, "
                f"divergence {divergence.get('ratio')} ({divergence.get('level')})",
            )
        )

        log.info(
            "supervisor_agent_competitive_estimate_generator",
            divergence=divergence.get("ratio"),
            level=divergence.get("level"),
        )
        return {
            **base,
            "estimate": estimate,
            "proposals": proposals,
            "divergence": divergence,
            "synthesis": synthesis,
            "agent_contributions": contributions,
        }


async def persistence_agent(state: SupervisorState) -> dict:
    with logfire.span("agent: persistence_agent"):
        step = _step_of(state)
        estimation_id = state.get("estimation_id")
        request = ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": estimation_id, "estimate": state.get("estimate") or {}},
            estimation_id=estimation_id,
            step=step,
        )
        result, contribution = await execute_guarded(request, state)
        log.info("supervisor_agent_persistence", outcome=contribution.get("outcome"))
        return {"saved": result, "agent_contributions": [contribution]}


def _apply_divergence_penalty(confidence: float, divergence: dict | None) -> float:
    ratio = float((divergence or {}).get("ratio") or 0.0)
    if ratio <= 0.0:
        return confidence
    penalty = get_settings().supervisor_divergence_penalty * min(ratio, 1.0)
    return max(0.0, min(1.0, confidence - penalty))


def _confidence_score(estimate: dict, issues: list[str], grounded: int, total: int) -> float:
    base = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(estimate.get("confidence"), 0.6)
    ratio = (grounded / total) if total else 0.0
    return max(0.0, min(1.0, base * ratio - 0.1 * len(issues)))


async def coherence_validator(state: SupervisorState) -> dict:
    with logfire.span("agent: coherence_validator"):
        step = _step_of(state)
        estimation_id = state.get("estimation_id")
        estimate = state.get("estimate") or {}
        matches = state.get("budget_matches") or []
        components = estimate.get("components") or []

        result, contribution = await guarded_dispatch(
            "coherence_validator",
            "validate_estimate",
            {
                "components": [
                    {
                        "name": c.get("name", "?"),
                        "estimated_hours": (c.get("engineer_days") or 0) * HOURS_PER_DAY,
                        "reference_amounts": _references_for(c.get("name", "?"), matches),
                    }
                    for c in components
                ],
                "total_hours": (estimate.get("total_engineer_days") or 0) * HOURS_PER_DAY,
            },
            step=step,
            estimation_id=estimation_id,
        )

        issues = _validate(estimate, matches)
        grounded = sum(1 for c in components if _references_for(c.get("name", "?"), matches))
        total = len(components)
        confidence = _confidence_score(estimate, issues, grounded, total)
        confidence = _apply_divergence_penalty(confidence, state.get("divergence"))

        log.info(
            "supervisor_agent_coherence_validator",
            issues=len(issues),
            confidence=round(confidence, 3),
            grounded=grounded,
            total=total,
        )
        update: dict = {
            "status": "validated" if not issues else "needs_review",
            "validation": result,
            "confidence": confidence,
            "out_of_range": any("outside the plausible range" in i for i in issues),
            "grounded_components": grounded,
            "agent_contributions": [contribution],
        }
        if issues:
            update["errors"] = issues
        if get_settings().supervisor_persistence_enabled:
            update["persist_requested"] = True
        return update
