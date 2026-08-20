"""Competition pattern (Session 14 LIVE): two estimators disagree, on purpose.

Where ``estimate_generator`` produces ONE consolidated number, this runs TWO estimators
with substantively different priors and then a synthesizer that turns their disagreement
into information. The shape is a fan-out / fan-in::

              ┌──▶ conservative_estimator ──┐
    START ────┤                             ├──▶ synthesizer ──▶ END
              └──▶ aggressive_estimator ────┘
                (both in the SAME superstep)

* **Parallel** — the two estimators are reached by two edges from ``START`` and run in
  the same LangGraph superstep. Neither sees the other's proposal; that independence is
  what makes the divergence meaningful rather than an echo.
* **Fan-in by reducer** — both write to ``proposals: Annotated[list, operator.add]``, so
  the synthesizer receives both regardless of which finished first.

The prompts are NOT "the same brief, one cautious / one bold". They apply DIFFERENT
criteria: the conservative estimator weights integration friction, undocumented legacy
interfaces, certification overhead and spec debt; the aggressive one weights reuse,
closed scope and strong historical analogs. Same evidence, different lens.

``compute_divergence`` is PURE ARITHMETIC — the model never judges how far apart the two
are, because "how uncertain is this project" must be reproducible and unit-testable. The
synthesizer is explicitly forbidden from averaging: it returns a range, its driving
assumptions and the open questions that would collapse it.

The whole subgraph is invoked from inside the supervisor's estimate node (see
``agents.competitive_estimate_generator``), so it stays one leg of the vertical trace
rather than a second graph to reason about.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Optional

import logfire
import structlog
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.config import get_settings
from src.domain.graph.schemas import EstimateProposal, SynthesizedEstimate

log = structlog.get_logger()


class CompetitionState(TypedDict, total=False):
    brief: str
    proposals: Annotated[list[dict], operator.add]
    divergence: Optional[dict]
    synthesis: Optional[dict]


_CONSERVATIVE_SYSTEM_PROMPT = (
    "You are a RISK-FIRST senior estimator. You have seen projects blow past their "
    "estimate and you price that memory in. Read the components, their scope and the "
    "historical references, then produce a single total in engineer-days under these "
    "criteria:\n"
    "- Weight INTEGRATION FRICTION heavily: undocumented or legacy interfaces, "
    "third-party SDKs with thin docs, protocols nobody owns anymore.\n"
    "- Treat VAGUE or open scope as LARGER, not smaller — unresolved scope becomes work.\n"
    "- Price CERTIFICATION / COMPLIANCE overhead (SOC2, FIPS, audits) as real effort.\n"
    "- Count TECHNICAL UNKNOWNS (first-of-its-kind work, no internal precedent) as a "
    "multiplier, not a footnote.\n"
    "Return your total, the assumptions it rests on, the risks that justify the caution, "
    "and one paragraph of reasoning. Set stance to 'conservative'."
)

_AGGRESSIVE_SYSTEM_PROMPT = (
    "You are a REUSE-FIRST senior estimator. You have seen teams gold-plate estimates "
    "out of fear and lose the bid. Read the components, their scope and the historical "
    "references, then produce a single total in engineer-days under these criteria:\n"
    "- Weight REUSE heavily: strong historical analogs, standard patterns, libraries and "
    "internal components the team already ships.\n"
    "- Treat CLOSED, well-understood scope as containable — do not pad for hypotheticals.\n"
    "- Trust the HISTORICAL REFERENCE HOURS as the base case when analogs are close.\n"
    "- Assume a COMPETENT team executing familiar work at a normal pace.\n"
    "Return your total, the assumptions it rests on, the risks you are consciously "
    "accepting, and one paragraph of reasoning. Set stance to 'aggressive'."
)

_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are the estimation LEAD reconciling two independent estimates of the same "
    "project: a risk-first (conservative) and a reuse-first (aggressive) one. You are "
    "given both proposals and the ARITHMETIC divergence between their totals.\n"
    "\n"
    "DO NOT AVERAGE the two numbers. An average destroys the only useful thing the "
    "disagreement produced. Instead:\n"
    "- Return a RANGE [low, high] that brackets the honest uncertainty. Anchor low near "
    "the reuse-first total and high near the risk-first total; widen it if the divergence "
    "is large.\n"
    "- List the DRIVING ASSUMPTIONS: the few beliefs that most move the number between "
    "low and high.\n"
    "- List the OPEN QUESTIONS whose answers would let a human collapse the range — the "
    "things that, once known, turn this into a point estimate.\n"
    "- Set confidence to reflect the SPREAD: a wide divergence is 'low' confidence in a "
    "single number, however sure each estimator was of their own.\n"
    "Write one paragraph of reasoning that explains the bracket, not an average."
)


def _proposal_lines(proposals: list[dict]) -> str:
    lines: list[str] = []
    for proposal in proposals:
        lines.append(
            f"- [{proposal.get('stance')}] total = {proposal.get('total_engineer_days')} "
            f"engineer-days\n"
            f"    assumptions: {'; '.join(proposal.get('assumptions') or []) or '—'}\n"
            f"    risks: {'; '.join(proposal.get('risks') or []) or '—'}"
        )
    return "\n".join(lines)


def compute_divergence(proposals: list[dict]) -> dict:
    totals = [
        int(p["total_engineer_days"]) for p in proposals if p.get("total_engineer_days") is not None
    ]
    if len(totals) < 2:
        only = totals[0] if totals else 0
        return {"low": only, "high": only, "spread": 0, "ratio": 0.0, "level": "low"}

    low, high = min(totals), max(totals)
    spread = high - low
    midpoint = (low + high) / 2
    ratio = (spread / midpoint) if midpoint else 0.0
    level = "high" if ratio >= 0.5 else "medium" if ratio >= 0.2 else "low"
    return {
        "low": low,
        "high": high,
        "spread": spread,
        "ratio": round(ratio, 3),
        "level": level,
    }


async def _estimate(stance: str, system_prompt: str, brief: str) -> dict:
    from src.dependencies import get_llm_wrapper

    settings = get_settings()
    wrapper = get_llm_wrapper()
    proposal, _meta = await asyncio.to_thread(
        wrapper.complete_structured,
        system_prompt=system_prompt,
        user_message=brief,
        response_model=EstimateProposal,
        model_override=settings.graph_generation_model,
    )
    data = proposal.model_dump()
    data["stance"] = stance
    return data


async def conservative_estimator(state: CompetitionState) -> dict:
    with logfire.span("competition: conservative"):
        proposal = await _estimate("conservative", _CONSERVATIVE_SYSTEM_PROMPT, state["brief"])
        log.info("competition_conservative", total=proposal.get("total_engineer_days"))
        return {"proposals": [proposal]}


async def aggressive_estimator(state: CompetitionState) -> dict:
    with logfire.span("competition: aggressive"):
        proposal = await _estimate("aggressive", _AGGRESSIVE_SYSTEM_PROMPT, state["brief"])
        log.info("competition_aggressive", total=proposal.get("total_engineer_days"))
        return {"proposals": [proposal]}


async def synthesizer(state: CompetitionState) -> dict:
    with logfire.span("competition: synthesizer"):
        from src.dependencies import get_llm_wrapper

        settings = get_settings()
        proposals = state.get("proposals") or []
        divergence = compute_divergence(proposals)

        wrapper = get_llm_wrapper()
        user_message = (
            f"{_proposal_lines(proposals)}\n\n"
            f"Arithmetic divergence: spread = {divergence['spread']} engineer-days "
            f"(low {divergence['low']}, high {divergence['high']}), "
            f"ratio = {divergence['ratio']} ({divergence['level']})."
        )
        synthesis, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=_SYNTHESIZER_SYSTEM_PROMPT,
            user_message=user_message,
            response_model=SynthesizedEstimate,
            model_override=settings.graph_generation_model,
        )
        log.info(
            "competition_synthesis",
            low=synthesis.low,
            high=synthesis.high,
            divergence=divergence["ratio"],
        )
        return {"divergence": divergence, "synthesis": synthesis.model_dump()}


def build_competition_subgraph():
    builder = StateGraph(CompetitionState)
    builder.add_node("conservative_estimator", conservative_estimator)
    builder.add_node("aggressive_estimator", aggressive_estimator)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "conservative_estimator")
    builder.add_edge(START, "aggressive_estimator")
    builder.add_edge("conservative_estimator", "synthesizer")
    builder.add_edge("aggressive_estimator", "synthesizer")
    builder.add_edge("synthesizer", END)
    return builder.compile()


COMPETITION_GRAPH = build_competition_subgraph()
