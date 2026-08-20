"""Session 14 (live) — the competition pattern.

Covers: the pure divergence arithmetic, the parallel fan-out / fan-in, the synthesizer
producing a RANGE (not an average), and the wiring that turns high divergence into low
confidence so the SAME human gate trips.
"""

from __future__ import annotations

import pytest

from src.core.config import get_settings
from src.domain.graph.supervisor.agents import _apply_divergence_penalty
from src.domain.graph.supervisor.competition import (
    COMPETITION_GRAPH,
    build_competition_subgraph,
    compute_divergence,
)

from .conftest import CONFIG, TRANSCRIPT, FakeWrapper, wire


def test_divergence_is_pure_arithmetic():
    proposals = [
        {"stance": "aggressive", "total_engineer_days": 100},
        {"stance": "conservative", "total_engineer_days": 300},
    ]
    d = compute_divergence(proposals)
    assert d["low"] == 100 and d["high"] == 300
    assert d["spread"] == 200
    assert d["ratio"] == pytest.approx(200 / 200)
    assert d["level"] == "high"


def test_divergence_levels_are_thresholded():
    close = compute_divergence([{"total_engineer_days": 100}, {"total_engineer_days": 110}])
    assert close["level"] == "low"
    mid = compute_divergence([{"total_engineer_days": 100}, {"total_engineer_days": 140}])
    assert mid["level"] == "medium"


def test_divergence_degrades_gracefully_with_fewer_than_two():
    assert compute_divergence([])["ratio"] == 0.0
    single = compute_divergence([{"total_engineer_days": 80}])
    assert single["low"] == single["high"] == 80
    assert single["ratio"] == 0.0


@pytest.mark.asyncio
async def test_both_estimators_run_and_accumulate_proposals(monkeypatch):
    wire(monkeypatch, wrapper=FakeWrapper(conservative_total=300, aggressive_total=100))

    result = await COMPETITION_GRAPH.ainvoke({"brief": "components + references"})

    stances = sorted(p["stance"] for p in result["proposals"])
    assert stances == ["aggressive", "conservative"]


@pytest.mark.asyncio
async def test_synthesizer_returns_a_range_not_an_average(monkeypatch):
    wire(monkeypatch, wrapper=FakeWrapper(conservative_total=300, aggressive_total=100))

    result = await COMPETITION_GRAPH.ainvoke({"brief": "components + references"})

    synthesis = result["synthesis"]
    midpoint = (100 + 300) / 2
    assert synthesis["low"] == 100 and synthesis["high"] == 300
    assert synthesis["low"] != midpoint and synthesis["high"] != midpoint
    assert result["divergence"]["level"] == "high"


def test_subgraph_compiles_with_the_parallel_topology():
    graph = build_competition_subgraph()
    nodes = set(graph.get_graph().nodes)
    assert {"conservative_estimator", "aggressive_estimator", "synthesizer"} <= nodes


def test_high_divergence_pulls_confidence_down():
    settings = get_settings()
    base = 0.9
    low_div = _apply_divergence_penalty(base, {"ratio": 0.0})
    high_div = _apply_divergence_penalty(base, {"ratio": 1.0})
    assert low_div == base
    assert high_div == pytest.approx(base - settings.supervisor_divergence_penalty)
    assert high_div < base


@pytest.mark.asyncio
async def test_competitive_graph_trips_the_gate_on_divergence(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    from src.domain.graph.supervisor.build import build_supervisor_graph

    wrapper = FakeWrapper(
        route_script=[
            "requirements_extractor",
            "budget_searcher",
            "estimate_generator",
            "coherence_validator",
            "finish",
        ],
        conservative_total=400,
        aggressive_total=100,
    )
    wire(
        monkeypatch,
        wrapper=wrapper,
        hours_by_component={"API": [80, 88], "App": [80, 88]},
    )

    graph = build_supervisor_graph(MemorySaver(), competitive=True)
    await graph.ainvoke({"transcript": TRANSCRIPT, "estimation_id": "s14-test"}, CONFIG)
    snapshot = await graph.aget_state(CONFIG)

    assert snapshot.next == ("human_review_gate",)
    payload = snapshot.interrupts[0].value
    assert payload["gate"] == "low_confidence_review"
    values = snapshot.values
    assert values["divergence"]["level"] == "high"
    assert values["estimate"]["range"] == {"low": 100, "high": 400}
