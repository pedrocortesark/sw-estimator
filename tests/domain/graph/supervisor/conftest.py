"""Shared doubles for the Session 14 supervisor tests.

Every test in this package runs with NO network and NO API key: the LLM wrapper, the
retrieval backend and the consensus function are replaced at the same module-level
seams the production code imports them from.
"""

from __future__ import annotations

from statistics import mean

import pytest

from src.domain.graph.schemas import (
    ComponentClassification,
    ConsolidatedEstimate,
    EstimateProposal,
    RequirementsExtraction,
    SupervisorDecision,
    SynthesizedEstimate,
)

TRANSCRIPT = "A" * 200
CONFIG = {"configurable": {"thread_id": "s14-test"}}

FULL_ROUTE = [
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
    "finish",
]


class FakeWrapper:
    def __init__(
        self,
        *,
        route_script: list[str] | None = None,
        requirements: list[str] | None = None,
        components: list[tuple[str, str]] | None = None,
        estimate: dict | None = None,
        route_error: Exception | None = None,
        conservative_total: int = 120,
        aggressive_total: int = 100,
    ) -> None:
        self.route_script = list(route_script or [])
        self.requirements = requirements or ["req one", "req two"]
        self.components = components or [("API", "backend"), ("App", "mobile")]
        self.estimate = estimate
        self.route_error = route_error
        self.conservative_total = conservative_total
        self.aggressive_total = aggressive_total
        self.calls: list[str] = []

    def complete_structured(self, *, response_model, **kwargs):
        self.calls.append(response_model.__name__)

        if response_model is SupervisorDecision:
            if self.route_error is not None:
                raise self.route_error
            target = self.route_script.pop(0) if self.route_script else "finish"
            return (
                SupervisorDecision(
                    next_agent=target, reason=f"scripted route to {target}", confidence="high"
                ),
                {},
            )

        if response_model is RequirementsExtraction:
            return RequirementsExtraction(requirements=self.requirements), {}

        if response_model is ComponentClassification:
            return (
                ComponentClassification(
                    components=[{"name": n, "category": c} for n, c in self.components]
                ),
                {},
            )

        if response_model is ConsolidatedEstimate:
            payload = self.estimate or {
                "components": [
                    {"name": n, "engineer_days": 10, "rationale": "from references"}
                    for n, _ in self.components
                ],
                "total_engineer_days": 10 * len(self.components),
                "confidence": "high",
                "reasoning": "consolidated from the anchors",
            }
            return ConsolidatedEstimate(**payload), {}

        if response_model is EstimateProposal:
            system_prompt = kwargs.get("system_prompt", "")
            is_conservative = "RISK-FIRST" in system_prompt
            total = self.conservative_total if is_conservative else self.aggressive_total
            stance = "conservative" if is_conservative else "aggressive"
            return (
                EstimateProposal(
                    stance=stance,
                    total_engineer_days=total,
                    assumptions=[f"{stance} assumption"],
                    risks=[f"{stance} risk"],
                    reasoning=f"{stance} reasoning",
                ),
                {},
            )

        if response_model is SynthesizedEstimate:
            low, high = sorted((self.conservative_total, self.aggressive_total))
            return (
                SynthesizedEstimate(
                    low=low,
                    high=high,
                    driving_assumptions=["scope closure", "integration friction"],
                    open_questions=["Is the legacy interface documented?"],
                    confidence="low" if (high - low) > 0.3 * high else "medium",
                    reasoning="bracketed the two proposals; did not average",
                ),
                {},
            )

        raise AssertionError(f"unexpected response_model: {response_model!r}")


def backend_factory(hours_by_component: dict[str, list[int]]):
    def _make(*_args, **_kwargs):
        async def _backend(query, sectors=None):
            name = query.split(" (")[0]
            return [
                {
                    "id": i,
                    "content_preview": f"historical {name}",
                    "sector": "logistics",
                    "budget_id": f"BUD-{name[:3].upper()}-{i}",
                    "estimated_hours": float(hours),
                    "distance": 0.1 + 0.05 * i,
                }
                for i, hours in enumerate(hours_by_component.get(name, []))
            ]

        return _backend

    return _make


def fake_consensus(neighbors):
    if not neighbors:
        return 0, 0.0, 0.0
    return int(mean(h for h, _ in neighbors)), 0.85, 0.1


def wire(monkeypatch, *, wrapper, hours_by_component=None):
    monkeypatch.setattr("src.dependencies.get_llm_wrapper", lambda: wrapper)
    monkeypatch.setattr(
        "src.domain.graph.supervisor.agents.make_retrieval_backend",
        backend_factory(hours_by_component or {}),
    )
    monkeypatch.setattr(
        "src.domain.graph.supervisor.agents.distance_weighted_consensus", fake_consensus
    )


@pytest.fixture
def wrapper() -> FakeWrapper:
    return FakeWrapper()
