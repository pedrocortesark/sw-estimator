"""Unit tests for evals/stress/metrics.py.

Coverage
--------
For each of the three stress metrics we test:

  1. A passing case (score 1.0, passed=True).
  2. A failing case (score 0.0, passed=False).
  3. A boundary / edge case (at-threshold, None input, empty session, …).

The tests are fully deterministic — no LLM calls, no network, no filesystem.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals.metrics import MetricResult
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    *,
    summary: str = "",
    anchors: list[str] | None = None,
    project_name: str | None = None,
    team_size: int | None = None,
    technologies: list[str] | None = None,
    agreed_scope: str | None = None,
) -> SimpleNamespace:
    """Build a minimal duck-typed Session proxy."""
    meta = SimpleNamespace(
        project_name=project_name,
        assumed_team_size=team_size,
        mentioned_technologies=technologies or [],
        agreed_scope=agreed_scope,
    )
    return SimpleNamespace(
        accumulated_summary=summary,
        anchors=anchors or [],
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# LatencyBudgetMetric
# ---------------------------------------------------------------------------


class TestLatencyBudgetMetric:
    def test_passes_when_within_budget(self) -> None:
        metric = LatencyBudgetMetric(budget_ms=3000)
        result = metric.evaluate({"latency_ms": 2400.0})

        assert isinstance(result, MetricResult)
        assert result.name == "latency_budget"
        assert result.score == 1.0
        assert result.passed is True
        assert "2400" in result.details or "2400.0" in result.details

    def test_fails_when_over_budget(self) -> None:
        metric = LatencyBudgetMetric(budget_ms=3000)
        result = metric.evaluate({"latency_ms": 5200.0})

        assert result.score == 0.0
        assert result.passed is False
        assert "5200" in result.details or "5200.0" in result.details

    def test_boundary_exactly_at_budget(self) -> None:
        """Exactly at the threshold counts as passing (≤)."""
        metric = LatencyBudgetMetric(budget_ms=3000)
        result = metric.evaluate({"latency_ms": 3000.0})

        assert result.passed is True
        assert result.score == 1.0

    def test_missing_field_returns_failed(self) -> None:
        metric = LatencyBudgetMetric(budget_ms=3000)
        result = metric.evaluate({})

        assert result.passed is False
        assert result.score == 0.0
        assert "not found" in result.details

    def test_accepts_namespace_observation(self) -> None:
        """Duck-typed access: works with SimpleNamespace, not just dicts."""
        metric = LatencyBudgetMetric(budget_ms=3000)
        obs = SimpleNamespace(latency_ms=1500.0)
        result = metric.evaluate(obs)

        assert result.passed is True

    def test_constructor_rejects_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="budget_ms must be positive"):
            LatencyBudgetMetric(budget_ms=0)


# ---------------------------------------------------------------------------
# CostBudgetMetric
# ---------------------------------------------------------------------------


class TestCostBudgetMetric:
    def test_passes_when_within_budget(self) -> None:
        metric = CostBudgetMetric(budget_usd=0.01)
        result = metric.evaluate({"cost_usd": 0.0031})

        assert isinstance(result, MetricResult)
        assert result.name == "cost_budget"
        assert result.score == 1.0
        assert result.passed is True

    def test_fails_when_over_budget(self) -> None:
        metric = CostBudgetMetric(budget_usd=0.005)
        result = metric.evaluate({"cost_usd": 0.0123})

        assert result.score == 0.0
        assert result.passed is False

    def test_boundary_exactly_at_budget(self) -> None:
        """Exactly at the threshold counts as passing (≤)."""
        metric = CostBudgetMetric(budget_usd=0.01)
        result = metric.evaluate({"cost_usd": 0.01})

        assert result.passed is True

    def test_zero_cost_always_passes(self) -> None:
        """Free (cached) turns should always pass the cost budget."""
        metric = CostBudgetMetric(budget_usd=0.01)
        result = metric.evaluate({"cost_usd": 0.0})

        assert result.passed is True

    def test_missing_field_returns_failed(self) -> None:
        metric = CostBudgetMetric(budget_usd=0.01)
        result = metric.evaluate({})

        assert result.passed is False
        assert "not found" in result.details

    def test_accepts_namespace_observation(self) -> None:
        metric = CostBudgetMetric(budget_usd=0.01)
        obs = SimpleNamespace(cost_usd=0.0005)
        result = metric.evaluate(obs)

        assert result.passed is True

    def test_constructor_rejects_negative_budget(self) -> None:
        with pytest.raises(ValueError, match="budget_usd must be non-negative"):
            CostBudgetMetric(budget_usd=-0.001)


# ---------------------------------------------------------------------------
# MemoryDriftMetric
# ---------------------------------------------------------------------------


class TestMemoryDriftMetric:
    def test_passes_when_fact_in_summary(self) -> None:
        session = _make_session(summary="The project is called Nimbus and uses Flutter.")
        metric = MemoryDriftMetric(fact="Nimbus")
        result = metric.evaluate(session)

        assert isinstance(result, MetricResult)
        assert result.name == "memory_drift"
        assert result.score == 1.0
        assert result.passed is True
        assert "nimbus" in result.details.lower() or "found" in result.details.lower()

    def test_fails_when_fact_not_in_session(self) -> None:
        session = _make_session(summary="Project Orion, stack: FastAPI, React.")
        metric = MemoryDriftMetric(fact="Flutter")
        result = metric.evaluate(session)

        assert result.score == 0.0
        assert result.passed is False

    def test_case_insensitive_match(self) -> None:
        session = _make_session(summary="Budget locked: 30000 EUR")
        metric = MemoryDriftMetric(fact="budget locked: 30000 EUR")
        result = metric.evaluate(session)

        assert result.passed is True

    def test_finds_fact_in_anchors(self) -> None:
        session = _make_session(
            summary="General project overview.",
            anchors=["stack includes Flutter", "team: 4 engineers"],
        )
        metric = MemoryDriftMetric(fact="Flutter", where=["anchors"])
        result = metric.evaluate(session)

        assert result.passed is True

    def test_finds_fact_in_metadata_project_name(self) -> None:
        session = _make_session(project_name="Nimbus")
        metric = MemoryDriftMetric(fact="Nimbus", where=["metadata"])
        result = metric.evaluate(session)

        assert result.passed is True

    def test_finds_fact_in_metadata_technologies(self) -> None:
        session = _make_session(technologies=["FastAPI", "React", "PostgreSQL"])
        metric = MemoryDriftMetric(fact="PostgreSQL", where=["metadata"])
        result = metric.evaluate(session)

        assert result.passed is True

    def test_fails_when_restricted_to_wrong_where(self) -> None:
        """Fact is in summary but where=["anchors"] — should not match."""
        session = _make_session(
            summary="Project Nimbus with FastAPI",
            anchors=[],
        )
        metric = MemoryDriftMetric(fact="Nimbus", where=["anchors"])
        result = metric.evaluate(session)

        assert result.passed is False

    def test_empty_session_always_fails(self) -> None:
        session = _make_session()
        metric = MemoryDriftMetric(fact="anything")
        result = metric.evaluate(session)

        assert result.passed is False

    def test_unknown_where_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            MemoryDriftMetric(fact="text", where=["invalid_field"])

    def test_default_where_searches_all_fields(self) -> None:
        """With default where (None → all three), any field match suffices."""
        session = _make_session(
            summary="",
            anchors=[],
            project_name="Nimbus",
        )
        metric = MemoryDriftMetric(fact="Nimbus")  # where defaults to all
        result = metric.evaluate(session)

        assert result.passed is True

    def test_boundary_fact_as_substring_of_longer_value(self) -> None:
        """Partial substring match: fact='Fast' should match 'FastAPI'."""
        session = _make_session(technologies=["FastAPI"])
        metric = MemoryDriftMetric(fact="Fast", where=["metadata"])
        result = metric.evaluate(session)

        assert result.passed is True

    def test_accepts_dict_session(self) -> None:
        """MemoryDriftMetric must accept a plain dict as session snapshot."""
        session_dict = {
            "accumulated_summary": "Budget: 30000 EUR",
            "anchors": [],
            "metadata": {
                "project_name": None,
                "assumed_team_size": None,
                "mentioned_technologies": [],
                "agreed_scope": None,
            },
        }
        metric = MemoryDriftMetric(fact="30000 EUR", where=["summary"])
        result = metric.evaluate(session_dict)

        assert result.passed is True
