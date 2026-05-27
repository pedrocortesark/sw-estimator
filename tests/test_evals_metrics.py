"""Unit tests for evals/metrics.py.

All tests are pure unit tests — no LLM calls, no HTTP, no Redis.
They build minimal response dicts that satisfy (or deliberately violate)
the EstimationResponse Pydantic model to verify each metric independently.
"""

from __future__ import annotations

import pytest

from evals.metrics import CostBoundsMetric, ContentRecallMetric, SchemaAdherenceMetric


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_task(name: str, hours: float, cost_usd: float) -> dict:
    return {"name": name, "hours": hours, "cost_usd": cost_usd}


def _make_phase(name: str, tasks: list[dict]) -> dict:
    return {
        "name": name,
        "tasks": tasks,
        "total_hours": sum(t["hours"] for t in tasks),
        "total_cost_usd": sum(t["cost_usd"] for t in tasks),
    }


def _make_valid_response(
    executive_summary: str = "This is a well-scoped project with clear requirements.",
    total_cost_usd: float = 50000.0,
    total_hours: float = 500.0,
    confidence_pct: float = 90.0,
    phase_name: str = "Backend Development",
) -> dict:
    """Build a minimal valid EstimationResponse dict.

    The single phase contains two tasks whose sums match the phase totals,
    and the phase total matches the grand total.
    """
    tasks = [
        _make_task("API design", total_hours * 0.4, total_cost_usd * 0.4),
        _make_task("Implementation", total_hours * 0.6, total_cost_usd * 0.6),
    ]
    phase = _make_phase(phase_name, tasks)
    return {
        "estimation": {
            "executive_summary": executive_summary,
            "phases": [phase],
            "total_hours": total_hours,
            "total_cost_usd": total_cost_usd,
            "team_composition": [
                {"role": "Backend Engineer", "count": 2, "dedication": "100%"}
            ],
            "duration_weeks": 12.0,
            "confidence_pct": confidence_pct,
        },
        "provider_used": "openai",
        "model_used": "gpt-4o-mini",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "cost_usd": 0.05,
        },
        "cached": False,
        "prompt_version": "v1",
    }


# ---------------------------------------------------------------------------
# SchemaAdherenceMetric
# ---------------------------------------------------------------------------


class TestSchemaAdherenceMetric:
    metric = SchemaAdherenceMetric()

    def test_valid_response_passes(self):
        response = _make_valid_response()
        assert self.metric.score(response) is True

    def test_missing_required_field_fails(self):
        response = _make_valid_response()
        del response["estimation"]["executive_summary"]
        assert self.metric.score(response) is False

    def test_phase_subtotal_mismatch_fails(self):
        """Phase total_hours does not match sum of task hours."""
        response = _make_valid_response()
        # Corrupt the phase total to be 50 % off
        response["estimation"]["phases"][0]["total_hours"] *= 1.5
        assert self.metric.score(response) is False

    def test_grand_total_mismatch_fails(self):
        """Grand total_hours doesn't match the sum of phase totals."""
        response = _make_valid_response()
        # Phase is correct but grand total is off
        response["estimation"]["total_hours"] *= 2
        assert self.metric.score(response) is False

    def test_empty_dict_fails(self):
        assert self.metric.score({}) is False


# ---------------------------------------------------------------------------
# CostBoundsMetric
# ---------------------------------------------------------------------------


class TestCostBoundsMetric:
    metric = CostBoundsMetric()

    def test_cost_within_bounds_passes(self):
        response = _make_valid_response(total_cost_usd=50000.0)
        expected = {"total_cost_usd_min": 18000, "total_cost_usd_max": 85000}
        assert self.metric.score(response, expected) is True

    def test_cost_below_min_fails(self):
        response = _make_valid_response(total_cost_usd=5000.0)
        expected = {"total_cost_usd_min": 18000, "total_cost_usd_max": 85000}
        assert self.metric.score(response, expected) is False

    def test_cost_above_max_fails(self):
        response = _make_valid_response(total_cost_usd=200000.0)
        expected = {"total_cost_usd_min": 18000, "total_cost_usd_max": 85000}
        assert self.metric.score(response, expected) is False

    def test_cost_exactly_at_boundary_passes(self):
        response = _make_valid_response(total_cost_usd=18000.0)
        expected = {"total_cost_usd_min": 18000, "total_cost_usd_max": 85000}
        assert self.metric.score(response, expected) is True

    def test_missing_estimation_key_fails(self):
        response = {"provider_used": "openai"}
        expected = {"total_cost_usd_min": 0, "total_cost_usd_max": 100000}
        assert self.metric.score(response, expected) is False


# ---------------------------------------------------------------------------
# ContentRecallMetric
# ---------------------------------------------------------------------------


class TestContentRecallMetric:
    metric = ContentRecallMetric()

    def test_all_keywords_in_summary_passes(self):
        response = _make_valid_response(
            executive_summary="This project requires Kafka, Spark and PostgreSQL."
        )
        expected = {"required_keywords": ["Kafka", "Spark"]}
        assert self.metric.score(response, expected) is True

    def test_keyword_in_phase_name_passes(self):
        response = _make_valid_response(phase_name="Kafka Integration Layer")
        expected = {"required_keywords": ["Kafka"]}
        assert self.metric.score(response, expected) is True

    def test_missing_keyword_fails(self):
        response = _make_valid_response(executive_summary="Simple CRUD application.")
        expected = {"required_keywords": ["Kubernetes"]}
        assert self.metric.score(response, expected) is False

    def test_empty_keywords_always_passes(self):
        response = _make_valid_response(executive_summary="Ambiguous project description.")
        expected = {"required_keywords": []}
        assert self.metric.score(response, expected) is True

    def test_case_insensitive_match(self):
        response = _make_valid_response(executive_summary="Uses kafka for streaming.")
        expected = {"required_keywords": ["Kafka"]}
        assert self.metric.score(response, expected) is True
