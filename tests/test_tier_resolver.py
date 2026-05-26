"""Unit tests for the tier resolver."""

from __future__ import annotations

import pytest

from src.schemas.estimation import EstimationResult, Phase, Task, TeamMember
from src.services.sessions import ProjectMetadata
from src.services.tier_resolver import resolve_tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(total_cost_usd: float, total_hours: float = 100.0) -> EstimationResult:
    """Build a minimal valid EstimationResult with the given cost."""
    cost_per_phase = round(total_cost_usd, 2)
    hours_per_phase = round(total_hours, 2)
    return EstimationResult(
        executive_summary="Test estimation.",
        phases=[
            Phase(
                name="Development",
                tasks=[Task(name="Implementation", hours=hours_per_phase, cost_usd=cost_per_phase)],
                total_hours=hours_per_phase,
                total_cost_usd=cost_per_phase,
            )
        ],
        total_hours=total_hours,
        total_cost_usd=total_cost_usd,
        team_composition=[TeamMember(role="Engineer", count=1, dedication="100%")],
        duration_weeks=4.0,
    )


def _meta(team_size: int | None = None) -> ProjectMetadata:
    return ProjectMetadata(assumed_team_size=team_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_enterprise_by_team_size() -> None:
    """team_size=10 triggers the large_team rule regardless of cost."""
    tier, rule = resolve_tier(_meta(team_size=10), _result(50_000))
    assert tier == "enterprise"
    assert rule == "large_team"


def test_enterprise_by_team_size_boundary() -> None:
    """Exactly 8 engineers is the minimum for the large_team rule."""
    tier, rule = resolve_tier(_meta(team_size=8), _result(10_000))
    assert tier == "enterprise"
    assert rule == "large_team"


def test_team_size_below_threshold_does_not_trigger_large_team() -> None:
    """team_size=7 must NOT trigger large_team."""
    tier, rule = resolve_tier(_meta(team_size=7), _result(100_000))
    # high_cost should win instead
    assert rule != "large_team"


def test_enterprise_by_cost() -> None:
    """cost > 85k triggers high_cost (team_size=2, so large_team does not fire)."""
    tier, rule = resolve_tier(_meta(team_size=2), _result(100_000))
    assert tier == "enterprise"
    assert rule == "high_cost"


def test_enterprise_by_cost_boundary() -> None:
    """cost of exactly 85_001 USD triggers high_cost."""
    tier, rule = resolve_tier(_meta(team_size=1), _result(85_001))
    assert tier == "enterprise"
    assert rule == "high_cost"


def test_standard_tier() -> None:
    """18k <= cost <= 85k with small team → standard."""
    tier, rule = resolve_tier(_meta(team_size=3), _result(40_000))
    assert tier == "standard"
    assert rule == "medium_cost"


def test_standard_tier_lower_boundary() -> None:
    """cost == 18_000 is the minimum for standard tier."""
    tier, rule = resolve_tier(_meta(team_size=1), _result(18_000))
    assert tier == "standard"
    assert rule == "medium_cost"


def test_standard_tier_upper_boundary() -> None:
    """cost == 85_000 is still within standard (rule uses <=)."""
    tier, rule = resolve_tier(_meta(team_size=1), _result(85_000))
    assert tier == "standard"
    assert rule == "medium_cost"


def test_starter_tier() -> None:
    """cost < 18k → starter."""
    tier, rule = resolve_tier(_meta(team_size=None), _result(10_000))
    assert tier == "starter"
    assert rule == "small_scope"


def test_starter_tier_boundary() -> None:
    """cost == 17_999 is just below the standard threshold → starter."""
    tier, rule = resolve_tier(_meta(team_size=1), _result(17_999))
    assert tier == "starter"
    assert rule == "small_scope"


def test_no_match_returns_unknown() -> None:
    """When no rule matches (simulated by emptying the rule table),
    resolve_tier must return ("unknown", "no_match")."""
    from unittest.mock import patch

    import src.services.tier_resolver as tr

    with patch.object(tr, "TIER_RULES", []):
        tier, rule = resolve_tier(_meta(team_size=None), _result(40_000))

    assert tier == "unknown"
    assert rule == "no_match"


def test_large_team_takes_priority_over_high_cost() -> None:
    """large_team is listed before high_cost, so it wins when both conditions
    are true."""
    tier, rule = resolve_tier(_meta(team_size=10), _result(200_000))
    assert rule == "large_team"
    assert tier == "enterprise"


def test_none_team_size_skips_large_team_rule() -> None:
    """When assumed_team_size is None the large_team predicate must not match."""
    tier, rule = resolve_tier(_meta(team_size=None), _result(40_000))
    assert rule != "large_team"
    assert tier == "standard"
