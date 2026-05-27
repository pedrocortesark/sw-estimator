"""Tier resolver — maps a session's metadata + estimation result to a named tier.

Tiers represent the complexity/scale bracket of an estimated project.  They
are resolved heuristically by evaluating a prioritised list of rules against
the current ProjectMetadata and the latest EstimationResult.  The first rule
whose predicate returns True wins.
"""

from __future__ import annotations

from typing import Callable

from src.schemas.estimation import EstimationResult
from src.services.sessions import ProjectMetadata

# ---------------------------------------------------------------------------
# Rule table — evaluated top-to-bottom; first match wins
# ---------------------------------------------------------------------------

TIER_RULES: list[dict] = [
    {
        "name": "large_team",
        "tier": "enterprise",
        "predicate": lambda meta, result: (
            meta.assumed_team_size is not None and meta.assumed_team_size >= 8
        ),
    },
    {
        "name": "high_cost",
        "tier": "enterprise",
        "predicate": lambda meta, result: result.total_cost_usd > 85_000,
    },
    {
        "name": "medium_cost",
        "tier": "standard",
        "predicate": lambda meta, result: 18_000 <= result.total_cost_usd <= 85_000,
    },
    {
        "name": "small_scope",
        "tier": "starter",
        "predicate": lambda meta, result: result.total_cost_usd < 18_000,
    },
]


def resolve_tier(
    metadata: ProjectMetadata,
    result: EstimationResult,
) -> tuple[str, str]:
    """Evaluate the tier rules and return the first matching (tier, rule_name).

    Rules are evaluated in declaration order.  The first rule whose predicate
    returns ``True`` is used; subsequent rules are skipped.

    Args:
        metadata: Accumulated project metadata for the session.
        result: The latest EstimationResult from the LLM.

    Returns:
        A ``(tier, rule_name)`` tuple.  Returns ``("unknown", "no_match")``
        when no rule matches.
    """
    for rule in TIER_RULES:
        predicate: Callable[[ProjectMetadata, EstimationResult], bool] = rule["predicate"]
        try:
            if predicate(metadata, result):
                return rule["tier"], rule["name"]
        except Exception:
            # Defensive: a buggy predicate must never crash the request.
            continue

    return "unknown", "no_match"
