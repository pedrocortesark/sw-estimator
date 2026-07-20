"""Budget searcher agent.

Searches for historical budget references for each requirement.
This agent has access ONLY to the search_budgets tool.
"""

from __future__ import annotations

import logfire
import structlog

from src.core.config import get_settings
from src.domain.multi_agent.state import BudgetMatch, EstimationState
from src.generation.rag.agent_retrieval import make_retrieval_backend

log = structlog.get_logger()


async def budget_searcher(state: EstimationState) -> dict:
    """Search for budget references for each requirement (search_budgets tool only)."""
    with logfire.span("agent: budget_searcher"):
        settings = get_settings()
        backend = make_retrieval_backend(
            top_k=settings.agent_search_top_k,
            distance_threshold=settings.agent_search_distance_threshold,
        )

        requirements = state.get("requirements") or []
        matches: list[BudgetMatch] = []
        errors: list[str] = []

        for requirement in requirements:
            try:
                items = await backend(requirement, None)
                for item in items:
                    hours = item.get("estimated_hours")
                    if hours is None:
                        continue
                    matches.append(
                        {
                            "component": requirement[:50],  # Truncate for readability
                            "reference_budget_id": item.get("budget_id"),
                            "amount": float(hours),
                            "distance": float(item.get("distance", 0.0)),
                        }
                    )
            except Exception as exc:  # noqa: BLE001 — soft-fail one requirement
                log.warning(
                    "agent_budget_searcher_error",
                    requirement=requirement[:50],
                    error=str(exc)[:200],
                )
                errors.append(f"search failed for {requirement[:30]!r}: {exc}")

        log.info(
            "agent_budget_searcher_done",
            requirements=len(requirements),
            matches=len(matches),
            errors=len(errors),
        )

        # Audit log entry
        audit_entry = {
            "agent": "budget_searcher",
            "tool": "search_budgets",
            "input_summary": f"{len(requirements)} requirements",
            "output_summary": f"{len(matches)} matches found, {len(errors)} errors",
        }

        update: dict = {
            "budget_matches": matches,
            "agent_actions": [audit_entry],
        }

        if errors:
            # Append errors to the estimate for visibility
            update["estimate"] = {"search_errors": errors}

        return update
