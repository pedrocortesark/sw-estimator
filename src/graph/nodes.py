"""Nodes for the LangGraph estimation graph (Session 13).

Each node is a pure function that receives the state and returns a partial
update. Nodes are instrumented with logfire spans for observability.
"""

from __future__ import annotations

from typing import Any

import logfire

from src.graph.state import BudgetMatch, Component, EstimationState


def extract_requirements(state: EstimationState) -> dict[str, Any]:
    """Extract requirements from the transcript.

    TODO: Reuse your S9-S12 requirement extraction logic here.
    For now, returns a simple list of requirements.
    """
    with logfire.span("node: extract_requirements"):
        transcript = state["transcript"]

        # Simple extraction: split by sentences and filter
        # TODO: Use your actual requirement extraction logic
        sentences = [s.strip() for s in transcript.split(".") if len(s.strip()) > 20]
        requirements = sentences[:10]  # Take first 10 as requirements

        logfire.info(
            "extracted requirements",
            count=len(requirements),
            transcript_length=len(transcript),
        )

        return {"requirements": requirements}


def classify_components(state: EstimationState) -> dict[str, Any]:
    """Group requirements into components with categories.

    TODO: Reuse your S9-S12 component classification logic here.
    For now, creates simple components from requirements.
    """
    with logfire.span("node: classify_components"):
        requirements = state["requirements"]

        # Simple classification: group by keywords
        # TODO: Use your actual classification logic
        components: list[Component] = []

        # Group requirements into components
        component_keywords = {
            "Authentication": ["auth", "login", "security", "oauth"],
            "Backend": ["backend", "api", "server", "database"],
            "Frontend": ["frontend", "ui", "interface", "web"],
            "Integration": ["integration", "connect", "sync", "import"],
        }

        for req in requirements:
            req_lower = req.lower()
            categorized = False

            for category, keywords in component_keywords.items():
                if any(kw in req_lower for kw in keywords):
                    # Find or create component
                    existing = next((c for c in components if c["category"] == category), None)
                    if existing:
                        existing["requirements"].append(req)
                    else:
                        components.append(
                            {
                                "name": category,
                                "category": category,
                                "requirements": [req],
                            }
                        )
                    categorized = True
                    break

            if not categorized:
                # Default component for uncategorized requirements
                existing = next((c for c in components if c["category"] == "General"), None)
                if existing:
                    existing["requirements"].append(req)
                else:
                    components.append(
                        {
                            "name": "General",
                            "category": "General",
                            "requirements": [req],
                        }
                    )

        logfire.info("classified components", count=len(components))

        return {"components": components}


def search_budgets(state: EstimationState) -> dict[str, Any]:
    """Search for reference budgets for each component.

    Sequential for now (parallel in the live session).
    TODO: Reuse your S9-S12 budget retrieval logic here.
    """
    with logfire.span("node: search_budgets"):
        components = state["components"]
        matches: list[BudgetMatch] = []

        for component in components:
            # TODO: Use your actual budget retrieval logic
            # For now, create a mock match
            match: BudgetMatch = {
                "component": component["name"],
                "budget_id": f"BUDGET-{component['category']}-001",
                "amount": 10000.0 + (len(component["requirements"]) * 1000),
                "description": f"Reference budget for {component['category']}",
                "relevance_score": 0.85,
            }
            matches.append(match)

            logfire.info(
                "found budget match",
                component=component["name"],
                budget_id=match["budget_id"],
                amount=match["amount"],
            )

        return {"budget_matches": matches}


def generate_estimate(state: EstimationState) -> dict[str, Any]:
    """Generate estimation from budget matches.

    TODO: Reuse your S9-S12 estimation generation logic here.
    For now, creates a simple estimate from budget matches.
    """
    with logfire.span("node: generate_estimate"):
        budget_matches = state["budget_matches"]

        # Simple estimation: sum of budget amounts
        # TODO: Use your actual estimation logic
        total_amount = sum(match["amount"] for match in budget_matches)

        estimate = {
            "total_amount": total_amount,
            "currency": "EUR",
            "components": [
                {
                    "name": match["component"],
                    "amount": match["amount"],
                    "budget_id": match["budget_id"],
                }
                for match in budget_matches
            ],
            "methodology": "Budget-based estimation",
        }

        logfire.info("generated estimate", total_amount=total_amount)

        return {"estimate": estimate}


def validate_and_consolidate(state: EstimationState) -> dict[str, Any]:
    """Validate the estimate and set final status.

    TODO: Add your validation logic here.
    For now, always validates successfully.
    """
    with logfire.span("node: validate_and_consolidate"):
        estimate = state["estimate"]
        errors: list[str] = []

        # Simple validation: check estimate exists and has required fields
        if not estimate:
            errors.append("No estimate generated")
            return {"status": "needs_review", "errors": errors}

        if "total_amount" not in estimate:
            errors.append("Estimate missing total_amount")
            return {"status": "needs_review", "errors": errors}

        if estimate["total_amount"] <= 0:
            errors.append("Invalid total_amount")
            return {"status": "needs_review", "errors": errors}

        logfire.info("validation passed", status="validated")

        return {"status": "validated", "errors": errors}
