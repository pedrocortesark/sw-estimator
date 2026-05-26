"""Evaluation metrics for the sw-estimator golden dataset.

Three independent, stateless metrics are provided:

- SchemaAdherenceMetric  — validates that a raw response dict passes
  full Pydantic validation of EstimationResponse (including all
  model_validators on totals, phase subtotals and confidence).

- CostBoundsMetric       — checks that total_cost_usd falls within
  the [min, max] range declared in the golden case.

- ContentRecallMetric    — checks that all required keywords appear
  somewhere in the executive_summary or in any phase name (case-
  insensitive substring match).
"""

from __future__ import annotations

from pydantic import ValidationError

from src.schemas.estimation import EstimationResponse


class SchemaAdherenceMetric:
    """Pass/fail metric: the response dict is a valid EstimationResponse.

    This catches:
    - Missing or wrongly-typed required fields
    - Phase subtotals that deviate > 5 % from the sum of tasks
    - Grand totals that deviate > 5 % from the sum of phases
    - Low-confidence responses whose executive_summary lacks the
      mandatory ``Out of scope:`` prefix
    """

    name = "schema_adherence"

    def score(self, response: dict) -> bool:
        """Return True if *response* passes full EstimationResponse validation.

        Args:
            response: Raw dict as returned by ``EstimationResponse.model_dump()``.

        Returns:
            True when validation succeeds, False on any ValidationError.
        """
        try:
            EstimationResponse(**response)
            return True
        except (ValidationError, TypeError):
            return False


class CostBoundsMetric:
    """Pass/fail metric: total_cost_usd is within the expected range."""

    name = "cost_bounds"

    def score(self, response: dict, expected: dict) -> bool:
        """Return True if the response cost falls within [min, max].

        Args:
            response: Raw dict with an ``estimation`` key containing
                      the EstimationResult fields.
            expected: Golden-case ``expected`` dict with keys
                      ``total_cost_usd_min`` and ``total_cost_usd_max``.

        Returns:
            True when the cost is within bounds, False otherwise.
        """
        try:
            cost = response["estimation"]["total_cost_usd"]
            return expected["total_cost_usd_min"] <= cost <= expected["total_cost_usd_max"]
        except (KeyError, TypeError):
            return False


class ContentRecallMetric:
    """Pass/fail metric: all required keywords appear in the response text."""

    name = "content_recall"

    def score(self, response: dict, expected: dict) -> bool:
        """Return True if every required keyword is found in the response.

        The search corpus is the union of:
        - ``estimation.executive_summary``
        - all phase names in ``estimation.phases``

        Matching is case-insensitive substring.

        Args:
            response: Raw dict with an ``estimation`` key.
            expected: Golden-case ``expected`` dict with a
                      ``required_keywords`` list (may be empty).

        Returns:
            True when all keywords are found (or the list is empty).
        """
        keywords: list[str] = expected.get("required_keywords", [])
        if not keywords:
            return True

        try:
            estimation = response["estimation"]
            corpus_parts: list[str] = [estimation.get("executive_summary", "")]
            for phase in estimation.get("phases", []):
                corpus_parts.append(phase.get("name", ""))
            corpus = " ".join(corpus_parts).lower()
        except (KeyError, TypeError, AttributeError):
            return False

        return all(kw.lower() in corpus for kw in keywords)
