"""Exact-match key-value cache for estimation results."""

from __future__ import annotations

from src.schemas.estimation import EstimationRequest, EstimationResult


class EstimationCache:
    """Simple key-value cache for estimation results.

    Get and set operations are no-ops until a backing store is wired in.
    """

    def get(self, request: EstimationRequest) -> EstimationResult | None:
        """Return a cached result for *request*, or ``None`` on a miss."""
        return None

    def set(self, request: EstimationRequest, result: EstimationResult) -> None:
        """Store *result* under the key derived from *request*."""
        return None
