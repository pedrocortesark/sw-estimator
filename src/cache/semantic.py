"""Semantic cache — return a cached EstimationResult when the input is semantically
similar to a previous request, avoiding redundant LLM calls."""

from __future__ import annotations

from src.schemas.estimation import EstimationRequest, EstimationResult


class EstimationSemanticCache:
    """Semantic similarity-based cache for estimation results.

    Lookup and store operations are no-ops until a vector store is wired in.
    """

    def lookup(self, request: EstimationRequest) -> EstimationResult | None:
        """Return a cached result if a semantically similar request exists."""
        return None

    def store(self, request: EstimationRequest, result: EstimationResult) -> None:
        """Persist a result so it can be retrieved by future similar requests."""
        return None
