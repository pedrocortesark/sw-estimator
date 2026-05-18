"""High-level EstimationService — orchestrates guardrails, cache, prompts, and LLM."""

from __future__ import annotations

from src.schemas.estimation import EstimationRequest, EstimationResponse


class EstimationService:
    """Orchestrates the full estimation pipeline."""

    async def estimate(self, request: EstimationRequest) -> EstimationResponse:
        """Run the estimation pipeline and return a structured response."""
        raise NotImplementedError
