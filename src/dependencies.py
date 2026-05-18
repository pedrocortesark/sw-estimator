"""FastAPI dependency providers for the estimation pipeline."""

from __future__ import annotations

from src.services.estimation import EstimationService


def get_estimation_service() -> EstimationService:
    """Return a default ``EstimationService`` with no cache or moderation client.

    In production this will be extended to wire in the Redis-backed semantic
    cache and the OpenAI client for moderation.  For now it is stateless and
    safe to create per-request.
    """
    return EstimationService()
