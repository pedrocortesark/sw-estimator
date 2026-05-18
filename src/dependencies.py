"""FastAPI dependency providers for the estimation pipeline."""

from __future__ import annotations

from src.services.estimation import EstimationService


def get_estimation_service() -> EstimationService:
    """Provide an ``EstimationService`` instance to route handlers."""
    raise NotImplementedError
