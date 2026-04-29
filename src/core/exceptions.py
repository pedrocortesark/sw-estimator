"""Domain exception hierarchy for sw-estimator.

All exceptions raised by the service layer are subclasses of EstimatorError.
This keeps HTTP concerns out of the service layer and lets exception handlers
in main.py translate them to the appropriate HTTP responses.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from src.core.logging import logger


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class EstimatorError(Exception):
    """Base exception for all sw-estimator domain errors."""


class ProviderRateLimitError(EstimatorError):
    """The LLM provider returned a rate-limit or quota-exceeded error (HTTP 429)."""


class ProviderAuthError(EstimatorError):
    """The LLM provider rejected the request due to invalid credentials (HTTP 401)."""


class UnknownProviderError(EstimatorError):
    """An unsupported provider name was requested."""


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def setup_exception_handlers(app) -> None:
    """Register domain exception handlers on the FastAPI app.

    Each handler translates a domain exception into a JSONResponse with the
    appropriate HTTP status code, keeping routers free of error-mapping logic.
    """

    @app.exception_handler(ProviderRateLimitError)
    async def rate_limit_handler(
        request: Request, exc: ProviderRateLimitError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "LLM provider rate limit exceeded. Please try again later."
            },
        )

    @app.exception_handler(ProviderAuthError)
    async def auth_error_handler(
        request: Request, exc: ProviderAuthError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing LLM provider credentials."},
        )

    @app.exception_handler(UnknownProviderError)
    async def unknown_provider_handler(
        request: Request, exc: UnknownProviderError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(EstimatorError)
    async def estimator_error_handler(
        request: Request, exc: EstimatorError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An unexpected error occurred while generating the estimation. Please try again."
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An unexpected error occurred while generating the estimation. Please try again."
            },
        )
