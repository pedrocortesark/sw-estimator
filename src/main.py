from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import get_settings
from src.core.exceptions import setup_exception_handlers
from src.core.logging import logger, configure_logging
from src.routers import health, estimation, sessions, embeddings, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Code BEFORE `yield` runs at server startup.
    Code AFTER `yield` runs at server shutdown.

    This is the correct place to initialise resources (logger, HTTP clients,
    DB connections, etc.) and clean them up on exit.
    """
    # --- Startup ---
    configure_logging()
    settings = get_settings()
    logger.info(
        "app_started",
        env=settings.app_env,
        log_level=settings.log_level,
    )

    yield  # <-- FastAPI serves requests here

    # --- Shutdown ---
    logger.info("Shutting down sw-estimator...")


def create_app() -> FastAPI:
    """Factory function that creates and configures the FastAPI instance.

    Using a factory instead of a module-level global makes testing much easier:
    each test can spin up its own clean app instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="SW Estimator",
        description="Estimates software development effort from meeting transcriptions using LLMs (CAG architecture).",
        version="0.1.0",
        lifespan=lifespan,
        # Hide API docs in production
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
    )

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(estimation.router)
    app.include_router(embeddings.router)
    app.include_router(search.router)

    setup_exception_handlers(app)

    return app


# Global instance used by uvicorn to start the server:
# uvicorn src.main:app
app = create_app()
