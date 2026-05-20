from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings
from src.core.exceptions import setup_exception_handlers
from src.core.logging import logger, configure_logging
from src.routers import health, estimation, sessions


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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5000",   # dotnet run (Blazor dev)
            "http://localhost:5014",   # dotnet run (Blazor dev — dynamic port)
            "http://localhost:8080",   # Docker blazor service
            "http://127.0.0.1:5000",
            "http://127.0.0.1:5014",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(estimation.router)

    setup_exception_handlers(app)

    return app


# Global instance used by uvicorn to start the server:
# uvicorn src.main:app
app = create_app()
