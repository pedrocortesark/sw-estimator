from contextlib import asynccontextmanager
from contextlib import AsyncExitStack

import logfire
import structlog
from fastapi import FastAPI

from src.core.config import get_settings
from src.core.exceptions import setup_exception_handlers
from src.core.logging import logger, configure_logging
from src.routers import health, estimation, sessions, embeddings, search
from src.api.routers import corpus_index, estimate_agent, estimate_graph, estimate_multi_agent

log = structlog.get_logger()


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

    # Configure Logfire for distributed tracing (Session 13)
    logfire.configure(
        service_name=settings.logfire_service_name,
        send_to_logfire="if-token-present",
    )
    logfire.instrument_fastapi(app)
    logfire.instrument_httpx()
    logfire.instrument_asyncpg()

    logger.info("logfire_configured", service_name=settings.logfire_service_name)

    # Session 13: build the estimation graph with a Postgres checkpointer over the
    # project database (its tables coexist with pgvector). Held open for the app's
    # lifetime via an AsyncExitStack; a failure here (e.g. Postgres unreachable)
    # leaves app.state.graph = None so the graph endpoint 503s WITHOUT taking down
    # the unrelated routers.
    app.state.graph = None
    app.state.multi_agent_graph = None
    app.state.supervisor_graph = None
    app.state._graph_stack = AsyncExitStack()
    try:
        from src.domain.graph.build import build_graph
        from src.domain.multi_agent.build import build_multi_agent_graph
        from src.domain.graph.supervisor.build import build_supervisor_graph
        from src.domain.graph.checkpointer import open_checkpointer

        checkpointer = await app.state._graph_stack.enter_async_context(open_checkpointer())
        app.state.graph = build_graph(checkpointer)
        app.state.multi_agent_graph = build_multi_agent_graph(checkpointer)
        app.state.supervisor_graph = build_supervisor_graph(
            checkpointer,
            competitive=settings.supervisor_competition_enabled,
            sandboxed=settings.supervisor_persistence_enabled,
        )
        log.info(
            "graphs_ready",
            supervisor_competitive=settings.supervisor_competition_enabled,
            supervisor_sandboxed=settings.supervisor_persistence_enabled,
        )
    except Exception as exc:  # noqa: BLE001 — the graph is optional infrastructure.
        log.error("graph_init_failed", error=str(exc)[:400])

    log.info("application_started", environment=settings.app_env)

    yield  # <-- FastAPI serves requests here

    # --- Shutdown ---
    await app.state._graph_stack.aclose()
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
    app.include_router(corpus_index.router)
    # Session 12 — hand-written agent over the budget retrieval (decision layer).
    app.include_router(estimate_agent.router)
    # Session 13 — LangGraph-based estimation pipeline.
    app.include_router(estimate_graph.router)
    # Session 14 — Multi-agent supervisor/workers with human-in-the-loop.
    app.include_router(estimate_multi_agent.router)

    setup_exception_handlers(app)

    return app


# Global instance used by uvicorn to start the server:
# uvicorn src.main:app
app = create_app()
