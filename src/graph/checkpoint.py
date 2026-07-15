"""Checkpoint and observability configuration for LangGraph (Session 13).

Provides:
- AsyncPostgresSaver checkpointer for graph state persistence
- Logfire configuration for distributed tracing
- Helper to run checkpointer setup on startup
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import logfire
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.core.config import get_settings


def configure_observability():
    """Configure Logfire for distributed tracing.

    Call this once at application startup.
    """
    # Logfire will use LOGFIRE_TOKEN from environment if available
    # Otherwise, it runs in local mode
    logfire.configure(
        service_name="sw-estimator",
        send_to_logfire="if-token-present",
    )

    # Instrument common libraries
    logfire.instrument_httpx()
    logfire.instrument_asyncpg()

    # Note: instrument_fastapi() is called in main.py after app is created


def get_checkpointer() -> AsyncPostgresSaver:
    """Get the AsyncPostgresSaver checkpointer.

    Uses the same PostgreSQL database as the RAG store.
    The checkpointer creates its own tables (graph_checkpoints, etc.)
    and coexists with the existing tables.

    Returns:
        AsyncPostgresSaver instance ready for use.
    """
    settings = get_settings()

    # Convert async URL to sync URL for psycopg (checkpointer uses sync internally)
    database_url = settings.database_url
    
    # Replace +asyncpg with psycopg for sync connection
    if "+asyncpg" in database_url:
        database_url = database_url.replace("+asyncpg", "+psycopg")
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    # Create checkpointer instance
    checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
    
    return checkpointer


async def setup_checkpointer(checkpointer: AsyncPostgresSaver):
    """Run checkpointer setup to create tables.

    Call this once at application startup.
    Creates the necessary tables for graph state persistence.

    Args:
        checkpointer: AsyncPostgresSaver instance.
    """
    await checkpointer.setup()


@asynccontextmanager
async def lifespan_with_graph(app):
    """Lifespan context manager that sets up checkpointer.

    Use this in FastAPI's lifespan parameter.

    Example:
        @asynccontextmanager
        async def lifespan(app):
            async with lifespan_with_graph(app):
                yield
    """
    # Setup observability
    configure_observability()

    # Setup checkpointer
    checkpointer = get_checkpointer()
    await setup_checkpointer(checkpointer)

    logfire.info("graph checkpointer setup complete")

    yield

    # Cleanup if needed
    logfire.info("shutting down graph")
