"""SQLAlchemy async engine and session factory for the RAG store (Session 8).

The canonical ``Settings.DATABASE_URL`` uses the async driver
(``postgresql+asyncpg://``) because every RAG endpoint (``POST /embeddings/ingest``
and ``POST /search``) is on the real-time request path and must not block the
event loop on Postgres I/O.

Alembic migrations, on the other hand, are a sync, one-shot admin operation —
they run with a separate engine built from the sync driver
(``postgresql+psycopg://``) derived from the same canonical URL in
``alembic/env.py``. Single env var, two stacks, no duplication.
"""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from src.core.config import get_settings


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    """Build the global async engine from ``Settings.database_url`` (singleton)."""
    return create_async_engine(
        get_settings().database_url,
        pool_pre_ping=True,
    )


@lru_cache
def get_async_session_factory() -> async_sessionmaker:
    """Session factory for the async stack. ``expire_on_commit=False`` so ORM
    objects (e.g. the freshly persisted document id) stay readable after the
    transaction commits."""
    return async_sessionmaker(
        bind=create_async_engine_from_settings(),
        autoflush=False,
        expire_on_commit=False,
    )
