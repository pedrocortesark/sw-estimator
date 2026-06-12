"""Alembic migration environment for the persistence layer (Session 8).

The DB URL is read from ``src.core.config.get_settings()`` (not from
alembic.ini) so the container, the dev host and CI all use the same source of
truth. Migrations are discovered by importing the model modules: every
SQLAlchemy model registered against ``Base.metadata`` becomes visible to
Alembic's autogenerate.
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import engine_from_config, pool
from sqlalchemy.dialects.postgresql.base import ischema_names

from src.core.config import get_settings
from src.persistence.models import Base  # noqa: F401 — ensure Base is registered
import src.rag.store.models  # noqa: F401 — register S8 tables on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The canonical Settings.DATABASE_URL uses the async driver (+asyncpg).
# Alembic is sync, so we derive the sync URL (+psycopg) here.
url = get_settings().database_url
if "+asyncpg" in url:
    sync_url = url.replace("+asyncpg", "+psycopg")
elif url.startswith("postgresql://"):
    sync_url = url.replace("postgresql://", "postgresql+psycopg://", 1)
else:
    sync_url = url
config.set_main_option("sqlalchemy.url", sync_url)

# Teach reflection about the ``vector`` column type. Without this,
# ``alembic check`` / autogenerate against a DB that already has vector
# columns cannot map them back and produces inconsistent diffs.
ischema_names["vector"] = Vector

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without a live connection — used by ``alembic upgrade --sql``."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
