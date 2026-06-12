"""SQLAlchemy declarative base for the persistence layer.

Single ``Base`` — picked up by Alembic env.py. Concrete ORM models live next
to the subsystem that owns the table (e.g. ``src/rag/store/models.py`` for
the vector store) and are imported by ``env.py`` to register against this
``metadata``.
"""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Single declarative base — picked up by Alembic env.py."""
