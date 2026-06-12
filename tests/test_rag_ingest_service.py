"""Tests for src/rag/ingest_service.py — RagIngestService.

Strategy: replace every collaborator (chunker, embedder, session factory,
store) with fakes. The ingest service is a pure orchestrator, so the
test surface is the order in which it calls them and the data it hands off.

We never spin up a real Postgres here; the transactional semantics (rollback
on duplicate) are an emergent property of the service that the integration
smoke test exercises against a real DB.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.embedding.embedder import OpenAIEmbedder
from src.rag.ingest_service import DuplicateDocumentError, RagIngestService
from src.rag.schemas import Budget, BudgetComponent, Chunk, ClientMetadata, EmbeddedChunk
from src.rag.store.repository import ChunkStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_budget() -> Budget:
    return Budget(
        budget_id="BUD-2024-001",
        client_metadata=ClientMetadata(name="FintechCorp", sector="finance", country="ES"),
        project_summary="Mobile banking API",
        main_technology="ruby_on_rails",
        year=2024,
        total_estimated_hours=120,
        components=[
            BudgetComponent(
                component_id="AUTH-001",
                name="OAuth backend",
                description="OAuth 2.0 with JWT",
                tech_stack=["rails"],
                estimated_hours=120,
                complexity="high",
                dependencies=[],
            )
        ],
    )


def _make_chunk(text: str = "t") -> Chunk:
    return Chunk(chunk_id="b::c", text=text, metadata={}, token_count=1)


def _make_embedded(vector: list[float] | None = None) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id="b::c",
        text="t",
        metadata={},
        token_count=1,
        embedding=vector if vector is not None else [0.1, 0.2, 0.3],
    )


def _make_session_factory(execute_return: int | None = None):
    """Build a fake ``async_sessionmaker`` whose context manager yields an
    AsyncMock session. ``session.begin()`` is an awaitable that returns a
    sync context manager — matching what a real ``session.begin()`` does.

    ``execute_return`` is what ``session.execute(...).scalar_one_or_none()``
    returns; pass ``None`` for "no duplicate" and an int for "duplicate of
    document N".
    """
    session = AsyncMock()

    # session.begin() must return an async context manager directly
    # (not a coroutine). Replace the AsyncMock that ``AsyncMock(spec=...)``
    # auto-created with a regular MagicMock that returns the CM.
    begin_cm = MagicMock()
    begin_cm.__aenter__ = AsyncMock(return_value=None)
    begin_cm.__aexit__ = AsyncMock(return_value=None)
    session.begin = MagicMock(return_value=begin_cm)

    # Default: find_document_id returns None (no duplicate).
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = execute_return
    session.execute.return_value = execute_result

    # In SQLAlchemy 2.0 async, ``async with session_factory() as session``
    # is used directly — the factory itself implements ``__aenter__`` /
    # ``__aexit__``. So ``_factory()`` must return the session CM, not an
    # awaitable.
    factory = MagicMock()
    factory.__aenter__ = AsyncMock(return_value=session)
    factory.__aexit__ = AsyncMock(return_value=None)
    # Calling the factory must return the factory itself (not a coroutine).
    factory.side_effect = lambda: factory

    return factory, session


# ---------------------------------------------------------------------------
# Tests — duplicate guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_raises_duplicate_when_source_path_exists() -> None:
    """The 409 contract: if a document with this source_path is already in
    the DB, the service must raise ``DuplicateDocumentError`` BEFORE calling
    the embedder. The exception carries the existing document_id."""
    session_factory, session = _make_session_factory(execute_return=99)
    store = MagicMock(spec=ChunkStore)
    store.find_document_id = AsyncMock(return_value=99)
    chunker = MagicMock()
    embedder = MagicMock(spec=OpenAIEmbedder)

    service = RagIngestService(
        chunker=chunker,
        embedder=embedder,
        session_factory=session_factory,
        store=store,
    )

    with pytest.raises(DuplicateDocumentError) as exc_info:
        await service.ingest(
            source_path="data/x.json",
            document_type="historical_budget",
            budget=_make_budget(),
        )

    assert exc_info.value.document_id == 99
    # The expensive calls must NOT have happened.
    chunker.chunk.assert_not_called()
    embedder.embed_many.assert_not_called()
    store.persist_document_with_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_calls_chunker_then_embedder_then_store() -> None:
    """The three calls in order. If you ever add a fourth collaborator
    (e.g. an audit log) it must go AFTER persist, not between embed and
    persist, or you break the atomicity guarantee."""
    session_factory, _session = _make_session_factory(execute_return=None)
    store = MagicMock(spec=ChunkStore)
    store.find_document_id = AsyncMock(return_value=None)
    store.persist_document_with_chunks = AsyncMock(return_value=42)

    call_order: list[str] = []
    chunker = MagicMock()
    chunker.chunk.side_effect = lambda budgets: (call_order.append("chunk"), _make_chunk())[1]
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_many.side_effect = (
        lambda chunks: call_order.append("embed") or [_make_embedded()]
    )
    store.persist_document_with_chunks.side_effect = (
        lambda *a, **kw: call_order.append("persist") or 42
    )

    service = RagIngestService(
        chunker=chunker, embedder=embedder,
        session_factory=session_factory, store=store,
    )
    await service.ingest(
        source_path="data/x.json",
        document_type="historical_budget",
        budget=_make_budget(),
    )

    assert call_order == ["chunk", "embed", "persist"]


@pytest.mark.asyncio
async def test_ingest_response_carries_document_id_from_store() -> None:
    """The HTTP response must echo the document_id the store assigned.
    Losing it would break the duplicate-check on the next call (and the
    audit trail of who ingested what)."""
    session_factory, _session = _make_session_factory(execute_return=None)
    store = MagicMock(spec=ChunkStore)
    store.find_document_id = AsyncMock(return_value=None)
    store.persist_document_with_chunks = AsyncMock(return_value=7)

    chunker = MagicMock()
    chunker.chunk.return_value = [_make_chunk()]
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_many.return_value = [_make_embedded([0.1] * 4)]

    service = RagIngestService(
        chunker=chunker, embedder=embedder,
        session_factory=session_factory, store=store,
    )
    response = await service.ingest(
        source_path="data/x.json",
        document_type="historical_budget",
        budget=_make_budget(),
    )

    assert response.document_id == 7


@pytest.mark.asyncio
async def test_ingest_response_reports_chunk_count_and_embedding_dim() -> None:
    """``chunks_created`` and ``embedding_dimension`` are observable metrics
    for ops. If a chunk arrives without an embedding (impossible in
    practice but the type allows it), dimension must be 0, not crash."""
    session_factory, _session = _make_session_factory(execute_return=None)
    store = MagicMock(spec=ChunkStore)
    store.find_document_id = AsyncMock(return_value=None)
    store.persist_document_with_chunks = AsyncMock(return_value=1)

    chunker = MagicMock()
    chunker.chunk.return_value = [_make_chunk(), _make_chunk("second"), _make_chunk("third")]
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_many.return_value = [
        _make_embedded([0.1] * 8),
        _make_embedded([0.1] * 8),
        _make_embedded([0.1] * 8),
    ]

    service = RagIngestService(
        chunker=chunker, embedder=embedder,
        session_factory=session_factory, store=store,
    )
    response = await service.ingest(
        source_path="data/x.json",
        document_type="historical_budget",
        budget=_make_budget(),
    )

    assert response.chunks_created == 3
    assert response.embedding_dimension == 8


@pytest.mark.asyncio
async def test_ingest_response_dim_is_zero_when_no_chunks() -> None:
    """Edge case: a budget with zero components. Unusual (the schema
    enforces ``min_length=1``) but the service must not crash on it."""
    session_factory, _session = _make_session_factory(execute_return=None)
    store = MagicMock(spec=ChunkStore)
    store.find_document_id = AsyncMock(return_value=None)
    store.persist_document_with_chunks = AsyncMock(return_value=1)

    chunker = MagicMock()
    chunker.chunk.return_value = []  # zero chunks
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_many.return_value = []  # no embeddings either

    service = RagIngestService(
        chunker=chunker, embedder=embedder,
        session_factory=session_factory, store=store,
    )
    response = await service.ingest(
        source_path="data/x.json",
        document_type="historical_budget",
        budget=_make_budget(),
    )
    assert response.chunks_created == 0
    assert response.embedding_dimension == 0


@pytest.mark.asyncio
async def test_ingest_passes_metadata_to_store() -> None:
    """The document-level metadata (budget_id, sector, year) is the
    filterable JSONB column. If we forget to pass it, the index on
    ``metadata`` is useless. We assert the shape, not the bytes."""
    session_factory, _session = _make_session_factory(execute_return=None)
    store = MagicMock(spec=ChunkStore)
    store.find_document_id = AsyncMock(return_value=None)
    store.persist_document_with_chunks = AsyncMock(return_value=1)

    chunker = MagicMock()
    chunker.chunk.return_value = [_make_chunk()]
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_many.return_value = [_make_embedded()]

    service = RagIngestService(
        chunker=chunker, embedder=embedder,
        session_factory=session_factory, store=store,
    )
    await service.ingest(
        source_path="data/x.json",
        document_type="historical_budget",
        budget=_make_budget(),
    )

    kwargs = store.persist_document_with_chunks.await_args.kwargs
    assert kwargs["doc_metadata"]["budget_id"] == "BUD-2024-001"
    assert kwargs["doc_metadata"]["client_sector"] == "finance"
    assert kwargs["doc_metadata"]["year"] == 2024
    assert kwargs["source_path"] == "data/x.json"
    assert kwargs["document_type"] == "historical_budget"
