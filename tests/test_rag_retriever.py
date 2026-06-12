"""Tests for src/rag/retriever.py — SemanticRetriever.

The retriever embeds a query and ranks chunks by cosine distance. It does
not contain branching logic — the test surface is: (1) the embedder is
called once with the query text, (2) the store is called with the
resulting vector and k, (3) each row is mapped to a SearchHit preserving
the DB primary keys and the distance.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.embedding.embedder import OpenAIEmbedder
from src.rag.retriever import SemanticRetriever
from src.rag.store.repository import ChunkStore


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_embeds_query_via_embedder() -> None:
    """The retriever MUST embed the query with the same model as ingest.
    Mixing models is the classic RAG footgun (distances become meaningless)."""
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_one = MagicMock(return_value=[0.1, 0.2, 0.3])

    store = MagicMock(spec=ChunkStore)
    store.search = AsyncMock(return_value=[])

    # session_factory is a no-op here: a real factory is required but the
    # retriever only enters the CM on a successful embed+search.
    factory = MagicMock()

    service = SemanticRetriever(embedder=embedder, session_factory=factory, store=store)

    # Make factory() return an async CM that yields a no-op session.
    fake_session = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=fake_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=None)

    await service.search(query="test query", k=3)

    embedder.embed_one.assert_called_once_with("test query")


@pytest.mark.asyncio
async def test_search_calls_store_with_embedded_vector_and_k() -> None:
    """The retriever must pass the embedded vector and the k to the store
    in a single call. Splitting into multiple calls would break k-NN."""
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_one = MagicMock(return_value=[0.5, 0.6, 0.7])

    store = MagicMock(spec=ChunkStore)
    store.search = AsyncMock(return_value=[])

    factory = MagicMock()
    fake_session = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=fake_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=None)

    service = SemanticRetriever(embedder=embedder, session_factory=factory, store=store)
    await service.search(query="anything", k=10)

    store.search.assert_awaited_once()
    args, kwargs = store.search.call_args
    # ``session`` is positional in ChunkStore.search; the rest are keyword.
    assert args[0] is fake_session
    assert kwargs["query_vector"] == [0.5, 0.6, 0.7]
    assert kwargs["k"] == 10


@pytest.mark.asyncio
async def test_search_maps_rows_to_search_hits() -> None:
    """The DB row shape must be projected onto SearchHit with the right field
    names: ``id`` → ``chunk_id``, ``metadata_`` → ``metadata``,
    ``distance`` cast to float (NUMERIC type might return Decimal in some
    drivers)."""
    row1 = MagicMock(
        id=11, document_id=22, chunk_type="budget_component",
        content="first chunk content", metadata_={"k": "v"},
        distance=0.1234,
    )
    row2 = MagicMock(
        id=12, document_id=22, chunk_type="budget_component",
        content="second chunk content", metadata_={"k": "v2"},
        distance=0.5678,
    )
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_one = MagicMock(return_value=[0.0, 0.0])
    store = MagicMock(spec=ChunkStore)
    store.search = AsyncMock(return_value=[row1, row2])

    factory = MagicMock()
    fake_session = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=fake_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=None)

    service = SemanticRetriever(embedder=embedder, session_factory=factory, store=store)
    response = await service.search(query="q", k=5)

    assert len(response.results) == 2
    assert response.results[0].chunk_id == 11
    assert response.results[0].document_id == 22
    assert response.results[0].content == "first chunk content"
    assert response.results[0].distance == 0.1234
    assert response.results[0].metadata == {"k": "v"}
    assert response.results[1].chunk_id == 12
    assert response.results[1].distance == 0.5678


@pytest.mark.asyncio
async def test_search_response_carries_query_and_k() -> None:
    """The response echoes the request's query and k — handy for clients
    that batch requests and want to assert against their own input."""
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_one = MagicMock(return_value=[0.0])
    store = MagicMock(spec=ChunkStore)
    store.search = AsyncMock(return_value=[])

    factory = MagicMock()
    fake_session = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=fake_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=None)

    service = SemanticRetriever(embedder=embedder, session_factory=factory, store=store)
    response = await service.search(query="my exact query string", k=7)

    assert response.query == "my exact query string"
    assert response.k == 7


@pytest.mark.asyncio
async def test_search_response_search_time_is_positive() -> None:
    """Even a 0-result search must return a non-negative search_time_ms.
    If it were 0, the metric is uninformative; if it could be negative,
    the timer is buggy."""
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_one = MagicMock(return_value=[0.0])
    store = MagicMock(spec=ChunkStore)
    store.search = AsyncMock(return_value=[])

    factory = MagicMock()
    fake_session = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=fake_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=None)

    service = SemanticRetriever(embedder=embedder, session_factory=factory, store=store)
    response = await service.search(query="q", k=1)

    assert response.search_time_ms >= 0


@pytest.mark.asyncio
async def test_search_with_empty_corpus_returns_empty_results() -> None:
    """Empty corpus is a 200 with zero results, not an error. The
    ``SearchResponse`` pydantic model must accept an empty list."""
    embedder = MagicMock(spec=OpenAIEmbedder)
    embedder.embed_one = MagicMock(return_value=[0.0])
    store = MagicMock(spec=ChunkStore)
    store.search = AsyncMock(return_value=[])

    factory = MagicMock()
    fake_session = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=fake_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=None)

    service = SemanticRetriever(embedder=embedder, session_factory=factory, store=store)
    response = await service.search(query="q", k=5)

    assert response.results == []
