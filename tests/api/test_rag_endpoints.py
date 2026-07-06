"""HTTP integration tests for the RAG endpoints (Session 8).

We never start a real Postgres or call OpenAI. Both dependencies are
overridden with fakes:

* ``get_rag_ingest_service`` — a fake that returns a canned IngestResponse
  (or raises ``DuplicateDocumentError``) so we can exercise the 200/409/500
  paths of POST /embeddings/ingest.
* ``get_semantic_retriever`` — a fake that returns a canned SearchResponse
  so we can exercise POST /search.

Pattern follows ``tests/api/test_sessions.py``: build the app fresh, install
the override, hit endpoints through ``ASGITransport`` (no TCP).
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.dependencies import get_rag_ingest_service, get_semantic_retriever
from src.main import create_app
from src.generation.rag.ingest_service import DuplicateDocumentError, RagIngestService
from src.generation.rag.retriever import SemanticRetriever
from src.generation.rag.schemas import (
    Budget,
    BudgetComponent,
    ClientMetadata,
    IngestResponse,
    SearchHit,
    SearchResponse,
)


# ---------------------------------------------------------------------------
# Fixtures — fake services
# ---------------------------------------------------------------------------


_SAMPLE_BUDGET = {
    "budget_id": "BUD-2024-001",
    "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
    "project_summary": "Mobile banking API with OAuth 2.0 authentication",
    "main_technology": "ruby_on_rails",
    "year": 2024,
    "total_estimated_hours": 480,
    "components": [
        {
            "component_id": "AUTH-001",
            "name": "OAuth 2.0 backend",
            "description": "OAuth 2.0 with JWT",
            "tech_stack": ["ruby_on_rails", "postgresql"],
            "estimated_hours": 120,
            "complexity": "high",
            "dependencies": [],
        }
    ],
}


def _make_ingest_response() -> IngestResponse:
    return IngestResponse(
        document_id=42,
        chunks_created=4,
        embedding_dimension=1536,
        ingestion_time_ms=1240,
    )


def _make_search_response() -> SearchResponse:
    return SearchResponse(
        query="test query",
        k=5,
        search_time_ms=87,
        results=[
            SearchHit(
                chunk_id=156,
                document_id=12,
                chunk_type="budget_component",
                content="Some content here...",
                distance=0.231,
                metadata={"scope": "backend", "technologies": ["python", "fastapi"]},
            ),
            SearchHit(
                chunk_id=158,
                document_id=12,
                chunk_type="budget_component",
                content="More content here...",
                distance=0.302,
                metadata={"scope": "frontend"},
            ),
        ],
    )


class _FakeRagService:
    """Returns a canned IngestResponse, or raises DuplicateDocumentError on
    demand, so we can exercise the 200/409 paths without a real DB."""

    def __init__(self) -> None:
        self.last_call: dict | None = None
        self.raise_duplicate: bool = False

    async def ingest(self, *, source_path: str, document_type: str, budget: Budget) -> IngestResponse:
        self.last_call = {"source_path": source_path, "document_type": document_type}
        if self.raise_duplicate:
            raise DuplicateDocumentError(document_id=42)
        return _make_ingest_response()


class _FakeRetriever:
    """Returns a canned SearchResponse."""

    async def search(self, *, query: str, k: int) -> SearchResponse:
        return _make_search_response()


@pytest_asyncio.fixture
async def ingest_client():
    """AsyncClient wired to a fresh app with a fake ingest service."""
    app = create_app()
    fake = _FakeRagService()
    app.dependency_overrides[get_rag_ingest_service] = lambda: fake

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        ac._fake_service = fake  # type: ignore[attr-defined]
        yield ac


@pytest_asyncio.fixture
async def search_client():
    """AsyncClient wired to a fresh app with a fake retriever."""
    app = create_app()
    fake = _FakeRetriever()
    app.dependency_overrides[get_semantic_retriever] = lambda: fake

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests — POST /embeddings/ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_returns_200_with_response_shape(ingest_client: AsyncClient) -> None:
    """Happy path: a valid Budget produces a 200 with the four expected
    fields (document_id, chunks_created, embedding_dimension, ingestion_time_ms)."""
    response = await ingest_client.post(
        "/embeddings/ingest",
        json={
            "source_path": "data/foo.json",
            "document_type": "historical_budget",
            "content": _SAMPLE_BUDGET,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == 42
    assert body["chunks_created"] == 4
    assert body["embedding_dimension"] == 1536
    assert body["ingestion_time_ms"] == 1240


@pytest.mark.asyncio
async def test_ingest_passes_source_path_to_service(ingest_client: AsyncClient) -> None:
    """The source_path is the duplicate-check key. The router must pass it
    through unchanged, not lowercase or strip it."""
    response = await ingest_client.post(
        "/embeddings/ingest",
        json={
            "source_path": "data/CASE_Sensitive_Path.JSON",
            "document_type": "historical_budget",
            "content": _SAMPLE_BUDGET,
        },
    )
    assert response.status_code == 200
    assert ingest_client._fake_service.last_call["source_path"] == "data/CASE_Sensitive_Path.JSON"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ingest_returns_409_on_duplicate(ingest_client: AsyncClient) -> None:
    """The 409 contract from the exercise: when the source_path already
    exists, return ``{"detail": "Document already ingested", "document_id": N}``."""
    ingest_client._fake_service.raise_duplicate = True  # type: ignore[attr-defined]
    response = await ingest_client.post(
        "/embeddings/ingest",
        json={
            "source_path": "data/already_ingested.json",
            "document_type": "historical_budget",
            "content": _SAMPLE_BUDGET,
        },
    )
    assert response.status_code == 409, response.text
    body = response.json()
    assert body["detail"] == "Document already ingested"
    assert body["document_id"] == 42


@pytest.mark.asyncio
async def test_ingest_returns_422_on_malformed_budget(ingest_client: AsyncClient) -> None:
    """Malformed JSON must fail at the Pydantic validation stage (422),
    not at the service. The router does not catch ValidationError — FastAPI's
    built-in handler does — so the test only checks the status code and
    that the service was NOT called."""
    response = await ingest_client.post(
        "/embeddings/ingest",
        json={
            "source_path": "data/x.json",
            "document_type": "historical_budget",
            "content": {"this": "is not a budget"},  # missing required fields
        },
    )
    assert response.status_code == 422, response.text
    assert ingest_client._fake_service.last_call is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ingest_returns_500_when_service_unavailable() -> None:
    """When no OpenAI key is configured, ``get_rag_ingest_service`` returns
    ``None`` and the router maps that to a 500 with a generic message —
    never revealing the underlying cause to the client."""
    app = create_app()
    # Override the dependency to return ``None`` to simulate the no-key case.
    app.dependency_overrides[get_rag_ingest_service] = lambda: None

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        response = await ac.post(
            "/embeddings/ingest",
            json={
                "source_path": "data/x.json",
                "document_type": "historical_budget",
                "content": _SAMPLE_BUDGET,
            },
        )

    assert response.status_code == 500
    assert "embedding service is not available" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Tests — POST /search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_200_with_response_shape(search_client: AsyncClient) -> None:
    """Happy path: a valid query produces a 200 with the documented shape
    (query echo, k, search_time_ms, results list)."""
    response = await search_client.post(
        "/search",
        json={"query": "REST API with OAuth", "k": 5},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query"] == "test query"  # echoed from the fake
    assert body["k"] == 5
    assert isinstance(body["search_time_ms"], int)
    assert body["search_time_ms"] >= 0
    assert len(body["results"]) == 2

    # Each result has the six fields the SearchHit schema declares.
    first = body["results"][0]
    for key in ("chunk_id", "document_id", "chunk_type", "content", "distance", "metadata"):
        assert key in first, f"missing key {key} in search hit"


@pytest.mark.asyncio
async def test_search_results_are_ordered_by_distance() -> None:
    """The retriever returns chunks ordered by cosine distance ASC. The
    router does not reorder. This is implicit in the SQL but worth
    pinning down with a fake that returns results in a specific order."""
    # Build a retriever whose fake returns ordered rows.
    ordered = [
        SearchHit(
            chunk_id=i,
            document_id=99,
            chunk_type="budget_component",
            content=f"chunk {i}",
            distance=0.1 * (i + 1),  # strictly increasing
            metadata={},
        )
        for i in range(1, 4)
    ]
    fake = _FakeRetrieverOrdered(ordered)

    app = create_app()
    app.dependency_overrides[get_semantic_retriever] = lambda: fake

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        response = await ac.post("/search", json={"query": "q", "k": 3})

    body = response.json()
    distances = [r["distance"] for r in body["results"]]
    assert distances == sorted(distances), f"results not ordered: {distances}"


@pytest.mark.asyncio
async def test_search_returns_422_when_k_above_max() -> None:
    """``SearchRequest.k`` is bounded [1, 50]. Out-of-range values are
    rejected at the Pydantic layer (422), not silently clamped."""
    app = create_app()
    app.dependency_overrides[get_semantic_retriever] = lambda: _FakeRetriever()

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        response = await ac.post(
            "/search", json={"query": "q", "k": 999}  # 999 > 50
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_returns_422_on_empty_query() -> None:
    """``SearchRequest.query`` is ``min_length=1``. Empty strings are 422,
    not 200 with empty results."""
    app = create_app()
    app.dependency_overrides[get_semantic_retriever] = lambda: _FakeRetriever()

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        response = await ac.post("/search", json={"query": "", "k": 5})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_returns_500_when_retriever_unavailable() -> None:
    """When no OpenAI key is configured, ``get_semantic_retriever`` returns
    ``None`` and the router maps that to a 500."""
    app = create_app()
    app.dependency_overrides[get_semantic_retriever] = lambda: None

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        response = await ac.post("/search", json={"query": "q", "k": 5})

    assert response.status_code == 500
    assert "embedding service is not available" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------


class _FakeRetrieverOrdered:
    """Variant of ``_FakeRetriever`` that returns a caller-supplied ordered
    list. Used by the ordering test."""

    def __init__(self, ordered: list[SearchHit]) -> None:
        self._ordered = ordered

    async def search(self, *, query: str, k: int) -> SearchResponse:
        return SearchResponse(
            query=query, k=k, search_time_ms=1, results=self._ordered,
        )
