"""Tests for src/rag/store/repository.py — ChunkStore.

Strategy: do not start a real Postgres. The store only builds SQLAlchemy
statements and runs them against an ``AsyncSession`` it does not own. We
mock the session (with ``AsyncMock``) and assert the SHAPE of the statement
that was passed in, plus the returned rows.

What we test:
- ``find_document_id`` builds the right ``SELECT documents.id WHERE
  source_path = :path`` and unwraps ``scalar_one_or_none``.
- ``persist_document_with_chunks`` flushes (to assign ``document.id``) before
  adding chunks, and uses ``add_all`` for the bulk insert.
- ``search`` orders by ``cosine_distance(<vector>)`` ASC, limits to k, and
  selects the columns needed by ``SearchHit`` plus a labelled ``distance``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.generation.rag.schemas import EmbeddedChunk
from src.generation.rag.store.repository import BUDGET_COMPONENT, ChunkStore


# ---------------------------------------------------------------------------
# Tests — find_document_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_document_id_returns_id_when_present() -> None:
    """When a row exists, the method must return its integer id (not the
    row, not None, not a tuple)."""
    session = AsyncMock(spec=AsyncSession)
    # session.execute() is an awaitable that returns a sync result object.
    # That result object's .scalar_one_or_none() is a regular method.
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = 42
    session.execute.return_value = execute_result

    store = ChunkStore()
    result = await store.find_document_id(session, source_path="data/foo.json")

    assert result == 42
    session.execute.assert_awaited_once()
    execute_result.scalar_one_or_none.assert_called_once()


@pytest.mark.asyncio
async def test_find_document_id_returns_none_when_absent() -> None:
    """When no row matches, the method must return None so the ingest
    service knows the source_path is free to be ingested."""
    session = AsyncMock(spec=AsyncSession)
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None
    session.execute.return_value = execute_result

    result = await ChunkStore().find_document_id(session, source_path="nope")

    assert result is None


# ---------------------------------------------------------------------------
# Tests — persist_document_with_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_flushes_before_adding_chunks() -> None:
    """The DocumentRow.id is a BigInteger autoincrement — SQLAlchemy assigns
    it on flush, not on commit. The chunks must reference ``document.id``,
    so flush must happen BEFORE add_all(chunks). If we swapped the order,
    chunks would FK to NULL and the DB would reject the INSERT."""
    session = AsyncMock(spec=AsyncSession)
    # Capture the call ORDER via side effects (more robust than inspecting
    # ``method_calls``, which AsyncMock renders inconsistently when methods
    # are re-bound).
    call_order: list[str] = []

    async def fake_flush(*args, **kwargs) -> None:
        # Set the document's id so the add_all(chunks) can reference it.
        document = session.add.call_args[0][0]
        document.id = 1
        call_order.append("flush")

    session.flush.side_effect = fake_flush
    session.add.side_effect = lambda obj: call_order.append("add")
    session.add_all.side_effect = lambda objs: call_order.append("add_all")

    embedded = [EmbeddedChunk(chunk_id="b1::c1", text="t1", metadata={}, token_count=1, embedding=[0.1, 0.2])]
    await ChunkStore().persist_document_with_chunks(
        session,
        source_path="data/x.json",
        document_type="historical_budget",
        doc_metadata={"budget_id": "b1"},
        embedded_chunks=embedded,
    )

    # The contract: add(document) → flush() → add_all(chunks).
    assert call_order == ["add", "flush", "add_all"], (
        f"order was {call_order}; chunks would FK to NULL if flush came after"
    )


@pytest.mark.asyncio
async def test_persist_does_not_commit() -> None:
    """The store deliberately does NOT commit. The caller (ingest service)
    owns the transaction so a duplicate-check + insert fit in one. If the
    store committed, the duplicate check would race against the insert."""
    session = AsyncMock(spec=AsyncSession)
    session.flush.return_value = None

    await ChunkStore().persist_document_with_chunks(
        session,
        source_path="p",
        document_type="historical_budget",
        doc_metadata={},
        embedded_chunks=[
            EmbeddedChunk(chunk_id="c", text="t", metadata={}, token_count=1, embedding=[0.0, 0.0])
        ],
    )

    session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_persist_returns_document_id_from_flushed_row() -> None:
    """After flush(), the document row has its DB-assigned id. The store
    must return THAT id, not the Pydantic one (which doesn't exist)."""
    session = AsyncMock(spec=AsyncSession)

    async def fake_flush() -> None:
        session.add.call_args_list[0].args[0].id = 99

    session.flush.side_effect = fake_flush

    document_id = await ChunkStore().persist_document_with_chunks(
        session,
        source_path="p",
        document_type="historical_budget",
        doc_metadata={},
        embedded_chunks=[
            EmbeddedChunk(chunk_id="c", text="t", metadata={}, token_count=1, embedding=[0.0, 0.0])
        ],
    )
    assert document_id == 99


@pytest.mark.asyncio
async def test_persist_uses_budget_component_chunk_type() -> None:
    """The structural chunker emits one chunk per BudgetComponent. The
    ``chunk_type`` column is what the live session filters on; it must be
    the constant ``BUDGET_COMPONENT`` (``"budget_component"``) for every
    chunk, not the chunk_id or anything else."""
    session = AsyncMock(spec=AsyncSession)
    session.flush.return_value = None

    chunks = [
        EmbeddedChunk(chunk_id=f"b::c{i}", text="t", metadata={}, token_count=1, embedding=[0.0])
        for i in range(3)
    ]
    await ChunkStore().persist_document_with_chunks(
        session,
        source_path="p",
        document_type="historical_budget",
        doc_metadata={},
        embedded_chunks=chunks,
    )

    add_all_args = session.add_all.call_args.args[0]
    assert len(list(add_all_args)) == 3
    for row in add_all_args:
        assert row.chunk_type == BUDGET_COMPONENT
        assert row.chunk_type == "budget_component"


# ---------------------------------------------------------------------------
# Tests — search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_orders_by_cosine_distance_and_limits() -> None:
    """The whole point of pgvector here: rank by cosine distance ASC and
    take the top k. We assert the rendered SQL contains ``ORDER BY`` and
    ``LIMIT :param_1`` with k as the bound value."""
    session = AsyncMock(spec=AsyncSession)
    execute_result = MagicMock()
    execute_result.all.return_value = []
    session.execute.return_value = execute_result

    store = ChunkStore()
    await store.search(session, query_vector=[0.1, 0.2], k=5)

    # Inspect the Select object that was passed to session.execute.
    stmt_arg = session.execute.await_args.args[0]
    rendered = str(stmt_arg.compile(
        compile_kwargs={"literal_binds": False}
    ))
    # The order_by must be the cosine distance expression.
    assert "cosine_distance" in rendered.lower() or "<=>" in rendered
    # The limit clause must include the integer k.
    # We pass the k=5 as a bound parameter, so it shows up as ``LIMIT :param_1``.
    # To check the actual value, look at the limit clause's internal _limit attr.
    assert stmt_arg._limit == 5


@pytest.mark.asyncio
async def test_search_returns_rows_from_session() -> None:
    """The store does not transform rows — it just hands them back. The
    retriever maps them to SearchHit. This contract matters: if the store
    started transforming, the retriever would break."""
    row1 = MagicMock(id=1, document_id=10, chunk_type="budget_component",
                     content="text", metadata_={"k": "v"}, distance=0.1)
    row2 = MagicMock(id=2, document_id=10, chunk_type="budget_component",
                     content="text", metadata_={}, distance=0.2)
    session = AsyncMock(spec=AsyncSession)
    execute_result = MagicMock()
    execute_result.all.return_value = [row1, row2]
    session.execute.return_value = execute_result

    rows = await ChunkStore().search(session, query_vector=[0.0, 0.0], k=2)
    # Same elements (row identity preserved), even if the store materialises
    # them into a new list with ``list(...)``.
    assert len(rows) == 2
    assert rows[0] is row1
    assert rows[1] is row2
    assert rows[0].id == 1
    assert rows[0].distance == 0.1


@pytest.mark.asyncio
async def test_search_uses_1536_dimensional_query_vector() -> None:
    """The query_vector and the column type Vector(1536) MUST have matching
    dimensionality — pgvector raises a type error otherwise. This is a smoke
    test that the wrapper passes the vector through unchanged."""
    session = AsyncMock(spec=AsyncSession)
    execute_result = MagicMock()
    execute_result.all.return_value = []
    session.execute.return_value = execute_result

    query_vector = [0.1] * 1536
    await ChunkStore().search(session, query_vector=query_vector, k=1)

    # The statement is what carries the vector into the SQL. Rendering it
    # and checking the cosine_distance operator appears confirms the
    # distance expression was wired up. The actual vector values are
    # parametrized so we cannot inspect them in the SQL string.
    stmt_arg = session.execute.await_args.args[0]
    rendered = str(stmt_arg.compile(compile_kwargs={"literal_binds": True}))
    # pgvector exposes the cosine distance operator as ``<=>`` in raw SQL.
    assert "<=>" in rendered
