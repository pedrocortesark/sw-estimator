"""Tests for src/rag/embedding/embedder.py — OpenAIEmbedder.

Strategy: replace the ``openai.OpenAI`` client with a ``MagicMock`` and
program ``client.embeddings.create`` to return objects shaped like the real
API. This way we test the wrapper's logic (batching, ordering, retry,
dimensions kwarg) without making a single network call.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI, RateLimitError

from src.generation.rag.embedding.embedder import (
    BATCH_SIZE,
    MODEL,
    OpenAIEmbedder,
    estimated_cost_usd,
)
from src.generation.rag.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "hello world") -> Chunk:
    """A single chunk with a small but positive token count."""
    return Chunk(
        chunk_id="test-1",
        text=text,
        metadata={},
        token_count=2,
    )


def _fake_response(vectors: list[list[float]]) -> MagicMock:
    """Build a MagicMock that quacks like ``openai.types.Embedding.create``'s
    return value: ``response.data[i].embedding`` must return ``vectors[i]``."""
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


# ---------------------------------------------------------------------------
# Tests — basic shape
# ---------------------------------------------------------------------------


def test_default_model_is_text_embedding_3_small() -> None:
    """The column type is ``Vector(1536)`` — it is hardcoded to match THIS
    model's native dimensionality. Changing the default model without
    re-embedding the corpus is a silent data corruption bug."""
    assert MODEL == "text-embedding-3-small"


def test_embed_one_returns_first_vector() -> None:
    """``embed_one`` is the single-text path used by the retriever."""
    client = MagicMock(spec=OpenAI)
    client.embeddings.create.return_value = _fake_response([[0.1, 0.2, 0.3]])
    embedder = OpenAIEmbedder(client=client)

    vector = embedder.embed_one("test text")

    assert vector == [0.1, 0.2, 0.3]
    # And the call shape: exactly one item, the right model.
    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs["input"] == ["test text"]
    assert kwargs["model"] == MODEL
    # No dimensions kwarg unless explicitly requested.
    assert "dimensions" not in kwargs


# ---------------------------------------------------------------------------
# Tests — batching
# ---------------------------------------------------------------------------


def test_embed_many_batches_above_batch_size() -> None:
    """With BATCH_SIZE=100, 250 chunks must trigger 3 API calls (100, 100, 50)
    — never 250 separate calls. If you regress batching, you regress latency
    and you blow past OpenAI rate limits."""
    client = MagicMock(spec=OpenAI)
    # Program each .create() to return as many fake vectors as were asked.
    client.embeddings.create.side_effect = lambda **kw: _fake_response(
        [[0.0] * 3 for _ in kw["input"]]
    )

    chunks = [_make_chunk(f"chunk-{i}") for i in range(250)]
    OpenAIEmbedder(client=client).embed_many(chunks)

    assert client.embeddings.create.call_count == 3
    # Per-call batch sizes: 100, 100, 50.
    batch_sizes = [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list]
    assert batch_sizes == [BATCH_SIZE, BATCH_SIZE, 50]


def test_embed_many_preserves_input_order_across_batches() -> None:
    """Critical: the retriever relies on vector[i] being the embedding of
    chunks[i]. If batching scrambles the order, the cosine_distance search
    will return wrong results. Test with 3 batches and vectors that are
    distinguishable."""
    client = MagicMock(spec=OpenAI)
    # Per-batch mock state: the FIRST chunk in each batch carries a global
    # index that the side_effect uses to build globally-unique vectors. This
    # is the only way to detect a batch boundary mishandling.
    next_global_index = {"i": 0}

    def _side_effect(**kw):
        batch_size = len(kw["input"])
        start = next_global_index["i"]
        next_global_index["i"] += batch_size
        return _fake_response([[float(start + j)] * 3 for j in range(batch_size)])

    client.embeddings.create.side_effect = _side_effect

    chunks = [_make_chunk(f"chunk-{i}") for i in range(150)]
    embedded = OpenAIEmbedder(client=client).embed_many(chunks)

    # Chunk i should have vector [float(i), float(i), float(i)].
    assert len(embedded) == 150
    for i, emb in enumerate(embedded):
        assert emb.embedding == [float(i)] * 3, f"order broken at chunk {i}"


def test_embed_many_attaches_vectors_to_chunks() -> None:
    """The output EmbeddedChunks must carry the original chunk's text and
    metadata, plus the vector. If the merge drops fields, retrieval context
    is lost (this is what the structural chunker's parent context header
    was added to prevent)."""
    client = MagicMock(spec=OpenAI)
    client.embeddings.create.return_value = _fake_response([[0.5, 0.5]])

    chunk = _make_chunk("specific text for the embedder")
    chunk.metadata = {"client_sector": "finance"}
    embedded = OpenAIEmbedder(client=client).embed_many([chunk])

    assert len(embedded) == 1
    assert embedded[0].text == "specific text for the embedder"
    assert embedded[0].metadata == {"client_sector": "finance"}
    assert embedded[0].embedding == [0.5, 0.5]


# ---------------------------------------------------------------------------
# Tests — dimensions kwarg (Matryoshka)
# ---------------------------------------------------------------------------


def test_dimensions_kwarg_is_propagated_to_api() -> None:
    """Matryoshka embeddings (256/512/1024) are a live-session discussion
    topic. The wrapper must forward the kwarg untouched so the column type
    stays the source of truth on the persistence side."""
    client = MagicMock(spec=OpenAI)
    client.embeddings.create.return_value = _fake_response([[0.1] * 256])

    embedder = OpenAIEmbedder(client=client, dimensions=256)
    embedder.embed_one("text")

    assert client.embeddings.create.call_args.kwargs["dimensions"] == 256


# ---------------------------------------------------------------------------
# Tests — retry behavior
# ---------------------------------------------------------------------------


def test_rate_limit_triggers_retry_and_succeeds() -> None:
    """OpenAI returns 429 under load. The embedder must retry up to 3 times
    (initial + 3 backoff windows = 4 attempts) with backoff. A flaky 429
    should NOT propagate to the caller."""
    client = MagicMock(spec=OpenAI)
    # First two calls raise 429, third succeeds. The wrapper has 4 attempts
    # (one + three retries with backoff (0, 1, 2, 4)).
    client.embeddings.create.side_effect = [
        RateLimitError("rate limited", response=MagicMock(), body=None),
        RateLimitError("rate limited", response=MagicMock(), body=None),
        _fake_response([[0.1, 0.2]]),
        _fake_response([[0.1, 0.2]]),  # extra in case of an extra call
    ]

    # Patch sleep to avoid waiting the real 1+2+4 seconds.
    with patch("src.generation.rag.embedding.embedder.time.sleep") as mock_sleep:
        vector = OpenAIEmbedder(client=client).embed_one("text")

    assert vector == [0.1, 0.2]
    # Exactly 3 calls: 2 failed + 1 success.
    assert client.embeddings.create.call_count == 3
    # And the backoff sleeps were scheduled (1, 2 — the 0 is a no-op).
    sleep_durations = [call.args[0] for call in mock_sleep.call_args_list if call.args]
    assert 1 in sleep_durations and 2 in sleep_durations


def test_rate_limit_exhausts_retries_raises() -> None:
    """If OpenAI keeps returning 429 past the retry budget, the embedder
    must raise so the caller can decide (the ingest service rolls back its
    transaction in that case)."""
    client = MagicMock(spec=OpenAI)
    client.embeddings.create.side_effect = RateLimitError(
        "still rate limited", response=MagicMock(), body=None
    )

    with patch("src.generation.rag.embedding.embedder.time.sleep"):
        with pytest.raises(RateLimitError):
            OpenAIEmbedder(client=client).embed_one("text")

    # 4 attempts: 1 initial + 3 retries.
    assert client.embeddings.create.call_count == 4


# ---------------------------------------------------------------------------
# Tests — cost math
# ---------------------------------------------------------------------------


def test_estimated_cost_matches_pricing_constant() -> None:
    """text-embedding-3-small is $0.02 per 1M input tokens. If OpenAI revises
    the price, ``PRICE_PER_MILLION_TOKENS_USD`` is the only knob to turn."""
    # 1M tokens → $0.02
    assert estimated_cost_usd(1_000_000) == pytest.approx(0.02)
    # 50k tokens → $0.001
    assert estimated_cost_usd(50_000) == pytest.approx(0.001)
    # 0 tokens → $0
    assert estimated_cost_usd(0) == 0.0
