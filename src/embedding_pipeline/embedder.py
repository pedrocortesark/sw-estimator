"""OpenAI embedder — wraps text-embedding-3-small with batched requests.

Pricing constant: $0.02 per million input tokens (text-embedding-3-small, 2024).
Update _PRICE_PER_MILLION_TOKENS when OpenAI changes the rate card.
"""

from __future__ import annotations

import time
from typing import Final

import structlog
from openai import OpenAI, RateLimitError

from src.core.config import (
    get_settings,
)  # uses src/core/config.py — field: openai_api_key
from src.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger(__name__)

_MODEL: Final[str] = "text-embedding-3-small"
_BATCH_SIZE: Final[int] = 100
_PRICE_PER_MILLION_TOKENS: Final[float] = 0.02  # USD — update when rate card changes
_RETRY_DELAYS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0)


def _call_with_retry(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Call embeddings.create with exponential backoff on RateLimitError."""
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            response = client.embeddings.create(input=texts, model=_MODEL)
            return [
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            ]
        except RateLimitError as exc:
            last_exc = exc
            if delay is None:
                break
            log.warning(
                "embedder.rate_limit_retry",
                attempt=attempt,
                wait_seconds=delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


class OpenAIEmbedder:
    """Embeds text using text-embedding-3-small (1 536 dims, default)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string. Useful for query-time embedding."""
        t0 = time.perf_counter()
        vectors = _call_with_retry(self._client, [text])
        latency = time.perf_counter() - t0
        log.info(
            "embedder.embed_one", tokens=len(text.split()), latency_s=round(latency, 3)
        )
        return vectors[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of Chunks in batches. Returns EmbeddedChunk objects."""
        if not chunks:
            return []

        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            texts = [c.text for c in batch]
            total_tokens = sum(c.token_count for c in batch)

            t0 = time.perf_counter()
            vectors = _call_with_retry(self._client, texts)
            latency = time.perf_counter() - t0

            log.info(
                "embedder.batch_done",
                batch_index=batch_start // _BATCH_SIZE,
                chunks_in_batch=len(batch),
                tokens_in_batch=total_tokens,
                latency_s=round(latency, 3),
            )
            all_vectors.extend(vectors)

        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=vector)
            for chunk, vector in zip(chunks, all_vectors)
        ]

    # ------------------------------------------------------------------
    # Cost estimation (pure utility, no API call)
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_cost(chunks: list[Chunk]) -> float:
        """Return the estimated cost in USD for embedding the given chunks."""
        total_tokens = sum(c.token_count for c in chunks)
        return (total_tokens / 1_000_000) * _PRICE_PER_MILLION_TOKENS
