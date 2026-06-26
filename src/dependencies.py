"""FastAPI dependency providers for the estimation pipeline."""

from __future__ import annotations

from functools import lru_cache

import structlog

from src.services.estimation import EstimationService

logger = structlog.get_logger()


# --- Session 8: pgvector persistence + semantic search ---------------------


@lru_cache
def get_openai_client():
    """Lazy OpenAI client used by the embedding pipeline (Session 8).

    Returns ``None`` when no API key is configured; the routers map that to a
    500 with a generic message. The contract matches the existing semantic
    cache one (``_make_semantic_cache``) for symmetry.
    """
    from openai import OpenAI

    from src.core.config import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        logger.warning("openai_client_disabled", reason="no_openai_key")
        return None
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache
def get_chunker():
    """Stateless structural chunker for the embedding pipeline (Session 8)."""
    from src.rag.chunking.structural import JSONStructuralChunker

    return JSONStructuralChunker()


@lru_cache
def get_embedder():
    """OpenAI embedder. ``None`` when no API key is configured."""
    from src.rag.embedding.embedder import OpenAIEmbedder

    client = get_openai_client()
    if client is None:
        return None
    # The embedder is hard-wired to text-embedding-3-small (1536 dims) — its
    # dimensionality is baked into the chunks.embedding column Vector(1536).
    # Using a chat model like gpt-4o-mini here triggers a 403.
    return OpenAIEmbedder(client=client)


@lru_cache
def get_chunk_store():
    """Stateless async data-access layer over documents/chunks."""
    from src.rag.store.repository import ChunkStore

    return ChunkStore()


@lru_cache
def get_generation_chunk_store():
    """Stateless async data-access layer for the generation pipeline (Session 9/10).

    This is the ChunkStore with search_filtered() and search_lexical() methods
    needed by the hybrid retrieval pipeline.
    """
    from src.generation.rag.store.repository import ChunkStore

    return ChunkStore()


@lru_cache
def get_rag_ingest_service():
    """Chunk → embed → persist orchestration. ``None`` without an OpenAI key;
    the router maps that to a 500."""
    from src.rag.ingest_service import RagIngestService

    from src.persistence.database import get_async_session_factory

    embedder = get_embedder()
    if embedder is None:
        return None
    return RagIngestService(
        chunker=get_chunker(),
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


@lru_cache
def get_semantic_retriever():
    """Query-side counterpart of the ingest service. Same ``None`` contract."""
    from src.rag.retriever import SemanticRetriever

    from src.persistence.database import get_async_session_factory

    embedder = get_embedder()
    if embedder is None:
        return None
    return SemanticRetriever(
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


# --- Session 10: cross-encoder reranking ------------------------------------


@lru_cache
def get_reranker():
    """Cross-encoder reranker singleton (Session 10). The model loads lazily on
    the first rerank, so building this is cheap and import-time has no torch cost."""
    from src.generation.rag.retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker.from_settings()


def _make_semantic_cache():
    """Try to create a Redis-backed ``EstimationSemanticCache``.

    Returns ``None`` (gracefully) when:
    - ``REDIS_URL`` is not set in settings
    - Redis Stack is unreachable (ping fails)
    - ``OPENAI_API_KEY`` is absent (can't embed transcripts)
    - Any other initialisation error
    """
    from src.core.config import get_settings

    settings = get_settings()
    if not settings.redis_url:
        logger.debug("semantic_cache_skipped", reason="REDIS_URL not set")
        return None
    if not settings.openai_api_key:
        logger.warning("semantic_cache_skipped", reason="OPENAI_API_KEY not set")
        return None

    try:
        import redis
        from redisvl.utils.vectorize import OpenAITextVectorizer

        from src.cache.semantic import EstimationSemanticCache

        redis_client = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        redis_client.ping()  # fail fast — don't silently block the request path

        vectorizer = OpenAITextVectorizer(
            model="text-embedding-3-small",
            api_config={"api_key": settings.openai_api_key},
        )

        cache = EstimationSemanticCache(
            redis_client,
            vectorizer,
            threshold=settings.semantic_cache_threshold,
        )
        logger.info("semantic_cache_ready", redis_url=settings.redis_url)
        return cache

    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_cache_unavailable", error=str(exc))
        return None


@lru_cache(maxsize=1)
def _shared_service() -> EstimationService:
    """Singleton ``EstimationService`` — created once per worker process.

    Using a singleton ensures the Redis connection and in-memory cache dict
    are shared across all requests, not re-created on every call.
    """
    cache = _make_semantic_cache()
    return EstimationService(cache=cache)


def get_estimation_service() -> EstimationService:
    """FastAPI dependency — returns the shared ``EstimationService``."""
    return _shared_service()
