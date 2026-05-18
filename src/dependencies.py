"""FastAPI dependency providers for the estimation pipeline."""

from __future__ import annotations

from functools import lru_cache

import structlog

from src.services.estimation import EstimationService

logger = structlog.get_logger()


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
