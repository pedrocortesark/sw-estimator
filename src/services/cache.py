"""Exact-match cache for LLM estimation responses backed by Redis.

Cache key strategy
------------------
SHA-256 over a deterministic JSON serialization of five call parameters:
    system_prompt, user_message, model, max_tokens, thinking_budget

Any change to the prompt text, model, or token budget produces a different
digest, invalidating the old entry automatically — no manual flushing needed.

Error policy
------------
Redis errors are *never* propagated to callers. The cache is best-effort:
a failure in ``get`` or ``set`` simply bypasses the cache and lets the
pipeline proceed normally.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis
import structlog
from redis import RedisError

logger = structlog.get_logger()

_KEY_PREFIX = "estimation:"


class EstimationCache:
    """Redis-backed exact-match cache for LLM estimation responses.

    Args:
        redis_client: A connected ``redis.Redis`` instance with
            ``decode_responses=True``.
        ttl: Time-to-live in seconds for each cached entry. Defaults to
            86 400 (24 hours).
    """

    def __init__(self, redis_client: redis.Redis, ttl: int = 86_400) -> None:
        self._redis = redis_client
        self.ttl = ttl

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_url(cls, url: str, ttl: int = 86_400) -> "EstimationCache":
        """Create an ``EstimationCache`` from a Redis connection URL.

        Args:
            url: Redis URL, e.g. ``"redis://localhost:6379/0"``.
            ttl: Time-to-live in seconds. Defaults to 86 400 (24 hours).
        """
        client = redis.from_url(url, decode_responses=True)
        return cls(redis_client=client, ttl=ttl)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        thinking_budget: int | None,
    ) -> str:
        """Generate a deterministic Redis key for the given call parameters.

        The key is ``estimation:<sha256-hex>`` where the digest is computed
        over a JSON object with ``sort_keys=True`` to guarantee stability
        regardless of dict insertion order.

        Args:
            system_prompt: The rendered system prompt string.
            user_message: The rendered user prompt string.
            model: LiteLLM model identifier (e.g. ``"gpt-4o-mini"``).
            max_tokens: Maximum number of tokens requested from the model.
            thinking_budget: Optional token budget for extended thinking
                (Anthropic-specific). ``None`` is serialised as JSON null.

        Returns:
            A string of the form ``"estimation:<64-char-hex-digest>"``.
        """
        payload = json.dumps(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model": model,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"{_KEY_PREFIX}{digest}"

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached response dict for *key*, or ``None`` on a miss.

        Redis errors are swallowed — a warning is logged and ``None`` is
        returned so the pipeline falls through to a live LLM call.

        Args:
            key: Cache key produced by :meth:`make_key`.

        Returns:
            Deserialised dict on a hit, ``None`` on a miss or error.
        """
        key_prefix = key[:24]
        try:
            raw = self._redis.get(key)
        except RedisError as exc:
            logger.warning(
                "cache_error", operation="get", key_prefix=key_prefix, error=str(exc)
            )
            return None

        if raw is None:
            logger.debug("cache_miss", key_prefix=key_prefix)
            return None

        logger.info("cache_hit", key_prefix=key_prefix)
        return json.loads(raw)

    def set(self, key: str, response: dict[str, Any]) -> None:
        """Persist *response* under *key* with the configured TTL.

        Redis errors are swallowed — a warning is logged and the pipeline
        continues normally. The cache is best-effort.

        Args:
            key: Cache key produced by :meth:`make_key`.
            response: Serialisable dict to cache (e.g. the full API response).
        """
        key_prefix = key[:24]
        try:
            self._redis.setex(key, self.ttl, json.dumps(response))
        except RedisError as exc:
            logger.warning(
                "cache_error", operation="set", key_prefix=key_prefix, error=str(exc)
            )
            return

        logger.info("cache_stored", key_prefix=key_prefix, ttl=self.ttl)
