"""Runtime-mutable model configuration, backed by Redis.

``Settings`` (.env) stays the immutable layer of *defaults*; this store holds
runtime *overrides* as deltas in a Redis hash, so the instructor can switch
models from the Settings UI mid-session without editing .env or recreating
containers. Overrides survive ``--reload`` restarts and are shared across
workers (Redis is the source of truth, read on every call).

Failure semantics:
- Reads degrade gracefully: if Redis is down, ``effective()`` falls back to
  the settings default (the service behaves exactly as configured by .env).
- Writes re-raise (`RuntimeConfigUnavailable`): a failed override must be
  visible to the caller — the API maps it to a 503.
"""

from __future__ import annotations

import redis
import structlog

from src.core.config import Settings

log = structlog.get_logger()

# The LLM model knobs that can be overridden at runtime. Each name is also a
# ``Settings`` field (the default). EMBEDDING_MODEL is deliberately absent:
# changing it would invalidate every stored vector, so it stays .env-only.
MODEL_KEYS: tuple[str, ...] = (
    "PRIMARY_MODEL",
    "FALLBACK_MODEL",
    "CRITIC_MODEL",
    "METADATA_EXTRACTOR_MODEL",
    "COMPRESSION_MODEL",
    "PROPOSITIONAL_CHUNKER_MODEL",
    "CONTEXTUAL_CHUNKER_MODEL",
)

HASH_KEY = "estimator:runtime_config"


class RuntimeConfigUnavailable(Exception):
    """Raised when an override WRITE fails (Redis unreachable)."""


class RuntimeModelConfig:
    """Redis-hash-backed override store for the LLM model knobs."""

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeModelConfig":
        return cls(redis.from_url(url, decode_responses=True), settings)

    def get(self, key: str) -> str | None:
        """Return the override for ``key``, or ``None`` (unset / Redis down)."""
        self._validate_key(key)
        try:
            return self._redis.hget(HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key=key, error=str(exc)[:200])
            return None

    def set(self, key: str, value: str | None) -> None:
        """Set the override for ``key``; ``None`` clears it (back to default)."""
        self._validate_key(key)
        try:
            if value is None:
                self._redis.hdel(HASH_KEY, key)
            else:
                self._redis.hset(HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective(self, key: str) -> str:
        """The value the pipeline should use: override if set, else the default."""
        return self.get(key) or self.default(key)

    def default(self, key: str) -> str:
        self._validate_key(key)
        return getattr(self._settings, key)

    def is_overridden(self, key: str) -> bool:
        return self.get(key) is not None

    def snapshot(self) -> dict[str, dict[str, str | bool]]:
        """One entry per model key: ``{effective, default, overridden}``."""
        try:
            overrides = self._redis.hgetall(HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            key: {
                "effective": overrides.get(key) or self.default(key),
                "default": self.default(key),
                "overridden": key in overrides,
            }
            for key in MODEL_KEYS
        }

    def reset_all(self) -> None:
        """Drop every override (back to .env defaults)."""
        try:
            self._redis.delete(HASH_KEY)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    @staticmethod
    def _validate_key(key: str) -> None:
        if key not in MODEL_KEYS:
            raise ValueError(f"Unknown model key: {key}")


# ---------------------------------------------------------------------------
# Session 10 — retrieval toggles (search mode + reranking)
# ---------------------------------------------------------------------------

RETRIEVAL_HASH_KEY = "estimator:retrieval_config"

RETRIEVAL_KEYS: tuple[str, ...] = (
    "SEARCH_MODE",
    "RERANK",
)


class RuntimeRetrievalConfig:
    """Redis-hash-backed override store for the Session 10 retrieval toggles.

    Reads degrade to the settings default if Redis is down (the pipeline keeps
    serving exactly as .env configured it); writes re-raise so a failed override
    is visible to the API (mapped to 503), matching :class:`RuntimeModelConfig`.
    """

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeRetrievalConfig":
        return cls(redis.from_url(url, decode_responses=True), settings)

    def _get_raw(self, key: str) -> str | None:
        try:
            return self._redis.hget(RETRIEVAL_HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key=key, error=str(exc)[:200])
            return None

    def _set_raw(self, key: str, value: str | None) -> None:
        try:
            if value is None:
                self._redis.hdel(RETRIEVAL_HASH_KEY, key)
            else:
                self._redis.hset(RETRIEVAL_HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def get_search_mode(self) -> str | None:
        """Override for search_mode ('vector' or 'hybrid'), or None if unset."""
        return self._get_raw("SEARCH_MODE")

    def set_search_mode(self, value: str | None) -> None:
        """Set the search_mode override; None clears it."""
        if value is not None and value not in ("vector", "hybrid"):
            raise ValueError(f"Invalid search_mode: {value}")
        self._set_raw("SEARCH_MODE", value)

    def get_rerank(self) -> bool | None:
        """Override for rerank (True/False), or None if unset."""
        raw = self._get_raw("RERANK")
        if raw is None:
            return None
        return raw.lower() == "true"

    def set_rerank(self, value: bool | None) -> None:
        """Set the rerank override; None clears it."""
        self._set_raw("RERANK", "true" if value else "false" if value is not None else None)

    def effective_search_mode(self) -> str:
        """The search_mode the pipeline should use: override if set, else the default."""
        return self.get_search_mode() or getattr(self._settings, "search_mode", "vector")

    def effective_rerank(self) -> bool:
        """The rerank value the pipeline should use: override if set, else the default."""
        override = self.get_rerank()
        if override is not None:
            return override
        return getattr(self._settings, "reranker_enabled", False)

    def snapshot(self) -> dict[str, dict[str, str | bool]]:
        """One entry per retrieval key: {effective, default, overridden}."""
        try:
            overrides = self._redis.hgetall(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            "search_mode": {
                "effective": self.effective_search_mode(),
                "default": getattr(self._settings, "search_mode", "vector"),
                "overridden": "SEARCH_MODE" in overrides,
            },
            "rerank": {
                "effective": self.effective_rerank(),
                "default": getattr(self._settings, "reranker_enabled", False),
                "overridden": "RERANK" in overrides,
            },
        }

    def reset_all(self) -> None:
        """Drop every override (back to .env defaults)."""
        try:
            self._redis.delete(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc
