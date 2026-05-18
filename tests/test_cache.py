"""Tests for src/services/cache.py — EstimationCache backed by FakeRedis."""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import pytest
from redis import RedisError

from src.services.cache import EstimationCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_PARAMS = dict(
    system_prompt="You are an expert estimator.",
    user_message="Meeting transcript: build a dashboard.",
    model="gpt-4o-mini",
    max_tokens=2048,
    thinking_budget=None,
)

SAMPLE_RESPONSE = {
    "estimation": {"executive_summary": "Three weeks, two devs."},
    "provider_used": "openai",
    "model_used": "gpt-4o-mini",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 200,
        "total_tokens": 300,
        "cost_usd": 0.001,
    },
}


@pytest.fixture()
def cache() -> EstimationCache:
    """EstimationCache backed by an in-memory FakeRedis (no network needed)."""
    client = fakeredis.FakeRedis(decode_responses=True)
    return EstimationCache(redis_client=client, ttl=300)


# ---------------------------------------------------------------------------
# make_key — determinism and sensitivity
# ---------------------------------------------------------------------------


def test_make_key_is_deterministic() -> None:
    """Same inputs must always produce the same key."""
    key1 = EstimationCache.make_key(**BASE_PARAMS)
    key2 = EstimationCache.make_key(**BASE_PARAMS)
    assert key1 == key2


def test_make_key_has_prefix() -> None:
    key = EstimationCache.make_key(**BASE_PARAMS)
    assert key.startswith("estimation:")


@pytest.mark.parametrize(
    "field, new_value",
    [
        ("system_prompt", "Different system prompt."),
        ("user_message", "Different user message."),
        ("model", "claude-haiku-4-5-20251001"),
        ("max_tokens", 4096),
        ("thinking_budget", 1024),
    ],
)
def test_make_key_changes_when_field_changes(field: str, new_value) -> None:
    """Changing any single field must produce a different cache key."""
    key_original = EstimationCache.make_key(**BASE_PARAMS)
    modified = {**BASE_PARAMS, field: new_value}
    key_modified = EstimationCache.make_key(**modified)
    assert key_original != key_modified, (
        f"Key did not change when '{field}' was modified"
    )


# ---------------------------------------------------------------------------
# get / set roundtrip
# ---------------------------------------------------------------------------


def test_set_and_get_roundtrip(cache: EstimationCache) -> None:
    """A value stored with set() must be returned verbatim by get()."""
    key = EstimationCache.make_key(**BASE_PARAMS)
    cache.set(key, SAMPLE_RESPONSE)
    result = cache.get(key)
    assert result == SAMPLE_RESPONSE


def test_get_returns_none_on_miss(cache: EstimationCache) -> None:
    """get() must return None when the key is not in the cache."""
    key = EstimationCache.make_key(**BASE_PARAMS)
    assert cache.get(key) is None


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


def test_set_applies_ttl(cache: EstimationCache) -> None:
    """set() must store the entry with the configured TTL."""
    key = EstimationCache.make_key(**BASE_PARAMS)
    cache.set(key, SAMPLE_RESPONSE)
    remaining = cache._redis.ttl(key)
    # TTL should be <= configured value and > 0 (FakeRedis honours setex)
    assert 0 < remaining <= cache.ttl


# ---------------------------------------------------------------------------
# Redis error handling — cache errors must never propagate
# ---------------------------------------------------------------------------


def test_get_returns_none_on_redis_error(cache: EstimationCache) -> None:
    """get() must swallow RedisError and return None."""
    key = EstimationCache.make_key(**BASE_PARAMS)
    with patch.object(cache._redis, "get", side_effect=RedisError("connection lost")):
        result = cache.get(key)
    assert result is None


def test_set_does_not_raise_on_redis_error(cache: EstimationCache) -> None:
    """set() must swallow RedisError and return normally."""
    key = EstimationCache.make_key(**BASE_PARAMS)
    with patch.object(cache._redis, "setex", side_effect=RedisError("connection lost")):
        cache.set(key, SAMPLE_RESPONSE)  # must not raise
