"""Tests for src/cache/semantic.py — EstimationSemanticCache.

Test strategy
-------------
We never start a real Redis Stack. Instead, every test builds the cache via
``object.__new__`` (bypassing ``__init__`` and its ``SearchIndex.create``
call), then wires:
- ``cache.index``      → ``MagicMock()`` (controls what ``index.query`` returns)
- ``cache.vectorizer`` → ``SimpleNamespace(embed=lambda: [0.1]*1536)``

This lets us test all logic branches — bucket building, threshold comparison,
log_only mode, and store field assembly — without a live database.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from src.cache.semantic import EstimationSemanticCache
from src.schemas.estimation import (
    EstimationRequest,
    EstimationResult,
    Phase,
    Task,
    TeamMember,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_TRANSCRIPT = "Build a dashboard for project analytics."  # ≥20 chars
_DUMMY_VECTOR = [0.1] * 1536
_VECTORIZER = SimpleNamespace(embed=lambda text: _DUMMY_VECTOR)


def _make_cache(
    *, threshold: float = 0.92, log_only: bool = False
) -> EstimationSemanticCache:
    """Return a cache instance with a MagicMock index — no Redis needed."""
    cache = object.__new__(EstimationSemanticCache)
    cache.threshold = threshold
    cache.ttl = 300
    cache.log_only = log_only
    cache.vectorizer = _VECTORIZER
    cache.index = MagicMock()
    return cache


def _make_request(**kwargs) -> EstimationRequest:
    """Return an EstimationRequest with optional extra attributes."""
    base = {"transcript": "Build a mobile app for booking medical appointments."}
    req = EstimationRequest(
        **(base | {k: v for k, v in kwargs.items() if k in ("transcript", "provider")})
    )
    # Attach future bucket fields as plain attributes
    for k, v in kwargs.items():
        if k not in ("transcript", "provider"):
            object.__setattr__(req, k, v)
    return req


_MINIMAL_RESULT = EstimationResult(
    executive_summary="Simple app — 40 h.",
    phases=[
        Phase(
            name="Backend",
            tasks=[Task(name="API", hours=40.0, cost_usd=4000.0)],
            total_hours=40.0,
            total_cost_usd=4000.0,
        )
    ],
    total_hours=40.0,
    total_cost_usd=4000.0,
    team_composition=[TeamMember(role="Engineer", count=1, dedication="100%")],
    duration_weeks=2.0,
)


def _hit_result(similarity: float) -> list[dict]:
    """Return a mock index.query result list with the given similarity."""
    # redisvl returns distance = 1 - similarity (cosine metric in [0,2])
    distance = 1.0 - similarity
    return [
        {
            "result_json": _MINIMAL_RESULT.model_dump_json(),
            "vector_distance": str(distance),
        }
    ]


# ---------------------------------------------------------------------------
# bucket_for — determinism and sensitivity
# ---------------------------------------------------------------------------


def test_bucket_for_contains_all_four_fields() -> None:
    req = _make_request(
        project_type="saas", detail_level="detailed", output_format="json"
    )
    bucket = EstimationSemanticCache.bucket_for(req, prompt_version="v1")
    assert "v1" in bucket
    assert "saas" in bucket
    assert "detailed" in bucket
    assert "json" in bucket


def test_bucket_for_is_deterministic() -> None:
    req = _make_request(
        project_type="saas", detail_level="detailed", output_format="json"
    )
    assert EstimationSemanticCache.bucket_for(
        req, "v1"
    ) == EstimationSemanticCache.bucket_for(req, "v1")


@pytest.mark.parametrize(
    "field, original, changed",
    [
        ("prompt_version", "v1", "v2"),
        ("project_type", "saas", "mobile"),
        ("detail_level", "detailed", "summary"),
        ("output_format", "json", "markdown"),
    ],
)
def test_bucket_for_changes_when_field_changes(
    field: str, original: str, changed: str
) -> None:
    kwargs = {
        "project_type": "saas",
        "detail_level": "detailed",
        "output_format": "json",
    }
    pv = "v1"
    if field == "prompt_version":
        req = _make_request(**kwargs)
        b1 = EstimationSemanticCache.bucket_for(req, pv)
        b2 = EstimationSemanticCache.bucket_for(req, changed)
    else:
        req1 = _make_request(**{**kwargs, field: original})
        req2 = _make_request(**{**kwargs, field: changed})
        b1 = EstimationSemanticCache.bucket_for(req1, pv)
        b2 = EstimationSemanticCache.bucket_for(req2, pv)
    assert b1 != b2, f"Bucket did not change when '{field}' was modified"


def test_bucket_for_falls_back_to_default_for_missing_fields() -> None:
    """EstimationRequest without extra attrs should still produce a valid bucket."""
    req = EstimationRequest(transcript=_TRANSCRIPT)
    bucket = EstimationSemanticCache.bucket_for(req, "v1")
    assert bucket == "v1:default:default:default"


# ---------------------------------------------------------------------------
# lookup — miss: empty index
# ---------------------------------------------------------------------------


def test_lookup_returns_none_on_empty_index() -> None:
    cache = _make_cache()
    cache.index.query.return_value = []  # no stored entries
    req = EstimationRequest(transcript=_TRANSCRIPT)
    assert cache.lookup(req) is None


# ---------------------------------------------------------------------------
# lookup — miss: below threshold
# ---------------------------------------------------------------------------


def test_lookup_returns_none_when_below_threshold() -> None:
    cache = _make_cache(threshold=0.92)
    # similarity 0.85 < 0.92 → miss
    cache.index.query.return_value = _hit_result(similarity=0.85)
    req = EstimationRequest(transcript=_TRANSCRIPT)
    assert cache.lookup(req) is None


# ---------------------------------------------------------------------------
# lookup — hit: similarity above threshold
# ---------------------------------------------------------------------------


def test_lookup_returns_result_on_hit() -> None:
    cache = _make_cache(threshold=0.92)
    # similarity 0.97 ≥ 0.92 → hit
    cache.index.query.return_value = _hit_result(similarity=0.97)
    req = EstimationRequest(transcript=_TRANSCRIPT)
    result = cache.lookup(req)
    assert isinstance(result, EstimationResult)
    assert result.total_hours == _MINIMAL_RESULT.total_hours


def test_lookup_deserialises_result_correctly() -> None:
    cache = _make_cache(threshold=0.92)
    cache.index.query.return_value = _hit_result(similarity=0.99)
    req = EstimationRequest(transcript=_TRANSCRIPT)
    result = cache.lookup(req)
    # Deep field check
    assert result.executive_summary == _MINIMAL_RESULT.executive_summary
    assert result.phases[0].name == "Backend"


# ---------------------------------------------------------------------------
# lookup — log_only mode: logs hit but still returns None
# ---------------------------------------------------------------------------


def test_lookup_returns_none_in_log_only_mode_despite_hit() -> None:
    cache = _make_cache(threshold=0.92, log_only=True)
    # similarity 0.98 would be a hit if log_only were False
    cache.index.query.return_value = _hit_result(similarity=0.98)
    req = EstimationRequest(transcript=_TRANSCRIPT)
    # Must return None even though similarity exceeds threshold
    assert cache.lookup(req) is None


def test_lookup_real_hit_not_affected_by_log_only_false() -> None:
    """Sanity: same setup without log_only=True must return the result."""
    cache = _make_cache(threshold=0.92, log_only=False)
    cache.index.query.return_value = _hit_result(similarity=0.98)
    req = EstimationRequest(transcript=_TRANSCRIPT)
    assert cache.lookup(req) is not None


# ---------------------------------------------------------------------------
# store — calls index.load with correct fields
# ---------------------------------------------------------------------------


def test_store_calls_index_load_once() -> None:
    cache = _make_cache()
    req = EstimationRequest(transcript=_TRANSCRIPT)
    cache.store(req, _MINIMAL_RESULT)
    cache.index.load.assert_called_once()


def test_store_passes_correct_fields_to_load() -> None:
    cache = _make_cache()
    req = EstimationRequest(transcript=_TRANSCRIPT)
    cache.store(req, _MINIMAL_RESULT)

    data: list[dict] = cache.index.load.call_args.args[0]  # first positional arg
    entry = data[0]

    assert "bucket" in entry
    assert "result_json" in entry
    assert "embedding" in entry
    assert "id" in entry

    # result_json must round-trip to an equivalent EstimationResult
    roundtripped = EstimationResult.model_validate_json(entry["result_json"])
    assert roundtripped.total_hours == _MINIMAL_RESULT.total_hours


def test_store_passes_ttl_to_load() -> None:
    cache = _make_cache()
    req = EstimationRequest(transcript=_TRANSCRIPT)
    cache.store(req, _MINIMAL_RESULT)

    kwargs = cache.index.load.call_args.kwargs
    assert kwargs.get("ttl") == cache.ttl


def test_store_does_not_raise_on_index_error() -> None:
    """store() must swallow exceptions — the cache is best-effort."""
    from redis import RedisError

    cache = _make_cache()
    cache.index.load.side_effect = RedisError("connection lost")
    req = EstimationRequest(transcript=_TRANSCRIPT)
    cache.store(req, _MINIMAL_RESULT)  # must not raise
