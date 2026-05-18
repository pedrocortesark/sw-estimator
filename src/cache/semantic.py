"""Semantic cache — return a cached EstimationResult for *similar* transcripts.

Why two conditions for a cache hit
------------------------------------
Exact-match caching (Capa 2) only fires when the rendered prompt is bit-for-bit
identical across requests. Semantic caching extends that to *similar* natural
language: "mobile app for booking medical appointments" and "medical appointment
scheduler for Android" should return the same estimation.

Two guards prevent false positives:

1. **Bucket equality** — ``{prompt_version}:{project_type}:{detail_level}:{output_format}``
   partitions the index by *context*. Even if two descriptions are semantically
   close, a change in the output format (e.g. "detailed" vs "executive-summary")
   produces a different bucket and therefore a guaranteed miss. This means the
   cache never serves a result generated under a different prompt template.

2. **Cosine similarity ≥ threshold** (default 0.92) — the embedding distance
   measures *semantic* proximity. RediSearch returns a ``vector_distance`` in
   ``[0, 2]`` (0 = identical, 2 = opposite) using the cosine metric, so
   ``similarity = 1.0 - distance``. A 0.92 threshold means the descriptions
   must share ≥ 92 % of their directional meaning — tight enough to avoid
   cross-domain matches, lenient enough to absorb rephrasing.

Why Redis Stack (not vanilla Redis)
--------------------------------------
RediSearch (bundled in Redis Stack) implements the HNSW/FLAT vector index that
makes KNN queries possible. Vanilla ``redis:alpine`` has no FT.CREATE or
FT.SEARCH commands, so the ``create()`` call would raise a ``ResponseError``.
Use ``redis/redis-stack`` or ``redis/redis-stack-server`` in docker-compose.

Why log_only mode
-------------------
Before enabling the cache in production, run it in *shadow* mode:
``log_only=True`` logs every hit but still calls the LLM, letting you inspect
the threshold calibration without serving stale results.

Why float32 bytes
-------------------
Redis stores vectors as raw bytes. redisvl 0.18 accepts ``list[float]``
directly (and converts internally), but explicit ``numpy.float32`` conversion
makes the dtype contract visible at the call site and avoids surprises if the
vectorizer returns ``float64`` arrays.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import numpy as np
import structlog
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema

from src.schemas.estimation import EstimationRequest, EstimationResult

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Schema definition (factored out so tests can inspect it in isolation)
# ---------------------------------------------------------------------------


def _build_schema(index_name: str) -> IndexSchema:
    """Build the redisvl IndexSchema for the semantic cache.

    Fields:
    - ``bucket``      (tag)    — bucket key for pre-filter
    - ``result_json`` (text)   — serialised EstimationResult
    - ``embedding``   (vector) — FLAT cosine index, float32, 1536 dims
      (text-embedding-3-small output size)
    """
    return IndexSchema.from_dict(
        {
            "index": {"name": index_name, "prefix": f"sem:{index_name}:"},
            "fields": [
                {"name": "bucket", "type": "tag"},
                {"name": "result_json", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _to_float32_bytes(vector: list[float]) -> bytes:
    """Convert a list of floats to a contiguous float32 byte array.

    redisvl accepts list[float] directly, but explicit numpy conversion
    guarantees the float32 dtype and makes the contract clear.
    """
    return np.array(vector, dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# EstimationSemanticCache
# ---------------------------------------------------------------------------


class EstimationSemanticCache:
    """Redis Stack–backed semantic cache for LLM estimation responses.

    Args:
        redis_client: A synchronous ``redis.Redis`` instance
            (*must* connect to Redis Stack, not vanilla Redis).
        vectorizer: Any object with ``embed(text: str) -> list[float]``.
            Typically an OpenAI ``text-embedding-3-small`` wrapper.
        threshold: Minimum cosine similarity for a cache hit. Default 0.92.
        ttl: Time-to-live in seconds for stored entries. Default 86 400 (24 h).
        log_only: When ``True``, log hits but still return ``None`` — useful
            for shadow-testing threshold calibration in production.
        index_name: Name of the RediSearch index. Default ``"estimations"``.
    """

    def __init__(
        self,
        redis_client,
        vectorizer,
        *,
        threshold: float = 0.92,
        ttl: int = 86_400,
        log_only: bool = False,
        index_name: str = "estimations",
    ) -> None:
        self.threshold = threshold
        self.ttl = ttl
        self.log_only = log_only
        self.vectorizer = vectorizer

        schema = _build_schema(index_name)
        self.index = SearchIndex(schema, redis_client=redis_client)

        # Create the index if it does not exist yet. ``overwrite=False`` is the
        # safe default — it raises if the index already exists, so we swallow
        # that specific case. Any other exception propagates (misconfigured
        # Redis, missing RediSearch module, etc.).
        try:
            self.index.create(overwrite=False)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "index already exists" in err_msg or "already exists" in err_msg:
                logger.debug("semantic_cache_index_exists", index=index_name)
            else:
                raise

    # ------------------------------------------------------------------
    # Bucket
    # ------------------------------------------------------------------

    @staticmethod
    def bucket_for(
        request: EstimationRequest | Any,
        prompt_version: str,
    ) -> str:
        """Build the bucket key that partitions the vector index by context.

        The bucket is a ``":"``-separated string of four fields:
        ``{prompt_version}:{project_type}:{detail_level}:{output_format}``

        Fields not present on *request* fall back to ``"default"`` so the
        method is forward-compatible with future ``EstimationRequest`` extensions
        without breaking callers that pass the current schema.

        Two requests with different buckets will **never** share a cache entry,
        even if their transcripts are semantically identical — because the
        rendered prompts would differ.
        """
        return ":".join(
            [
                prompt_version,
                getattr(request, "project_type", "default"),
                getattr(request, "detail_level", "default"),
                getattr(request, "output_format", "default"),
            ]
        )

    # ------------------------------------------------------------------
    # Lookup (read)
    # ------------------------------------------------------------------

    def lookup(
        self,
        request: EstimationRequest,
        prompt_version: str = "v1",
    ) -> EstimationResult | None:
        """Return a cached ``EstimationResult`` for a semantically similar request.

        Returns ``None`` on:
        - Empty index (no stored entries)
        - Best match distance above ``1 - threshold``
        - ``log_only=True`` (shadow mode — logs the hit but still returns None)
        - Any Redis error (swallowed; the pipeline falls through to the LLM)

        Args:
            request: The incoming estimation request.
            prompt_version: Template version used to render the prompt; included
                in the bucket key.

        Returns:
            Deserialised ``EstimationResult`` on a real hit, ``None`` otherwise.
        """
        bucket = self.bucket_for(request, prompt_version)

        try:
            raw_vector = self.vectorizer.embed(request.transcript)
            # Explicit float32 bytes — see module docstring for rationale
            embedding_bytes = _to_float32_bytes(raw_vector)

            query = VectorQuery(
                vector=embedding_bytes,
                vector_field_name="embedding",
                return_fields=["result_json", VectorQuery.DISTANCE_ID],
                filter_expression=Tag("bucket") == bucket,
                num_results=1,
                return_score=True,
            )
            results = self.index.query(query)
        except Exception as exc:
            logger.warning("semantic_cache_error", operation="lookup", error=str(exc))
            return None

        if not results:
            logger.debug("semantic_cache_miss", reason="no_results", bucket=bucket)
            return None

        best = results[0]
        # ``vector_distance`` is in [0, 2]: 0 = identical, 2 = opposite.
        # We invert it to get a [−1, 1] similarity where 1 = identical.
        distance = float(best.get(VectorQuery.DISTANCE_ID, 2.0))
        similarity = 1.0 - distance

        if similarity < self.threshold:
            logger.debug(
                "semantic_cache_miss",
                reason="below_threshold",
                similarity=round(similarity, 4),
                threshold=self.threshold,
                bucket=bucket,
            )
            return None

        # Hit confirmed — log regardless of log_only mode
        logger.info(
            "semantic_cache_hit",
            similarity=round(similarity, 4),
            threshold=self.threshold,
            bucket=bucket,
            log_only=self.log_only,
        )

        if self.log_only:
            # Shadow mode: record what *would* have been a hit but let the
            # pipeline proceed to a fresh LLM call for calibration purposes.
            return None

        return EstimationResult.model_validate_json(best["result_json"])

    # ------------------------------------------------------------------
    # Store (write)
    # ------------------------------------------------------------------

    def store(
        self,
        request: EstimationRequest,
        result: EstimationResult,
        prompt_version: str = "v1",
    ) -> None:
        """Persist an ``EstimationResult`` so it can be retrieved by similar future requests.

        The entry is stored as a Redis hash with three fields:
        - ``bucket``      — pre-filter tag (``bucket_for`` output)
        - ``result_json`` — ``EstimationResult`` serialised via ``model_dump_json``
        - ``embedding``   — float32 bytes of the transcript embedding

        TTL is applied via ``SearchIndex.load``'s built-in ``ttl`` parameter,
        which calls ``EXPIRE`` on each key after the hash is written — no
        separate round-trip needed.

        Errors are swallowed: the cache is best-effort and must not break the
        pipeline.
        """
        bucket = self.bucket_for(request, prompt_version)

        try:
            raw_vector = self.vectorizer.embed(request.transcript)
            embedding_bytes = _to_float32_bytes(raw_vector)

            entry: dict[str, Any] = {
                "id": str(uuid.uuid4()),  # unique key per stored entry
                "bucket": bucket,
                "result_json": result.model_dump_json(),
                "embedding": embedding_bytes,
            }
            # ``id_field="id"`` tells redisvl to use entry["id"] as the hash key
            # suffix; ttl is applied by redisvl after the HSET command.
            self.index.load([entry], id_field="id", ttl=self.ttl)
        except Exception as exc:
            logger.warning("semantic_cache_error", operation="store", error=str(exc))
            return

        logger.info(
            "semantic_cache_stored",
            bucket=bucket,
            ttl=self.ttl,
            threshold=self.threshold,
        )
