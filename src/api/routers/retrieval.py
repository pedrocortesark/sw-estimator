"""``POST /v1/retrieval/search`` — metadata-filtered semantic retrieval (S09/S10).

Canonical search endpoint: it embeds the query text and runs k-NN with a
relevance threshold plus optional structural filters (sector / project year /
chunk type). Session 10 adds hybrid search (vector + lexical + RRF) and
optional cross-encoder reranking.

Thin transport: validation lives in ``RetrievalRequest`` (422), auth in
``require_retrieval_key`` (401), rate limiting in the ``@limiter`` decorator
(429). Soft-fail (nothing crosses the threshold) is a 200 with
``low_confidence=true``, not an error.
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.rate_limiting import limiter
from src.api.security import require_retrieval_key
from src.core.config import get_settings
from src.dependencies import get_embedder
from src.generation.rag.errors import RetrievalError
from src.generation.rag.retrieval.pipeline import retrieve
from src.generation.rag.schemas import RetrievalRequest, RetrievalResult

log = structlog.get_logger()

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResult,
    dependencies=[Depends(require_retrieval_key)],
)
@limiter.limit("120/minute")
async def search(request: Request, payload: RetrievalRequest) -> RetrievalResult:
    """Return chunks within ``distance_threshold`` of the embedded query text.
    
    Session 10: supports hybrid search and reranking via payload parameters.
    If ``search_mode`` or ``rerank`` are None, falls back to settings defaults.
    """
    settings = get_settings()
    embedder = get_embedder()
    if embedder is None:
        log.error("retrieval_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    # Resolve None values to settings defaults
    search_mode = payload.search_mode or "vector"
    rerank = payload.rerank if payload.rerank is not None else settings.reranker_enabled

    try:
        query_embedding = await asyncio.to_thread(embedder.embed_one, payload.query_text)
        return await retrieve(
            query_embedding=query_embedding,
            query_text=payload.query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=payload.top_k,
            distance_threshold=payload.distance_threshold,
            sectors=payload.sectors,
            project_year_min=payload.project_year_min,
            project_year_max=payload.project_year_max,
            chunk_types=payload.chunk_types,
        )
    except RetrievalError as exc:
        raise HTTPException(status_code=502, detail="Retrieval failed.") from exc
    except Exception as exc:  # noqa: BLE001 — embedding/other failures → 502.
        log.error(
            "retrieval_failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=502, detail="Failed to run retrieval.") from exc
