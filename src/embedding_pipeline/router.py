"""FastAPI router for POST /embeddings/ingest."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status

from src.embedding_pipeline.chunker import JSONStructuralChunker
from src.embedding_pipeline.embedder import OpenAIEmbedder
from src.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])

_chunker = JSONStructuralChunker()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk and embed budget documents",
    description=(
        "Receives one or more pre-cleaned budget documents, splits each into "
        "structural chunks (one per BudgetComponent) and embeds them with "
        "text-embedding-3-small. Returns the vectors in memory — no persistence yet."
    ),
)
def ingest(request: IngestRequest) -> IngestResponse:
    """POST /embeddings/ingest"""
    bound = log.bind(total_budgets=len(request.budgets))
    bound.info("embeddings.ingest.started")

    # 1. Chunk
    chunks = _chunker.chunk(request.budgets)
    bound.info("embeddings.ingest.chunked", total_chunks=len(chunks))

    # 2. Embed (errors from the OpenAI API are caught here)
    try:
        embedder = OpenAIEmbedder()
        estimated_cost = embedder.estimate_cost(chunks)
        embedded = embedder.embed_many(chunks)
    except Exception as exc:
        bound.error("embeddings.ingest.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding service error. Check server logs for details.",
        ) from exc

    # 3. Assemble stats
    total_tokens = sum(c.token_count for c in chunks)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
    )

    bound.info(
        "embeddings.ingest.done",
        total_chunks=stats.total_chunks,
        total_tokens=stats.total_tokens,
        estimated_cost_usd=stats.estimated_cost_usd,
    )

    return IngestResponse(chunks=embedded, stats=stats)
