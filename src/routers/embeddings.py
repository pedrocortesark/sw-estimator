"""HTTP layer for the embedding pipeline.

Thin router: it maps service exceptions to status codes. The chunk → embed →
persist orchestration lives in ``RagIngestService``; no business logic here.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.dependencies import get_rag_ingest_service
from src.generation.rag.ingest_service import DuplicateDocumentError, RagIngestService
from src.generation.rag.schemas import IngestRequest, IngestResponse

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={409: {"description": "Document already ingested"}},
)
async def ingest(
    request: IngestRequest,
    service: RagIngestService | None = Depends(get_rag_ingest_service),
) -> IngestResponse | JSONResponse:
    """Persist one budget as a document + embedded chunks (one transaction)."""
    if service is None:
        # No OPENAI_API_KEY configured. Generic message to the client, detail logged.
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    log.info(
        "embeddings_ingest_received",
        source_path=request.source_path,
        document_type=request.document_type,
    )
    try:
        return await service.ingest(
            source_path=request.source_path,
            document_type=request.document_type,
            budget=request.content,
        )
    except DuplicateDocumentError as exc:
        # JSONResponse (not HTTPException) to keep the exercise's literal
        # top-level shape: {"detail": ..., "document_id": ...}.
        log.info(
            "embeddings_ingest_duplicate",
            source_path=request.source_path,
            document_id=exc.document_id,
        )
        return JSONResponse(
            status_code=409,
            content={"detail": "Document already ingested", "document_id": exc.document_id},
        )
    except Exception as exc:  # noqa: BLE001 — embedding/DB failures become a 500.
        log.error(
            "embeddings_ingest_failed",
            reason="ingest_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc
