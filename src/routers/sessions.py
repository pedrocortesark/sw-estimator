"""Sessions router — session lifecycle and session-scoped estimation."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.dependencies import get_estimation_service
from src.schemas.estimation import EstimationResponse
from src.services.document_extractor import (
    UnsupportedFileTypeError,
    build_attachment_block,
    extract_text,
)
from src.services.estimation import EstimationService
from src.services.sessions import session_store

router = APIRouter(prefix="/api/v1", tags=["Sessions"])


class SessionCreateResponse(BaseModel):
    """Payload returned when a new session is created."""

    session_id: str


class SessionInfoResponse(BaseModel):
    """Snapshot of a session's accumulated metadata and turn count."""

    session_id: str
    turn_count: int
    anchors_count: int
    summary_chars: int
    last_resolved_tier: str
    last_tier_rule: str
    project_metadata: dict


@router.get(
    "/sessions/{session_id}",
    response_model=SessionInfoResponse,
    summary="Get session info and accumulated project metadata",
)
async def get_session(session_id: str) -> SessionInfoResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return SessionInfoResponse(
        session_id=session_id,
        turn_count=session.history._turn_count(),
        anchors_count=len(session.anchors),
        summary_chars=session.summary_chars,
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
        project_metadata=session.metadata.model_dump(),
    )


@router.post(
    "/sessions",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
    description=(
        "Allocates a fresh in-memory session and returns its UUID. "
        "Clients must include this ``session_id`` in every subsequent "
        "request to preserve conversation history and project metadata "
        "across turns."
    ),
)
async def create_session() -> SessionCreateResponse:
    """Create a new session and return its identifier.

    A UUID v4 is used as the session identifier — it is unguessable,
    collision-resistant at the scale of this service, and requires no
    coordination with a shared counter or sequence.
    """
    session_id = str(uuid.uuid4())
    session_store.get_or_create(session_id)
    return SessionCreateResponse(session_id=session_id)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/estimate — multipart estimation with attachments
# ---------------------------------------------------------------------------

_MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024  # 10 MB per file


@router.post(
    "/sessions/{session_id}/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate with optional document attachments",
    description=(
        "Accepts ``multipart/form-data`` with a ``transcript`` field and an optional "
        "list of ``attachments`` (PDF or DOCX).  Attachment text is extracted locally "
        "and appended to the transcript before the LLM call — no provider upload needed."
    ),
)
async def session_estimate(
    session_id: str,
    transcript: Annotated[str, Form(min_length=20, max_length=8000)],
    attachments: Annotated[list[UploadFile] | None, Form()] = None,
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    """POST /api/v1/sessions/{session_id}/estimate

    Workflow:
    1. Validate that the session exists (404 otherwise).
    2. For each uploaded file, read bytes and extract text locally.
    3. Append each extracted block to the transcript with a clear separator.
    4. Delegate to ``EstimationService.estimate_conversational()`` which runs
       the full pipeline, updates session state, and emits a ``turn_observed``
       event with all turn-level observability fields in one place.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Call POST /api/v1/sessions first.",
        )

    enriched = transcript
    attachments_total_chars = 0

    for upload in attachments or []:
        raw = await upload.read()

        if len(raw) > _MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Attachment '{upload.filename}' exceeds the 10 MB limit.",
            )

        try:
            text = extract_text(upload.filename or "attachment", raw)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc

        block = build_attachment_block(upload.filename or "attachment", text)
        enriched += "\n\n" + block
        attachments_total_chars += len(block) + 2  # +2 for the "\n\n" separator

    return await service.estimate_conversational(
        session=session,
        transcript=transcript,
        enriched_transcript=enriched,
        attachments_total_chars=attachments_total_chars,
    )
