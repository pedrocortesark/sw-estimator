"""Sessions router — POST /api/v1/sessions."""

import uuid

from fastapi import APIRouter, status
from pydantic import BaseModel

from src.services.sessions import session_store

router = APIRouter(prefix="/api/v1", tags=["Sessions"])


class SessionCreateResponse(BaseModel):
    """Payload returned when a new session is created."""

    session_id: str


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
