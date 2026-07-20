"""Multi-agent estimation endpoint for Session 14.

Provides endpoints for:
- POST /v1/estimate/multi-agent: Start multi-agent estimation
- POST /v1/estimate/multi-agent/{id}/resume: Resume after human review
- GET /v1/estimate/multi-agent/{id}/state: Get current state
"""

from __future__ import annotations

import uuid

import logfire
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.types import Command

from src.api.deps import get_request_id
from src.api.rate_limiting import limiter
from src.api.security import require_estimate_key
from src.core.config import get_settings
from src.domain.multi_agent.schemas import (
    MultiAgentEstimateRequest,
    MultiAgentEstimateResponse,
    MultiAgentResumeRequest,
    MultiAgentStateResponse,
)

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate", tags=["estimate-multi-agent"])


def _require_multi_agent_graph(request: Request, request_id: str):
    """Get the multi-agent graph from app state."""
    graph = getattr(request.app.state, "multi_agent_graph", None)
    if graph is None:
        log.error("multi_agent_graph_unavailable", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail="Multi-agent graph is not available.",
        )
    return graph


def _build_response(estimation_id: str, snapshot) -> MultiAgentEstimateResponse:
    """Build response from graph snapshot."""
    values = snapshot.values or {}
    paused = bool(snapshot.next)

    # Determine status
    if paused:
        status = "awaiting_human_review"
    else:
        status = values.get("status", "needs_review")

    return MultiAgentEstimateResponse(
        estimation_id=estimation_id,
        status=status,
        estimate=values.get("estimate"),
        confidence=values.get("confidence"),
        requirements=values.get("requirements", []),
        budget_matches=values.get("budget_matches", []),
        validation=values.get("validation"),
        human_decision=values.get("human_decision"),
        agent_actions=values.get("agent_actions", []),
    )


@router.post(
    "/multi-agent",
    response_model=MultiAgentEstimateResponse,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def estimate_multi_agent(
    request: Request,
    payload: MultiAgentEstimateRequest,
) -> MultiAgentEstimateResponse:
    """Start multi-agent estimation.

    Runs the supervisor/workers graph until completion or human review pause.
    Returns the final estimate or pauses for human review if confidence is low.
    """
    request_id = get_request_id(request)
    graph = _require_multi_agent_graph(request, request_id)

    estimation_id = payload.estimation_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": estimation_id}}

    try:
        with logfire.span("multi_agent_estimate_start", estimation_id=estimation_id):
            # Run the graph
            await graph.ainvoke(
                {
                    "transcript": payload.transcript,
                    "estimation_id": estimation_id,
                    "requirements": [],
                    "budget_matches": [],
                    "agent_actions": [],
                    "awaiting_review": False,
                },
                config,
            )

            # Get final state
            snapshot = await graph.aget_state(config)

    except Exception as exc:
        log.error(
            "multi_agent_estimate_failed",
            request_id=request_id,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=502,
            detail="Multi-agent estimation failed.",
        ) from exc

    return _build_response(estimation_id, snapshot)


@router.post(
    "/multi-agent/{estimation_id}/resume",
    response_model=MultiAgentEstimateResponse,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def resume_multi_agent(
    request: Request,
    estimation_id: str,
    payload: MultiAgentResumeRequest,
) -> MultiAgentEstimateResponse:
    """Resume a paused multi-agent estimation with human decision.

    Continues the graph from the checkpoint with the human's decision.
    """
    request_id = get_request_id(request)
    graph = _require_multi_agent_graph(request, request_id)
    config = {"configurable": {"thread_id": estimation_id}}

    # Check if the graph is actually paused
    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail="Estimation is not paused (already completed or unknown).",
        )

    try:
        with logfire.span("multi_agent_estimate_resume", estimation_id=estimation_id):
            # Resume with human decision
            await graph.ainvoke(Command(resume=payload.decision), config)

            # Get updated state
            snapshot = await graph.aget_state(config)

    except Exception as exc:
        log.error(
            "multi_agent_resume_failed",
            request_id=request_id,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=502,
            detail="Failed to resume multi-agent estimation.",
        ) from exc

    return _build_response(estimation_id, snapshot)


@router.get(
    "/multi-agent/{estimation_id}/state",
    response_model=MultiAgentStateResponse,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def get_multi_agent_state(
    request: Request,
    estimation_id: str,
) -> MultiAgentStateResponse:
    """Get the current state of a multi-agent estimation.

    Returns the state snapshot including any pending human review.
    """
    request_id = get_request_id(request)
    graph = _require_multi_agent_graph(request, request_id)
    config = {"configurable": {"thread_id": estimation_id}}

    snapshot = await graph.aget_state(config)

    if not snapshot.created_at and not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail="Unknown estimation_id.",
        )

    values = snapshot.values or {}
    paused = bool(snapshot.next)

    # Determine status
    if paused:
        status = "awaiting_human_review"
    else:
        status = values.get("status", "running")

    return MultiAgentStateResponse(
        estimation_id=estimation_id,
        status=status,
        estimate=values.get("estimate"),
        confidence=values.get("confidence"),
        requirements=values.get("requirements", []),
        budget_matches=values.get("budget_matches", []),
        validation=values.get("validation"),
        human_decision=values.get("human_decision"),
        agent_actions=values.get("agent_actions", []),
    )
