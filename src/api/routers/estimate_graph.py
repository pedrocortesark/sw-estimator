"""``POST /v1/estimate/graph`` — transcript → estimate via LangGraph (S13).

LangGraph-based endpoint that replaces the hand-written agent loop from S12.
The graph has five nodes:
1. extract_requirements - Extract requirements from transcript
2. classify_components - Group requirements into components
3. search_budgets - Find reference budgets for each component
4. generate_estimate - Generate estimation from budgets
5. validate_and_consolidate - Validate and set final status

The endpoint maintains the same contract as /from-transcript:
- Input: transcript
- Output: estimate with status

The graph uses AsyncPostgresSaver for persistence and Logfire for observability.
"""

from __future__ import annotations

import uuid

import logfire
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.rate_limiting import limiter
from src.api.security import require_estimate_key
from src.graph import build_graph
from src.graph.checkpoint import get_checkpointer
from src.generation.rag.schemas import Estimate, EstimateRequest

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate", tags=["estimate-graph"])


@router.post(
    "/graph",
    response_model=Estimate,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def from_transcript_graph(request: Request, payload: EstimateRequest) -> Estimate:
    """Produce an estimate using the LangGraph pipeline.

    The graph runs sequentially (parallel in the live session).
    State is persisted via AsyncPostgresSaver.
    """
    try:
        # Generate estimation ID for thread tracking
        estimation_id = payload.idempotency_key or str(uuid.uuid4())

        # Get checkpointer and build graph
        checkpointer = get_checkpointer()
        graph = build_graph(checkpointer)

        # Setup checkpointer tables on first run
        await checkpointer.setup()

        # Configure execution with thread_id for persistence
        config = {"configurable": {"thread_id": estimation_id}}

        # Initial state
        initial_state = {
            "transcript": payload.transcript,
            "requirements": [],
            "components": [],
            "budget_matches": [],
            "estimate": None,
            "status": None,
            "errors": [],
            "estimation_id": estimation_id,
        }

        # Execute graph
        with logfire.span("graph_execution", estimation_id=estimation_id):
            result = await graph.ainvoke(initial_state, config)

        # Extract estimate from result
        estimate_data = result.get("estimate")
        status = result.get("status", "unknown")

        if not estimate_data:
            raise ValueError("Graph did not produce an estimate")

        # Convert to Estimate schema
        # TODO: Map graph output to Estimate schema properly
        estimate = Estimate(
            total_engineer_days=int(estimate_data.get("total_amount", 0) / 100),
            modules=[],
            duration_weeks=None,
            sources=[],
            assumptions=[],
            confidence="medium",
            reasoning=f"Graph-based estimation (status: {status})",
            insufficient_context_explanation=None,
        )

        logfire.info(
            "graph_execution_complete",
            estimation_id=estimation_id,
            status=status,
            total_amount=estimate_data.get("total_amount"),
        )

        return estimate

    except Exception as exc:
        log.error(
            "graph_execution_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=502,
            detail="Failed to produce an estimate via graph.",
        ) from exc
