"""Contract schemas for the Session 14 multi-agent endpoint.

Defines the request/response models for the multi-agent estimation API.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MultiAgentEstimateRequest(BaseModel):
    """Request payload for POST /v1/estimate/multi-agent."""

    transcript: str = Field(min_length=100, max_length=50_000)
    estimation_id: Optional[str] = Field(default=None, max_length=128)


class MultiAgentEstimateResponse(BaseModel):
    """Response payload for the multi-agent estimation."""

    estimation_id: str
    status: Literal["validated", "needs_review", "awaiting_human_review", "rejected"]
    estimate: Optional[dict] = None
    confidence: Optional[float] = None
    requirements: list[str] = Field(default_factory=list)
    budget_matches: list[dict] = Field(default_factory=list)
    validation: Optional[dict] = None
    human_decision: Optional[dict] = None
    agent_actions: list[dict] = Field(default_factory=list)


class MultiAgentResumeRequest(BaseModel):
    """Request payload for POST /v1/estimate/multi-agent/{id}/resume."""

    decision: dict = Field(
        description="Human decision: {action: 'approve'|'adjust'|'reject', ...}"
    )


class MultiAgentStateResponse(BaseModel):
    """Response payload for GET /v1/estimate/multi-agent/{id}/state."""

    estimation_id: str
    status: Literal["validated", "needs_review", "awaiting_human_review", "rejected", "running"]
    estimate: Optional[dict] = None
    confidence: Optional[float] = None
    requirements: list[str] = Field(default_factory=list)
    budget_matches: list[dict] = Field(default_factory=list)
    validation: Optional[dict] = None
    human_decision: Optional[dict] = None
    agent_actions: list[dict] = Field(default_factory=list)
