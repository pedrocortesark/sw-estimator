"""Requirements extractor agent.

Extracts requirements from the transcript using only the LLM (no business tools).
This agent has NO tool privileges - it can only use the language model.
"""

from __future__ import annotations

import asyncio

import logfire
import structlog

from src.core.config import get_settings
from src.domain.graph.schemas import RequirementsExtraction
from src.domain.multi_agent.state import EstimationState

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a software requirements analyst. Read the meeting transcript and extract "
    "a flat list of concrete, atomic requirements the client wants built. One requirement "
    "per item, concise technical English, regardless of the transcript language. Ignore "
    "small talk, anecdotes and digressions. Never invent requirements the transcript "
    "gives no evidence for."
)


async def requirements_extractor(state: EstimationState) -> dict:
    """Extract requirements from transcript (LLM only, no tools)."""
    with logfire.span("agent: requirements_extractor"):
        settings = get_settings()
        from src.dependencies import get_llm_wrapper

        wrapper = get_llm_wrapper()
        result, _meta = await wrapper.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_message=state["transcript"],
            response_model=RequirementsExtraction,
            model_override=settings.graph_extraction_model,
        )
        requirements = [r.strip() for r in result.requirements if r.strip()]

        log.info(
            "agent_requirements_extractor_done",
            requirements=len(requirements),
        )

        # Audit log entry
        audit_entry = {
            "agent": "requirements_extractor",
            "tool": None,  # No tools used
            "input_summary": f"transcript ({len(state['transcript'])} chars)",
            "output_summary": f"{len(requirements)} requirements extracted",
        }

        return {
            "requirements": requirements,
            "agent_actions": [audit_entry],
        }
