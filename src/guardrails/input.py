"""Input guardrails — validate raw user input before it reaches the LLM."""

from __future__ import annotations

from openai import AsyncOpenAI


def check_input(description: str, *, openai_client: AsyncOpenAI | None = None) -> None:
    """Validate the raw transcript before sending it to the LLM.

    Raises:
        ValueError: If the input violates any guardrail rule.
    """
    pass
