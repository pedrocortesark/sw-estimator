"""Prompt template loader — renders Jinja2 templates into (system, user) prompt pairs."""

from __future__ import annotations

from src.schemas.estimation import EstimationRequest


def render_estimation_prompt(
    request: EstimationRequest, version: str = "v1"
) -> tuple[str, str]:
    """Render the system and user prompts for the given request.

    Args:
        request: Validated estimation request.
        version: Template version directory name (e.g. ``"v1"``).

    Returns:
        A ``(system_prompt, user_prompt)`` tuple.
    """
    return ("", "")
