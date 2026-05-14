"""Prompt loader for the estimation service.

Renders Jinja2 templates from the prompts directory and returns
(system, user) strings ready to be sent to the LLM.

Directory convention:
    src/prompts/<domain>/<version>/system.j2
    src/prompts/<domain>/<version>/user.j2
    src/prompts/<domain>/<version>/examples.j2  (included by system.j2)

Prompt style
------------
Templates support two structural styles:
- "xml"      — sections wrapped in XML tags, as recommended by Anthropic.
- "markdown" — sections delimited by ## headers and ---, as preferred by OpenAI.

The style is resolved in this priority order:
  1. Explicit ``prompt_style`` argument (overrides everything).
  2. Auto-inferred from ``model`` name: models containing "claude" → "xml",
     everything else → "markdown".
  3. Default fallback when neither is provided: "markdown".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.schemas.estimation import EstimationRequest

_PROMPTS_DIR = Path(__file__).parent
_log = structlog.get_logger(__name__)

PromptStyle = Literal["xml", "markdown"]


def _infer_prompt_style(model: str | None) -> PromptStyle:
    """Derive the prompt style from the model name.

    Anthropic models ("claude-*") perform better with XML tags.
    All other models default to Markdown headers.
    """
    if model and "claude" in model.lower():
        return "xml"
    return "markdown"


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
    model: str | None = None,
    prompt_style: PromptStyle | None = None,
) -> tuple[str, str]:
    """Render system and user prompts for the estimation domain.

    Args:
        request:      The validated estimation request from the client.
        version:      Template version subfolder (default "v1"). Pass a
                      different value to adopt a new prompt version without
                      touching any other module.
        model:        The model identifier that will be used (e.g.
                      "anthropic/claude-3-5-haiku-20241022"). Used to
                      auto-infer ``prompt_style`` when it is not provided.
        prompt_style: Explicit style override — "xml" or "markdown".
                      When omitted, the style is inferred from ``model``.

    Returns:
        A (system, user) tuple of fully-rendered strings.
    """
    resolved_style: PromptStyle = (
        prompt_style if prompt_style is not None else _infer_prompt_style(model)
    )

    template_dir = _PROMPTS_DIR / "estimation" / version
    if not template_dir.exists():
        available = sorted(
            p.name for p in (_PROMPTS_DIR / "estimation").iterdir() if p.is_dir()
        )
        raise ValueError(f"Unknown prompt version {version!r}. Available: {available}")
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    context = {
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "description": request.description,
        "prompt_style": resolved_style,
        "reference_projects": [rp.model_dump() for rp in request.reference_projects]
        if request.reference_projects
        else [],
    }

    system = env.get_template("system.j2").render(**context)
    user = env.get_template("user.j2").render(**context)

    content_hash = hashlib.sha256((system + user).encode()).hexdigest()[:12]
    _log.debug(
        "prompt_rendered",
        version=version,
        prompt_style=resolved_style,
        content_hash=content_hash,
        system_chars=len(system),
        user_chars=len(user),
        has_reference_projects=bool(context["reference_projects"]),
    )

    return system, user
