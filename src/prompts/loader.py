"""Prompt template loader — renders Jinja2 templates into (system, user) prompt pairs.

Design notes
------------
* ``FileSystemLoader`` points at this file's own directory so template paths are
  always relative to the package, never to the caller's cwd.
* ``StrictUndefined`` makes missing context variables raise ``UndefinedError`` at
  render time rather than silently becoming empty strings — typos surface immediately.
* ``trim_blocks`` removes the newline right after a ``{% ... %}`` tag so that
  control-flow lines don't add blank rows to the rendered output.
* ``lstrip_blocks`` strips leading spaces/tabs before ``{% %}`` tags, letting you
  indent template logic for readability without those indents leaking into the
  rendered string.
* ``autoescape=False`` because prompts are plain text, not HTML — escaping
  characters like ``<``, ``&`` or ``>`` would corrupt the rendered content.
* ``keep_trailing_newline=True`` preserves the final newline in template files,
  which avoids confusing diffs and matches UNIX conventions.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from src.schemas.estimation import EstimationRequest

# Root of all templates = the directory that contains this file.
_TEMPLATES_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
    keep_trailing_newline=True,
)


def _infer_prompt_style(model: str | None) -> str:
    """Infer whether to use XML or Markdown prompt style based on the model name.

    Claude models prefer XML-structured prompts; all other models default to
    Markdown.

    Returns:
        ``"xml"`` for Anthropic/Claude models, ``"markdown"`` otherwise.
    """
    if model and "claude" in model.lower():
        return "xml"
    return "markdown"


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
    model: str | None = None,
    prompt_style: str | None = None,
    project_metadata=None,
) -> tuple[str, str]:
    """Render the system and user prompts for the given request.

    Args:
        request: Validated estimation request.
        version: Template version directory name (e.g. ``"v1"``).

    Returns:
        A ``(system_prompt, user_prompt)`` tuple.
    """

    # Build context; fields that don't exist yet on the model use safe defaults
    # so old request objects remain compatible when new fields are added later.
    def _val(field: str, default: str) -> str:
        v = getattr(request, field, default)
        return v.value if hasattr(v, "value") else str(v)

    ctx = {
        "description": request.transcript,
        "transcript": request.transcript,
        "project_type": _val("project_type", "saas"),
        "detail_level": _val("detail_level", "medium"),
        "output_format": _val("output_format", "phases_table"),
        "prompt_style": prompt_style or _infer_prompt_style(model),
        "reference_projects": getattr(request, "reference_projects", None) or [],
    }

    # project_metadata may be passed explicitly or carried on the request object.
    # Pre-populate all known template fields so StrictUndefined never raises.
    _pm_raw = (
        project_metadata
        if project_metadata is not None
        else getattr(request, "project_metadata", None)
    )
    _pm = _pm_raw or {}
    ctx["project_metadata"] = {
        "project_name": _pm.get("project_name")
        if isinstance(_pm, dict)
        else getattr(_pm, "project_name", None),
        "assumed_team_size": _pm.get("assumed_team_size")
        if isinstance(_pm, dict)
        else getattr(_pm, "assumed_team_size", None),
        "mentioned_technologies": _pm.get("mentioned_technologies")
        if isinstance(_pm, dict)
        else getattr(_pm, "mentioned_technologies", None),
        "agreed_scope": _pm.get("agreed_scope")
        if isinstance(_pm, dict)
        else getattr(_pm, "agreed_scope", None),
    }

    try:
        system = _env.get_template(f"estimation/{version}/system.j2").render(**ctx)
        user = _env.get_template(f"estimation/{version}/user.j2").render(**ctx)
    except TemplateNotFound:
        available = sorted(
            p.name for p in (_TEMPLATES_DIR / "estimation").iterdir() if p.is_dir()
        )
        raise ValueError(
            f"Unknown prompt version '{version}'. Available versions: {available}"
        )
    return system, user


def render_summarizer_prompt(
    accumulated_summary: str | None,
    messages: list[dict],
) -> str:
    """Render the summarizer prompt for the given conversation window.

    Args:
        accumulated_summary: The current rolling summary (may be empty).
        messages: List of message dicts with ``role`` and ``content`` keys.

    Returns:
        A single rendered string ready to send as a user message to the LLM.
    """
    template = _env.get_template("summarizer/v1.j2")
    return template.render(
        previous_summary=accumulated_summary or "",
        messages=messages,
    )
