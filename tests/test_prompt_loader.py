"""Tests for src/prompts/loader.py — Jinja2 prompt rendering."""

import pytest
from jinja2 import UndefinedError

from src.prompts.loader import _infer_prompt_style, render_estimation_prompt
from src.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    *,
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.PHASES_TABLE,
    description: str = (
        "The client wants a SaaS platform where teams can manage projects "
        "with kanban boards, time tracking, and Slack notifications."
    ),
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )


# ---------------------------------------------------------------------------
# Return-type and basic structure
# ---------------------------------------------------------------------------

def test_returns_tuple_of_two_strings():
    system, user = render_estimation_prompt(_make_request())
    assert isinstance(system, str)
    assert isinstance(user, str)


def test_both_strings_are_non_empty():
    system, user = render_estimation_prompt(_make_request())
    assert len(system) > 0
    assert len(user) > 0


# ---------------------------------------------------------------------------
# system.j2 — static content always present
# ---------------------------------------------------------------------------

def test_system_contains_role_description():
    system, _ = render_estimation_prompt(_make_request())
    assert "senior software estimation consultant" in system


def test_system_contains_rules_section():
    system, _ = render_estimation_prompt(_make_request())
    assert "## Rules" in system


def test_system_contains_examples():
    """examples.j2 must be included via {% include %} in system.j2."""
    system, _ = render_estimation_prompt(_make_request())
    # All three examples should appear
    assert "Example 1" in system
    assert "Example 2" in system
    assert "Example 3" in system


# ---------------------------------------------------------------------------
# system.j2 — output_format conditional blocks
# ---------------------------------------------------------------------------

def test_system_phases_table_instructions_when_output_format_phases_table():
    system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.PHASES_TABLE)
    )
    assert "breakdown by project phase" in system


def test_system_line_items_instructions_when_output_format_line_items():
    system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.LINE_ITEMS)
    )
    assert "flat, exhaustive list" in system


def test_system_narrative_instructions_when_output_format_narrative():
    system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.NARRATIVE)
    )
    assert "prose" in system


def test_system_does_not_contain_other_output_format_instructions():
    """phases_table block must not leak into a narrative-format render."""
    system, _ = render_estimation_prompt(
        _make_request(output_format=OutputFormat.NARRATIVE)
    )
    assert "breakdown by project phase" not in system
    assert "flat, exhaustive list" not in system


# ---------------------------------------------------------------------------
# system.j2 — detail_level conditional blocks
# ---------------------------------------------------------------------------

def test_system_summary_instructions_when_detail_level_summary():
    system, _ = render_estimation_prompt(
        _make_request(detail_level=DetailLevel.SUMMARY)
    )
    assert "high-level totals only" in system


def test_system_medium_instructions_when_detail_level_medium():
    system, _ = render_estimation_prompt(
        _make_request(detail_level=DetailLevel.MEDIUM)
    )
    assert "3–8 tasks per module" in system


def test_system_detailed_instructions_when_detail_level_detailed():
    system, _ = render_estimation_prompt(
        _make_request(detail_level=DetailLevel.DETAILED)
    )
    assert "most granular breakdown" in system


def test_system_does_not_contain_other_detail_level_instructions():
    """summary block must not leak into a detailed render."""
    system, _ = render_estimation_prompt(
        _make_request(detail_level=DetailLevel.DETAILED)
    )
    assert "high-level totals only" not in system
    assert "3–8 tasks per module" not in system


# ---------------------------------------------------------------------------
# user.j2 — description and metadata interpolation
# ---------------------------------------------------------------------------

def test_user_contains_description():
    description = (
        "Build a mobile app for iOS and Android that lets users track their "
        "daily water intake with reminders and a weekly progress chart."
    )
    _, user = render_estimation_prompt(_make_request(description=description))
    assert description in user


def test_user_contains_project_type_value():
    _, user = render_estimation_prompt(
        _make_request(project_type=ProjectType.MOBILE_APP)
    )
    assert "mobile_app" in user


def test_user_contains_detail_level_value():
    _, user = render_estimation_prompt(
        _make_request(detail_level=DetailLevel.DETAILED)
    )
    assert "detailed" in user


def test_user_contains_output_format_value():
    _, user = render_estimation_prompt(
        _make_request(output_format=OutputFormat.LINE_ITEMS)
    )
    assert "line_items" in user


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

def test_default_version_is_v1():
    """Calling without version argument must not raise — v1 must exist."""
    system, user = render_estimation_prompt(_make_request())
    assert system
    assert user


def test_unknown_version_raises():
    with pytest.raises(Exception):
        render_estimation_prompt(_make_request(), version="v99")


# ---------------------------------------------------------------------------
# StrictUndefined — missing template variable must raise
# ---------------------------------------------------------------------------

def test_strict_undefined_raises_on_missing_variable(tmp_path):
    """A template that references an undefined variable must raise UndefinedError,
    not silently render an empty string. This validates that StrictUndefined
    is configured on the Environment."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    template_dir = tmp_path
    (template_dir / "bad.j2").write_text("Hello {{ undefined_var }}!")

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
    )
    with pytest.raises(UndefinedError):
        env.get_template("bad.j2").render()


# ---------------------------------------------------------------------------
# Cross-product smoke test — all enum combinations render without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("project_type", list(ProjectType))
@pytest.mark.parametrize("detail_level", list(DetailLevel))
@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_all_enum_combinations_render(project_type, detail_level, output_format):
    system, user = render_estimation_prompt(
        _make_request(
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    )
    assert system
    assert user


# ---------------------------------------------------------------------------
# _infer_prompt_style — unit tests for the inference helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", [
    "claude-3-5-haiku-20241022",
    "anthropic/claude-3-5-sonnet-20241022",
    "claude-opus-4",
    "CLAUDE-3-HAIKU",          # case-insensitive
])
def test_infer_style_returns_xml_for_claude_models(model):
    assert _infer_prompt_style(model) == "xml"


@pytest.mark.parametrize("model", [
    "gpt-4o",
    "openai/gpt-4o-mini",
    "gemini/gemini-2.0-flash",
    "mistral/mistral-large-latest",
    None,                       # no model → default
])
def test_infer_style_returns_markdown_for_non_claude_models(model):
    assert _infer_prompt_style(model) == "markdown"


# ---------------------------------------------------------------------------
# prompt_style — explicit override
# ---------------------------------------------------------------------------

def test_explicit_xml_style_overrides_non_claude_model():
    """Even a GPT model must produce XML when forced via prompt_style="xml"."""
    system, _ = render_estimation_prompt(
        _make_request(), model="gpt-4o", prompt_style="xml"
    )
    assert "<task>" in system
    assert "<rules>" in system
    # markdown headers must NOT appear
    assert "## Your task" not in system


def test_explicit_markdown_style_overrides_claude_model():
    """Even a Claude model must produce markdown when forced via prompt_style="markdown"."""
    system, _ = render_estimation_prompt(
        _make_request(), model="claude-3-5-haiku-20241022", prompt_style="markdown"
    )
    assert "## Your task" in system
    assert "## Rules" in system
    # XML tags must NOT appear
    assert "<task>" not in system


# ---------------------------------------------------------------------------
# prompt_style — auto-inference via model argument
# ---------------------------------------------------------------------------

def test_claude_model_produces_xml_delimiters():
    system, _ = render_estimation_prompt(
        _make_request(), model="anthropic/claude-3-5-haiku-20241022"
    )
    assert "<task>" in system
    assert "</task>" in system
    assert "<rules>" in system
    assert "</rules>" in system


def test_openai_model_produces_markdown_delimiters():
    system, _ = render_estimation_prompt(
        _make_request(), model="openai/gpt-4o"
    )
    assert "## Your task" in system
    assert "## Rules" in system
    assert "<task>" not in system


def test_no_model_defaults_to_markdown():
    system, _ = render_estimation_prompt(_make_request())
    assert "## Your task" in system
    assert "<task>" not in system


def test_xml_style_all_section_tags_present():
    """All five section tags must appear when using XML style."""
    system, _ = render_estimation_prompt(
        _make_request(), prompt_style="xml"
    )
    for tag in ["task", "output_format", "detail_level", "examples", "rules"]:
        assert f"<{tag}>" in system, f"Opening tag <{tag}> not found"
        assert f"</{tag}>" in system, f"Closing tag </{tag}> not found"


def test_markdown_style_no_xml_tags():
    """No XML tags should appear when using markdown style."""
    system, _ = render_estimation_prompt(
        _make_request(), prompt_style="markdown"
    )
    assert "<task>" not in system
    assert "</rules>" not in system
