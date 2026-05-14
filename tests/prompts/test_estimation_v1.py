"""Tests for the estimation prompt template v1.

These tests run in milliseconds — no API calls, no LLM, no network.
They validate the rendered text of system.j2 and user.j2 directly.
"""

import pytest

from src.prompts.loader import render_estimation_prompt
from src.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

DESCRIPTION = (
    "The client needs a logistics platform where warehouse operators can track "
    "inbound shipments, assign storage locations via QR scan, and generate "
    "daily inventory reports exported to Excel."
)


def _req(
    *,
    output_format: OutputFormat = OutputFormat.PHASES_TABLE,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    prompt_style: str | None = None,
    **kwargs,
) -> tuple[str, str]:
    request = EstimationRequest(
        description=DESCRIPTION,
        project_type=ProjectType.INTERNAL_TOOL,
        detail_level=detail_level,
        output_format=output_format,
    )
    return render_estimation_prompt(request, prompt_style=prompt_style, **kwargs)


# ---------------------------------------------------------------------------
# Test 1 — description appears inside the project_description block
# ---------------------------------------------------------------------------

def test_user_description_inside_xml_project_description_block():
    """With XML style the description must be wrapped in <project_description>."""
    _, user = _req(prompt_style="xml")
    assert "<project_description>" in user
    assert DESCRIPTION in user
    assert "</project_description>" in user
    # The description must appear between the opening and closing tags
    start = user.index("<project_description>")
    end = user.index("</project_description>")
    assert DESCRIPTION in user[start:end]


def test_user_description_inside_markdown_project_description_block():
    """With Markdown style the description must appear under ## Project description."""
    _, user = _req(prompt_style="markdown")
    assert "## Project description" in user
    assert DESCRIPTION in user
    # Description must come after the header
    header_pos = user.index("## Project description")
    desc_pos = user.index(DESCRIPTION)
    assert desc_pos > header_pos


def test_user_description_not_leaked_outside_block_in_xml():
    """The description text must not appear before the opening tag."""
    _, user = _req(prompt_style="xml")
    tag_pos = user.index("<project_description>")
    assert DESCRIPTION not in user[:tag_pos]


# ---------------------------------------------------------------------------
# Test 2 — output_format conditional: phases_table vs narrative
# ---------------------------------------------------------------------------

def test_system_phases_table_keyword_present_when_phases_table():
    """phases_table format must inject instructions about project phase breakdown."""
    system, _ = _req(output_format=OutputFormat.PHASES_TABLE)
    assert "breakdown by project phase" in system


def test_system_phases_table_keyword_absent_when_narrative():
    """phases_table keyword must not appear when output_format is narrative."""
    system, _ = _req(output_format=OutputFormat.NARRATIVE)
    assert "breakdown by project phase" not in system


def test_system_narrative_keyword_present_when_narrative():
    """narrative format must inject prose-writing instructions."""
    system, _ = _req(output_format=OutputFormat.NARRATIVE)
    assert "prose" in system


def test_system_narrative_keyword_absent_when_phases_table():
    """Prose instruction must not appear when output_format is phases_table."""
    system, _ = _req(output_format=OutputFormat.PHASES_TABLE)
    assert "prose" not in system


# ---------------------------------------------------------------------------
# Test 3 — detail_level conditional: detailed assumptions vs summary
# ---------------------------------------------------------------------------

def test_system_detailed_includes_assumptions_instruction():
    """detail_level=detailed must instruct the model to list technical assumptions."""
    system, _ = _req(detail_level=DetailLevel.DETAILED)
    assert "key technical assumptions" in system


def test_system_summary_does_not_include_assumptions_instruction():
    """detail_level=summary must NOT include the per-module assumptions instruction."""
    system, _ = _req(detail_level=DetailLevel.SUMMARY)
    assert "key technical assumptions" not in system


def test_system_medium_does_not_include_assumptions_instruction():
    """detail_level=medium must NOT include the per-module assumptions instruction."""
    system, _ = _req(detail_level=DetailLevel.MEDIUM)
    assert "key technical assumptions" not in system


def test_system_detailed_does_not_include_summary_instruction():
    """detail_level=detailed must not leak the summary-level instruction."""
    system, _ = _req(detail_level=DetailLevel.DETAILED)
    assert "high-level totals only" not in system
