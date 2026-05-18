"""Tests for the estimation prompt template v1.

These tests run in milliseconds — no API calls, no LLM, no network.
They validate the rendered text of system.j2 and user.j2 directly.
"""

import pytest

from src.prompts.loader import render_estimation_prompt
from src.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)

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
        transcript=DESCRIPTION,
        project_type=ProjectType.INTERNAL_TOOL,
        detail_level=detail_level,
        output_format=output_format,
    )
    return render_estimation_prompt(request, prompt_style=prompt_style, **kwargs)


# ---------------------------------------------------------------------------
# Test 1 — description appears inside the project_description block
# ---------------------------------------------------------------------------


def test_user_description_inside_xml_project_description_block():
    """In v1 the description is wrapped in <transcript> tags."""
    _, user = _req(prompt_style="xml")
    assert "<transcript>" in user
    assert DESCRIPTION in user
    assert "</transcript>" in user
    # The description must appear between the opening and closing tags
    start = user.index("<transcript>")
    end = user.index("</transcript>")
    assert DESCRIPTION in user[start:end]


def test_user_description_inside_markdown_project_description_block():
    """In v1 the description appears after a transcript label in user prompt."""
    _, user = _req(prompt_style="markdown")
    assert "<transcript>" in user
    assert DESCRIPTION in user
    # Description must come after the transcript tag
    header_pos = user.index("<transcript>")
    desc_pos = user.index(DESCRIPTION)
    assert desc_pos > header_pos


def test_user_description_not_leaked_outside_block_in_xml():
    """The description text must not appear before the opening tag."""
    _, user = _req(prompt_style="xml")
    tag_pos = user.index("<transcript>")
    assert DESCRIPTION not in user[:tag_pos]


# ---------------------------------------------------------------------------
# Test 2 — output_format conditional: phases_table vs narrative
# ---------------------------------------------------------------------------


def test_system_phases_table_keyword_present_when_phases_table():
    """phases_table format must inject instructions about ordered phases."""
    system, _ = _req(output_format=OutputFormat.PHASES_TABLE)
    assert "Structure the estimate as ordered phases" in system


def test_system_phases_table_keyword_absent_when_narrative():
    """phases_table keyword must not appear when output_format is narrative."""
    system, _ = _req(output_format=OutputFormat.NARRATIVE)
    assert "Structure the estimate as ordered phases" not in system


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
    """detail_level=detailed must instruct the model to list assumptions per phase."""
    system, _ = _req(detail_level=DetailLevel.DETAILED)
    assert "assumptions" in system
    assert "phase" in system


def test_system_summary_does_not_include_assumptions_instruction():
    """detail_level=summary must NOT include the per-phase assumptions instruction."""
    system, _ = _req(detail_level=DetailLevel.SUMMARY)
    assert "List assumptions per phase" not in system


def test_system_medium_does_not_include_assumptions_instruction():
    """detail_level=medium must NOT include the per-phase assumptions instruction."""
    system, _ = _req(detail_level=DetailLevel.MEDIUM)
    assert "List assumptions per phase" not in system


def test_system_detailed_does_not_include_summary_instruction():
    """detail_level=detailed must not leak the summary-level instruction."""
    system, _ = _req(detail_level=DetailLevel.DETAILED)
    assert "high-level totals only" not in system
