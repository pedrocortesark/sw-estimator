"""Tests for the reference_projects feature (Bonus 2).

Verifies that:
- When reference_projects is provided, the user prompt contains the project names.
- Each project's hours appear in the prompt.
- Notes are rendered when present and omitted when absent.
- When reference_projects is None (default), the block is absent.
- The block uses the correct delimiters per prompt_style.
"""

import pytest

from src.prompts.loader import render_estimation_prompt
from src.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    ReferenceProject,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_REQ = dict(
    transcript="A logistics platform for warehouse inventory tracking and QR-based check-in.",
    project_type=ProjectType.INTERNAL_TOOL,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.PHASES_TABLE,
)

_REF_PROJECTS = [
    ReferenceProject(
        name="Acme CRM",
        description="Internal CRM for a 50-person sales team.",
        total_hours=320,
        notes="Heavy custom reporting added 40 h.",
    ),
    ReferenceProject(
        name="Beta Inventory",
        description="Simple stock tracker with barcode scanning.",
        total_hours=180,
        notes=None,
    ),
]


def _user(reference_projects=None, prompt_style="xml") -> str:
    req = EstimationRequest(**_BASE_REQ, reference_projects=reference_projects)
    _, user = render_estimation_prompt(req, prompt_style=prompt_style)
    return user


# ---------------------------------------------------------------------------
# Block presence / absence
# ---------------------------------------------------------------------------


def test_reference_projects_block_absent_when_none():
    """No reference_projects → the block must not appear in the user prompt."""
    user = _user(reference_projects=None)
    assert "reference_projects" not in user
    assert "Reference projects" not in user


def test_reference_projects_block_present_when_provided():
    """reference_projects provided → the block must appear."""
    user = _user(reference_projects=_REF_PROJECTS)
    assert "reference_projects" in user or "Reference projects" in user


# ---------------------------------------------------------------------------
# Content correctness
# ---------------------------------------------------------------------------


def test_reference_project_names_appear_in_user_prompt():
    """Both project names must be listed."""
    user = _user(reference_projects=_REF_PROJECTS)
    assert "Acme CRM" in user
    assert "Beta Inventory" in user


def test_reference_project_hours_appear_in_user_prompt():
    """Actual hours for each reference project must be present."""
    user = _user(reference_projects=_REF_PROJECTS)
    assert "320" in user
    assert "180" in user


def test_reference_project_notes_appear_when_set():
    """Notes must be included for projects that have them."""
    user = _user(reference_projects=_REF_PROJECTS)
    assert "Heavy custom reporting added 40 h." in user


def test_reference_project_notes_omitted_when_none():
    """Projects without notes must not emit a 'Note:' label."""
    user = _user(reference_projects=_REF_PROJECTS)
    # Split on the Beta Inventory line and verify no Note: follows immediately
    beta_pos = user.index("Beta Inventory")
    # The next project separator or end of block should come before any "Note:"
    snippet = user[beta_pos : beta_pos + 200]
    assert "Note:" not in snippet


# ---------------------------------------------------------------------------
# Style-aware delimiters
# ---------------------------------------------------------------------------


def test_reference_projects_xml_delimiters():
    """XML style must wrap the block in <reference_projects> tags."""
    user = _user(reference_projects=_REF_PROJECTS, prompt_style="xml")
    assert "<reference_projects>" in user
    assert "</reference_projects>" in user


def test_reference_projects_markdown_header():
    """Markdown style must use a ## header instead of XML tags."""
    user = _user(reference_projects=_REF_PROJECTS, prompt_style="markdown")
    assert "## Reference projects" in user
    assert "<reference_projects>" not in user


# ---------------------------------------------------------------------------
# Schema validation — ReferenceProject
# ---------------------------------------------------------------------------


def test_reference_project_notes_field_is_optional():
    """notes must be optional at the schema level."""
    rp = ReferenceProject(name="X", description="Test project.", total_hours=100)
    assert rp.notes is None


def test_estimation_request_reference_projects_defaults_to_none():
    """reference_projects must default to None (backward-compatible)."""
    req = EstimationRequest(**_BASE_REQ)
    assert req.reference_projects is None
