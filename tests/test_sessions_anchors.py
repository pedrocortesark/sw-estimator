"""Unit tests for Session.update_anchors heuristic anchor detection."""

from __future__ import annotations

import pytest

from src.services.sessions import ProjectMetadata, Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(
    project_name: str | None = None,
    team_size: int | None = None,
    techs: list[str] | None = None,
) -> ProjectMetadata:
    return ProjectMetadata(
        project_name=project_name,
        assumed_team_size=team_size,
        mentioned_technologies=techs or [],
    )


def _session() -> Session:
    """Return a fresh session with max_turns=6 (default)."""
    return Session(session_id="test-session")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_anchors_project_name_stable() -> None:
    """Matching project_name across two turns creates a project_name anchor."""
    s = _session()
    prev = _meta(project_name="InvoiceApp")
    new = _meta(project_name="InvoiceApp")

    s.update_anchors(prev, new)

    assert "project_name:InvoiceApp" in s.anchors


def test_anchors_project_name_none_ignored() -> None:
    """None project_name must NOT generate an anchor."""
    s = _session()
    s.update_anchors(_meta(project_name=None), _meta(project_name=None))

    assert not any(a.startswith("project_name:") for a in s.anchors)


def test_anchors_project_name_changed_not_anchored() -> None:
    """Different project names between turns must NOT create an anchor."""
    s = _session()
    s.update_anchors(_meta(project_name="AppA"), _meta(project_name="AppB"))

    assert not any(a.startswith("project_name:") for a in s.anchors)


def test_anchors_team_size_stable() -> None:
    """Matching team_size across turns creates a team_size anchor."""
    s = _session()
    s.update_anchors(_meta(team_size=4), _meta(team_size=4))

    assert "team_size:4" in s.anchors


def test_anchors_team_size_none_ignored() -> None:
    """None team_size must NOT generate an anchor."""
    s = _session()
    s.update_anchors(_meta(team_size=None), _meta(team_size=None))

    assert not any(a.startswith("team_size:") for a in s.anchors)


def test_anchors_tech_in_common() -> None:
    """Technologies present in both turns are anchored individually."""
    s = _session()
    s.update_anchors(
        _meta(techs=["React", "PostgreSQL", "Docker"]),
        _meta(techs=["React", "PostgreSQL", "Stripe"]),
    )

    assert "tech:react" in s.anchors
    assert "tech:postgresql" in s.anchors
    assert "tech:stripe" not in s.anchors  # only in new, not in previous
    assert "tech:docker" not in s.anchors  # only in previous, not in new


def test_anchors_accumulate_across_turns() -> None:
    """Anchors from multiple calls accumulate in order without duplicates."""
    s = _session()

    # Turn 1 → 2: project_name stabilises
    s.update_anchors(_meta(project_name="MyApp"), _meta(project_name="MyApp"))
    assert "project_name:MyApp" in s.anchors
    count_after_turn1 = len(s.anchors)

    # Turn 2 → 3: team_size stabilises, project_name still stable
    s.update_anchors(
        _meta(project_name="MyApp", team_size=3),
        _meta(project_name="MyApp", team_size=3),
    )
    # project_name anchor must not be duplicated
    assert s.anchors.count("project_name:MyApp") == 1
    assert "team_size:3" in s.anchors
    assert len(s.anchors) == count_after_turn1 + 1  # only team_size added


def test_anchors_no_duplicates() -> None:
    """Calling update_anchors with the same stable fact three times never
    adds the anchor more than once."""
    s = _session()
    meta = _meta(project_name="Stable", team_size=5, techs=["Python"])

    s.update_anchors(meta, meta)
    s.update_anchors(meta, meta)
    s.update_anchors(meta, meta)

    assert s.anchors.count("project_name:Stable") == 1
    assert s.anchors.count("team_size:5") == 1
    assert s.anchors.count("tech:python") == 1


def test_anchors_empty_initially() -> None:
    """A freshly created session has no anchors."""
    s = _session()
    assert s.anchors == []
