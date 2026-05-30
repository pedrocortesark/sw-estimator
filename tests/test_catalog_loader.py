"""Tests for src.ingest.catalog — models, loader, and audit report."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ingest.catalog import (
    CatalogDecision,
    CatalogSource,
    DataCatalog,
    QualityScore,
    Sensitivity,
    generate_audit_report,
    load_catalog,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATALOG_YAML = Path(__file__).parent.parent / "data" / "catalog" / "catalog.yaml"

_GOOD_QUALITY = dict(completeness=4, consistency=4, actuality=4, reliability=4)
_GOOD_SENSITIVITY = dict(has_pii=False, pii_flags=[], access_level="internal")


def _make_source(**overrides) -> dict:
    base = dict(
        name="my_source",
        description="A test source.",
        location="data/my_source",
        format="json",
        quality=_GOOD_QUALITY,
        sensitivity=_GOOD_SENSITIVITY,
        decision="include",
    )
    base.update(overrides)
    return base


def _make_catalog(*sources: dict, version: str = "1.0.0") -> DataCatalog:
    return DataCatalog.model_validate({"version": version, "sources": list(sources)})


# ---------------------------------------------------------------------------
# load_catalog — real YAML file
# ---------------------------------------------------------------------------


def test_load_catalog_real_yaml():
    catalog = load_catalog(CATALOG_YAML)
    assert catalog.version
    assert len(catalog.sources) >= 3


def test_load_catalog_real_yaml_has_included_source():
    catalog = load_catalog(CATALOG_YAML)
    assert catalog.included_sources(), "Expected at least one included source"


def test_load_catalog_real_yaml_has_excluded_source():
    catalog = load_catalog(CATALOG_YAML)
    excluded = [s for s in catalog.sources if s.decision is CatalogDecision.EXCLUDE]
    assert excluded, "Expected at least one excluded source"


# ---------------------------------------------------------------------------
# included_sources() and find()
# ---------------------------------------------------------------------------


def test_included_sources_returns_only_include_decision():
    catalog = _make_catalog(
        _make_source(name="a", decision="include"),
        _make_source(
            name="b",
            decision="review",
            decision_reason="Pending owner sign-off.",
        ),
        _make_source(
            name="c",
            decision="exclude",
            decision_reason="Obsolete data.",
        ),
    )
    included = catalog.included_sources()
    assert [s.name for s in included] == ["a"]


def test_find_returns_matching_source():
    catalog = _make_catalog(
        _make_source(name="alpha"),
        _make_source(name="beta"),
    )
    assert catalog.find("beta") is not None
    assert catalog.find("beta").name == "beta"  # type: ignore[union-attr]


def test_find_returns_none_for_unknown_name():
    catalog = _make_catalog(_make_source(name="alpha"))
    assert catalog.find("does_not_exist") is None


# ---------------------------------------------------------------------------
# Validation — decision_reason required when not include
# ---------------------------------------------------------------------------


def test_review_without_reason_raises():
    with pytest.raises(ValidationError, match="decision_reason"):
        _make_catalog(_make_source(name="x", decision="review"))


def test_exclude_without_reason_raises():
    with pytest.raises(ValidationError, match="decision_reason"):
        _make_catalog(_make_source(name="x", decision="exclude"))


def test_review_with_reason_is_valid():
    catalog = _make_catalog(
        _make_source(name="x", decision="review", decision_reason="Needs audit.")
    )
    assert catalog.sources[0].decision is CatalogDecision.REVIEW


# ---------------------------------------------------------------------------
# Validation — duplicate names
# ---------------------------------------------------------------------------


def test_duplicate_source_name_raises():
    with pytest.raises(ValidationError, match="Duplicate"):
        _make_catalog(
            _make_source(name="dup"),
            _make_source(name="dup"),
        )


# ---------------------------------------------------------------------------
# Validation — snake_case name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_name", ["MySource", "my-source", "my source", "123source"])
def test_non_snake_case_name_raises(bad_name: str):
    with pytest.raises(ValidationError, match="snake_case"):
        _make_catalog(_make_source(name=bad_name))


@pytest.mark.parametrize("good_name", ["my_source", "source_2024", "presupuestos_json"])
def test_valid_snake_case_name_accepted(good_name: str):
    catalog = _make_catalog(_make_source(name=good_name))
    assert catalog.sources[0].name == good_name


# ---------------------------------------------------------------------------
# QualityScore.is_rag_ready
# ---------------------------------------------------------------------------


def test_is_rag_ready_all_dimensions_at_threshold():
    q = QualityScore(completeness=3, consistency=3, actuality=3, reliability=3)
    assert q.is_rag_ready is True


def test_is_rag_ready_all_dimensions_above_threshold():
    q = QualityScore(completeness=5, consistency=5, actuality=5, reliability=5)
    assert q.is_rag_ready is True


def test_is_rag_ready_one_dimension_at_2_is_false():
    q = QualityScore(completeness=4, consistency=2, actuality=4, reliability=4)
    assert q.is_rag_ready is False


def test_is_rag_ready_actuality_1_is_false():
    """The rate_card_xlsx scenario from data/catalog/catalog.yaml."""
    q = QualityScore(completeness=4, consistency=5, actuality=1, reliability=4)
    assert q.is_rag_ready is False


def test_is_rag_ready_not_averaged():
    """3+3+1+5 = 12 → avg 3.0, but actuality=1 must block individually."""
    q = QualityScore(completeness=3, consistency=3, actuality=1, reliability=5)
    assert q.is_rag_ready is False


# ---------------------------------------------------------------------------
# generate_audit_report
# ---------------------------------------------------------------------------


def test_generate_audit_report_contains_version():
    catalog = load_catalog(CATALOG_YAML)
    report = generate_audit_report(catalog)
    assert catalog.version in report


def test_generate_audit_report_summary_counters():
    catalog = _make_catalog(
        _make_source(name="a", decision="include"),
        _make_source(name="b", decision="review", decision_reason="Pending."),
        _make_source(name="c", decision="exclude", decision_reason="Obsolete."),
    )
    report = generate_audit_report(catalog)
    assert "Total" in report
    assert "Included" in report
    assert "Review" in report
    assert "Excluded" in report


def test_generate_audit_report_rag_ready_flag_present():
    catalog = _make_catalog(_make_source(name="good_source", decision="include"))
    report = generate_audit_report(catalog)
    assert "rag_ready" in report


def test_generate_audit_report_not_rag_ready_flag():
    catalog = _make_catalog(
        _make_source(
            name="weak_source",
            decision="include",
            quality=dict(completeness=4, consistency=4, actuality=1, reliability=4),
        )
    )
    report = generate_audit_report(catalog)
    assert "not_rag_ready" in report


def test_generate_audit_report_decision_reason_for_review():
    reason = "Needs owner sign-off before indexing."
    catalog = _make_catalog(
        _make_source(name="r", decision="review", decision_reason=reason)
    )
    report = generate_audit_report(catalog)
    assert reason in report


def test_generate_audit_report_decision_reason_for_excluded():
    reason = "Actuality score is 1, data is from 2024."
    catalog = _make_catalog(
        _make_source(name="e", decision="exclude", decision_reason=reason)
    )
    report = generate_audit_report(catalog)
    assert reason in report
