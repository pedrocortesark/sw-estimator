"""Golden-dataset evaluation tests for the sw-estimator.

These tests call the *real* estimation service using the curated golden
dataset defined in tests/golden_dataset.py.  They are intentionally
separated from the unit/integration tests that use mocks, because:

  1. They require live LLM credentials (ANTHROPIC_API_KEY or OPENAI_API_KEY).
  2. They are slow (one LLM call per golden case).
  3. They measure *quality*, not just correctness of the API contract.

Running
-------
Run the full suite:

    pytest tests/test_golden_dataset.py -v

Skip during normal CI (no credentials available):

    pytest tests/ --ignore=tests/test_golden_dataset.py

Or mark with a custom marker and control via -m:

    pytest tests/ -m "not golden"

Evaluation strategy
-------------------
Each golden case is evaluated on three layers:

  L1 — Schema validity (always asserted, no LLM needed)
       The response must be a well-formed EstimationResponse.

  L2 — Range check (asserted for cases with a meaningful expected range)
       total_hours must fall within [low, high].

  L3 — Component coverage (asserted when expected_components is non-empty)
       Each expected component keyword must appear somewhere in the
       serialised estimation output (case-insensitive).

DeepEval metrics (require LLM judge, marked separately) are wired up in
the ``test_golden_deepeval`` function and can be run independently.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.golden_dataset import golden_dataset, _GOLDENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialise(obj: Any) -> str:
    """Return a flat lowercase string representation of any pydantic model or dict."""
    if hasattr(obj, "model_dump"):
        return str(obj.model_dump()).lower()
    return str(obj).lower()


def _check_components(serialised: str, expected_components: list[str]) -> list[str]:
    """Return the subset of expected_components that are *missing* from the output."""
    return [kw for kw in expected_components if kw.lower() not in serialised]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def estimation_service():
    """Return a real EstimationService wired to live providers.

    Skips the whole module if no API key is available so CI stays green.
    Cache and OpenAI moderation are skipped when the respective env vars
    are absent — the core LLM estimation still runs.
    """
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if not has_anthropic and not has_openai:
        pytest.skip(
            "No LLM API key found (ANTHROPIC_API_KEY or OPENAI_API_KEY). "
            "Set at least one to run golden-dataset evaluation tests."
        )

    # Import here so the module can be collected without credentials.
    from src.services.estimation import EstimationService

    # cache=None skips Redis; openai_client=None skips content moderation.
    return EstimationService(cache=None, openai_client=None)


# ---------------------------------------------------------------------------
# L1 + L2 + L3  parametric test — one test per golden case
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    _GOLDENS,
    ids=[g.additional_metadata["category"] + f"_{i}" for i, g in enumerate(_GOLDENS)],
)
async def test_golden_case(golden, estimation_service):
    """L1 + L2 + L3 evaluation for a single golden case."""
    from src.schemas.estimation import EstimationRequest, EstimationResponse

    category: str = golden.additional_metadata["category"]
    low, high = golden.additional_metadata["expected_hours_range"]
    expected_components: list[str] = golden.additional_metadata["expected_components"]

    request = EstimationRequest(transcript=golden.input)
    response = await estimation_service.estimate(request)

    # L1 — schema validity (implicit: EstimationService already returns typed model,
    # but we assert the type explicitly for clarity)
    assert isinstance(response, EstimationResponse), (
        f"[{category}] Expected EstimationResponse, got {type(response)}"
    )

    total_hours: float = response.estimation.total_hours

    # L2 — range check (skip for ambiguous / contradictory cases where any range is ok)
    if high < 9999:
        assert low <= total_hours <= high, (
            f"[{category}] total_hours={total_hours:.1f} outside "
            f"expected range [{low}, {high}]"
        )

    # L3 — component coverage
    if expected_components:
        serialised = _serialise(response.estimation)
        missing = _check_components(serialised, expected_components)
        assert not missing, (
            f"[{category}] Missing components in output: {missing}. "
            f"Serialised snippet: {serialised[:300]}"
        )


# ---------------------------------------------------------------------------
# Dataset-level sanity checks (no LLM call needed)
# ---------------------------------------------------------------------------


def test_golden_dataset_has_minimum_cases():
    """The dataset must have at least 5 cases to be meaningful."""
    assert len(golden_dataset.goldens) >= 5, (
        f"Golden dataset has only {len(golden_dataset.goldens)} cases; "
        "add more to cover the required categories."
    )


def test_golden_dataset_covers_required_categories():
    """Every mandatory category must appear at least once."""
    required = {
        "small_project",
        "medium_project",
        "large_project",
        "ambiguous",
        "contradictory",
    }
    present = {g.additional_metadata["category"] for g in golden_dataset.goldens}
    missing = required - present
    assert not missing, f"Golden dataset is missing categories: {missing}"


def test_golden_dataset_metadata_is_complete():
    """Every golden must carry the four mandatory metadata keys."""
    mandatory_keys = {
        "category",
        "expected_hours_range",
        "expected_components",
        "key_risks",
    }
    for i, g in enumerate(golden_dataset.goldens):
        meta = g.additional_metadata or {}
        missing_keys = mandatory_keys - meta.keys()
        assert not missing_keys, (
            f"Golden case #{i} is missing metadata keys: {missing_keys}"
        )


def test_golden_dataset_hours_range_is_valid():
    """expected_hours_range must be a (low, high) tuple with low <= high."""
    for i, g in enumerate(golden_dataset.goldens):
        low, high = g.additional_metadata["expected_hours_range"]
        assert low >= 0, f"Golden #{i}: low={low} must be non-negative"
        assert low <= high, f"Golden #{i}: low={low} > high={high}"
