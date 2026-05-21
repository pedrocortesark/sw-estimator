"""Family 2 — Soft deterministic tests (statistical / consistency).

These tests verify *distributional* properties of the estimation system.
They do not check a single response; they run the system N times on the same
input and assert that the distribution of results has the expected shape.

What this family catches
------------------------
- Consistency drift: the same transcript produces wildly different hour
  estimates across runs due to temperature/sampling variance.
- Bimodal distributions: the system has two "personalities" for the same
  input (e.g. "20–30 h" half the time, "80–100 h" the other half), which
  suggests the prompt is sitting on a decision boundary.
- Confidence instability: the model alternates between high-confidence and
  low-confidence responses for the same well-scoped transcript.

Metrics used
------------
Coefficient of Variation (CV) = stdev(midpoints) / mean(midpoints)

CV is scale-invariant, so it works across projects of very different sizes
(a 30 h project and a 3000 h project both accept CV < 0.25).

A CV threshold of 0.25 means: "the standard deviation of the midpoint
estimates must not exceed 25 % of their mean."  This is deliberately
generous — LLMs are non-deterministic — but it catches the bimodal case
where CV would typically exceed 0.40–0.60.

Cost note
---------
Each parametric test makes N_RUNS live LLM calls.  With 8 golden cases and
N_RUNS = 5, that is 40 calls.  Run this suite selectively:

    pytest tests/test_soft_properties.py -v -m soft

Or only against one category to debug a specific failure:

    pytest tests/test_soft_properties.py -k "small_project" -m soft

CI recommendation: run on PRs to main, not on every commit.
"""

from __future__ import annotations

import asyncio
import os
import statistics

import pytest

from tests.golden_dataset import _GOLDENS

# Number of independent runs per golden case.
# 5 is the minimum that makes CV meaningful; raise to 10 for higher confidence.
N_RUNS: int = 5

# Maximum acceptable coefficient of variation for total_hours midpoints.
CV_THRESHOLD: float = 0.25

# Maximum acceptable standard deviation for confidence_pct across runs.
# A well-scoped transcript should produce stable confidence scores.
CONFIDENCE_STD_THRESHOLD: float = 15.0  # percentage points


# Seconds to wait between individual LLM calls inside N_RUNS loops.
# Keeps consumption well below Anthropic's 5 req/min + 4 000 output-token/min.
_INTER_RUN_SLEEP_S: float = 20.0


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def estimation_service():
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No LLM API key found — set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
    from src.services.estimation import EstimationService

    return EstimationService(cache=None, openai_client=None)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _hours_midpoint(response) -> float:
    """Return the total_hours value as the representative scalar for one run.

    EstimationResult carries a single total_hours value (not a range), so we
    use it directly as the midpoint for the CV calculation.
    """
    return response.estimation.total_hours


# ---------------------------------------------------------------------------
# Parametric consistency test — one test per golden case
# ---------------------------------------------------------------------------


@pytest.mark.soft
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    _GOLDENS,
    ids=[f"{g.additional_metadata['category']}_{i}" for i, g in enumerate(_GOLDENS)],
)
async def test_soft_consistency(golden, estimation_service):
    """Run N_RUNS estimates for the same golden input and check CV.

    The test is skipped for the 'ambiguous' and 'contradictory' categories
    because high variance is the *expected* behaviour for those inputs.
    """
    category: str = golden.additional_metadata["category"]

    if category in ("ambiguous", "contradictory"):
        pytest.skip(
            f"[{category}] Consistency check skipped: high variance is expected "
            "for ambiguous and contradictory inputs."
        )

    from src.schemas.estimation import EstimationRequest

    request = EstimationRequest(transcript=golden.input)

    responses = []
    for i in range(N_RUNS):
        if i > 0:
            await asyncio.sleep(_INTER_RUN_SLEEP_S)
        try:
            responses.append(await estimation_service.estimate(request))
        except Exception as exc:
            if "rate_limit" in str(exc).lower():
                pytest.skip(
                    f"[{category}] Anthropic rate limit hit on run {i + 1}/{N_RUNS} "
                    "— re-run with a higher-tier API key."
                )
            raise

    midpoints = [_hours_midpoint(r) for r in responses]
    mean = statistics.mean(midpoints)

    # Guard against degenerate case where mean is near zero
    assert mean > 0, (
        f"[{category}] Mean of midpoints is {mean:.1f} — all estimates returned 0 hours?"
    )

    # Coefficient of Variation
    if len(midpoints) > 1:
        cv = statistics.stdev(midpoints) / mean
        assert cv < CV_THRESHOLD, (
            f"[{category}] Inconsistent estimates across {N_RUNS} runs:\n"
            f"  midpoints = {[round(m, 1) for m in midpoints]}\n"
            f"  mean = {mean:.1f} h\n"
            f"  CV = {cv:.2f} (threshold {CV_THRESHOLD})\n"
            "Possible causes: prompt on a decision boundary, temperature too high, "
            "or the input is genuinely ambiguous for this model."
        )


# ---------------------------------------------------------------------------
# Confidence stability test — well-scoped inputs should not flip confidence
# ---------------------------------------------------------------------------


@pytest.mark.soft
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    [g for g in _GOLDENS if g.additional_metadata["category"] == "small_project"],
    ids=["small_project_confidence"],
)
async def test_soft_confidence_stability(golden, estimation_service):
    """For a well-scoped small project, confidence_pct should be stable across runs.

    A small, clearly-scoped project should consistently produce high-confidence
    responses.  If confidence flips between high and low across runs, the model
    is sitting on the out-of-scope decision boundary for this input.
    """
    from src.schemas.estimation import EstimationRequest

    request = EstimationRequest(transcript=golden.input)
    responses = []
    for i in range(N_RUNS):
        if i > 0:
            await asyncio.sleep(_INTER_RUN_SLEEP_S)
        try:
            responses.append(await estimation_service.estimate(request))
        except Exception as exc:
            if "rate_limit" in str(exc).lower():
                pytest.skip(
                    f"[small_project] Anthropic rate limit hit on run {i + 1}/{N_RUNS} "
                    "— re-run with a higher-tier API key."
                )
            raise

    confidences = [r.estimation.confidence_pct for r in responses]
    mean_conf = statistics.mean(confidences)
    std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0.0

    assert std_conf < CONFIDENCE_STD_THRESHOLD, (
        f"[small_project] confidence_pct is unstable across {N_RUNS} runs:\n"
        f"  values = {[round(c, 1) for c in confidences]}\n"
        f"  mean = {mean_conf:.1f} %\n"
        f"  stdev = {std_conf:.1f} pp (threshold {CONFIDENCE_STD_THRESHOLD} pp)\n"
        "Possible causes: prompt wording near the out-of-scope boundary, or the "
        "model is uncertain about the transcript's scope."
    )
