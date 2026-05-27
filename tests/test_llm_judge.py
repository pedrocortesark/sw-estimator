"""Family 3 — LLM-as-judge tests (subjective quality evaluation).

These tests verify properties that cannot be expressed as structural or
numerical assertions.  A second LLM call — the *judge* — evaluates the
quality of the estimation output against a human-written criterion.

DeepEval's GEval metric orchestrates the judge prompt, normalises the result
to [0, 1], and integrates with pytest via ``assert_test``.

Criteria covered
----------------
C1  JustificationCoherence
    The risks and technical concerns in the estimation are relevant to the
    project scope described in the transcript.  A hallucinated risk (e.g.
    "PCI-DSS compliance" for a landing page) fails this criterion.

C2  ScopeAcknowledgment (ambiguous inputs only)
    For transcripts that lack critical details, the estimation must
    explicitly surface the missing information and flag the uncertainty
    rather than inventing scope.

C3  ContradictionDetection (contradictory inputs only)
    For transcripts with mutually-exclusive constraints, the estimation
    must call out the contradiction rather than producing a fictional plan.

C4  ComponentRelevance
    The phase and task names are coherent with the described project.
    A backend-heavy project should not have most effort in "UX Design".

Threshold guidance
------------------
The default threshold (0.6) is conservative for early runs.  Calibrate it
against a manual review: look at the GEval scores for 5-10 known-good and
known-bad outputs in your domain, and pick the value that correctly separates
them.  A threshold of 0.7–0.8 is typical once calibrated.

Cost note
---------
Each GEval metric fires one additional LLM call (the judge).  With 4 criteria
over the golden dataset, each run costs roughly 4 × (golden count) extra calls.
Reserve this suite for pre-merge CI and scheduled nightly runs.

    pytest tests/test_llm_judge.py -v -m judge
"""

from __future__ import annotations

import json
import os

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.golden_dataset import _GOLDENS

# Default GEval threshold.  Increase toward 0.8 after manual calibration.
_THRESHOLD = 0.6


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
# Helpers
# ---------------------------------------------------------------------------


def _actual_output(response) -> str:
    """Serialise the estimation to a compact JSON string for the judge."""
    return json.dumps(response.estimation.model_dump(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# C1 — JustificationCoherence
# Applied to all well-defined golden cases (not ambiguous/contradictory)
# ---------------------------------------------------------------------------


@pytest.mark.judge
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    [
        g
        for g in _GOLDENS
        if g.additional_metadata["category"] not in ("ambiguous", "contradictory")
    ],
    ids=[
        f"coherence_{g.additional_metadata['category']}_{i}"
        for i, g in enumerate(_GOLDENS)
        if g.additional_metadata["category"] not in ("ambiguous", "contradictory")
    ],
)
async def test_justification_coherence(golden, estimation_service):
    """C1: Technical risks in the output are relevant to the described scope."""
    from src.schemas.estimation import EstimationRequest

    category = golden.additional_metadata["category"]
    try:
        response = await estimation_service.estimate(
            EstimationRequest(transcript=golden.input)
        )
    except Exception as exc:
        if "rate_limit" in str(exc).lower():
            pytest.skip(f"[{category}] Anthropic rate limit — re-run later.")
        raise

    coherence = GEval(
        name="JustificationCoherence",
        criteria=(
            "Determine whether the phases, tasks, and team composition in the "
            "actual output are coherent with the project scope described in the "
            "input.  A coherent estimation contains components that are clearly "
            "necessary for the described project and does not include unrelated "
            "components.  Technical risks or components that would never apply to "
            "the described scope count as hallucinations and lower the score."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=_THRESHOLD,
    )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=_actual_output(response),
    )

    assert_test(test_case, [coherence])


# ---------------------------------------------------------------------------
# C2 — ScopeAcknowledgment (ambiguous inputs)
# ---------------------------------------------------------------------------


@pytest.mark.judge
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    [g for g in _GOLDENS if g.additional_metadata["category"] == "ambiguous"],
    ids=["scope_acknowledgment_ambiguous"],
)
async def test_scope_acknowledgment(golden, estimation_service):
    """C2: Ambiguous inputs must surface missing information explicitly."""
    from src.schemas.estimation import EstimationRequest

    try:
        response = await estimation_service.estimate(
            EstimationRequest(transcript=golden.input)
        )
    except Exception as exc:
        if "rate_limit" in str(exc).lower():
            pytest.skip("[ambiguous] Anthropic rate limit — re-run later.")
        raise

    scope_ack = GEval(
        name="ScopeAcknowledgment",
        criteria=(
            "The input is a vague project description that omits critical details "
            "(payment flow, scale targets, notification channels, etc.).  "
            "Evaluate whether the actual output explicitly acknowledges the missing "
            "information, lists the assumptions it was forced to make, and flags "
            "that the estimate has high uncertainty due to underspecified requirements.  "
            "An output that invents a fully-detailed plan without acknowledging the "
            "gaps scores low.  An output that lists assumptions and warns about "
            "the wide range of uncertainty scores high."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=_THRESHOLD,
    )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=_actual_output(response),
    )

    assert_test(test_case, [scope_ack])


# ---------------------------------------------------------------------------
# C3 — ContradictionDetection (contradictory inputs)
# ---------------------------------------------------------------------------


@pytest.mark.judge
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    [g for g in _GOLDENS if g.additional_metadata["category"] == "contradictory"],
    ids=["contradiction_detection_contradictory"],
)
async def test_contradiction_detection(golden, estimation_service):
    """C3: Contradictory constraints must be explicitly called out."""
    from src.schemas.estimation import EstimationRequest

    try:
        response = await estimation_service.estimate(
            EstimationRequest(transcript=golden.input)
        )
    except Exception as exc:
        if "rate_limit" in str(exc).lower():
            pytest.skip("[contradictory] Anthropic rate limit — re-run later.")
        raise

    contradiction = GEval(
        name="ContradictionDetection",
        criteria=(
            "The input contains mutually exclusive constraints: a full ERP system "
            "in two weeks, with a budget of $5 000, using one part-time junior "
            "developer, with a microservices and event-sourcing stack.  "
            "Evaluate whether the actual output explicitly identifies that these "
            "constraints cannot coexist and explains why.  An output that produces "
            "a straight-faced delivery plan without flagging the impossibility "
            "scores low.  An output that clearly calls out the contradiction and "
            "explains the mismatch between scope, timeline, budget, and team scores high."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=_THRESHOLD,
    )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=_actual_output(response),
    )

    assert_test(test_case, [contradiction])


# ---------------------------------------------------------------------------
# C4 — ComponentRelevance (medium and large projects)
# ---------------------------------------------------------------------------


@pytest.mark.judge
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "golden",
    [
        g
        for g in _GOLDENS
        if g.additional_metadata["category"] in ("medium_project", "large_project")
    ],
    ids=[
        f"component_relevance_{g.additional_metadata['category']}_{i}"
        for i, g in enumerate(_GOLDENS)
        if g.additional_metadata["category"] in ("medium_project", "large_project")
    ],
)
async def test_component_relevance(golden, estimation_service):
    """C4: Phase and task names must match the dominant technical concerns."""
    from src.schemas.estimation import EstimationRequest

    category = golden.additional_metadata["category"]
    expected_components: list[str] = golden.additional_metadata["expected_components"]

    # Build a human-readable hint for the judge prompt
    components_hint = (
        f"  Expected components: {', '.join(expected_components)}.\n"
        if expected_components
        else ""
    )

    try:
        response = await estimation_service.estimate(
            EstimationRequest(transcript=golden.input)
        )
    except Exception as exc:
        if "rate_limit" in str(exc).lower():
            pytest.skip(f"[{category}] Anthropic rate limit — re-run later.")
        raise

    relevance = GEval(
        name="ComponentRelevance",
        criteria=(
            "Evaluate whether the phases and tasks in the actual output reflect the "
            "dominant technical concerns of the project described in the input.  "
            f"{components_hint}"
            "The estimation should allocate meaningful effort to the components "
            "that the project genuinely requires, and should not assign the majority "
            "of effort to components that are peripheral or unmentioned in the input.  "
            "Missing a clearly required component (e.g. no 'payment' phase for a "
            "multi-PSP e-commerce project) counts as a relevance failure."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=_THRESHOLD,
    )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=_actual_output(response),
    )

    assert_test(test_case, [relevance])
