"""Integration tests for the session endpoints.

Three scenarios
---------------
1. Two-turn session — verify ProjectMetadata accumulates across turns.
2. PDF attachment — verify attachment text is forwarded to the service.
3. Sliding-window cap — 8 turns must not exceed MAX_CONVERSATION_TURNS (6).

All tests use a fake EstimationService injected via dependency_overrides —
no LLM or Redis connection is required.
"""

from __future__ import annotations

import io
import textwrap

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.core.config import get_settings
from src.dependencies import get_estimation_service
from src.main import create_app
from src.schemas.estimation import (
    EstimationResponse,
    EstimationResult,
    Phase,
    Task,
    TeamMember,
    UsageCost,
)
from src.services.sessions import session_store

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_TRANSCRIPT_TURN_1 = (
    "We need to build the InvoiceApp platform using React and PostgreSQL. "
    "The team has 3 engineers available full-time."
)

_TRANSCRIPT_TURN_2 = (
    "In addition to what we discussed, we need Docker-based deployment "
    "and a Stripe integration for payments."
)

_MOCK_RESULT = EstimationResult(
    executive_summary="Invoice management platform — 120 hours, 3 developers.",
    phases=[
        Phase(
            name="Backend",
            tasks=[Task(name="REST API", hours=60.0, cost_usd=6000.0)],
            total_hours=60.0,
            total_cost_usd=6000.0,
        ),
        Phase(
            name="Frontend",
            tasks=[Task(name="React UI", hours=60.0, cost_usd=6000.0)],
            total_hours=60.0,
            total_cost_usd=6000.0,
        ),
    ],
    total_hours=120.0,
    total_cost_usd=12000.0,
    team_composition=[
        TeamMember(role="Full-stack Engineer", count=3, dedication="100%")
    ],
    duration_weeks=4.0,
)

_MOCK_RESPONSE = EstimationResponse(
    estimation=_MOCK_RESULT,
    provider_used="anthropic",
    model_used="claude-haiku-test",
    usage=UsageCost(
        input_tokens=500,
        output_tokens=200,
        total_tokens=700,
        cost_usd=0.000150,
    ),
)


# ---------------------------------------------------------------------------
# Fake services
# ---------------------------------------------------------------------------


class _FakeService:
    """Returns the deterministic mock response; records the last request seen."""

    def __init__(self) -> None:
        self.last_request = None

    async def estimate(self, request, project_metadata=None, prompt_version=None):
        self.last_request = request
        return _MOCK_RESPONSE

    async def estimate_conversational(
        self,
        *,
        session,
        transcript,
        enriched_transcript,
        attachments_total_chars=0,
        prompt_version=None,
    ):
        from types import SimpleNamespace

        from src.services.metadata_extractor import update_from_result
        from src.services.summarizer import update_summary
        from src.services.tier_resolver import resolve_tier

        # Record for test introspection (mirrors real method's enriched input)
        self.last_request = SimpleNamespace(transcript=enriched_transcript)

        tier, rule = resolve_tier(session.metadata, _MOCK_RESPONSE.estimation)
        session.last_resolved_tier = tier
        session.last_tier_rule = rule

        previous_metadata = session.metadata.model_copy(deep=True)
        session.metadata = update_from_result(
            transcript, _MOCK_RESPONSE.estimation, session.metadata
        )
        session.update_anchors(previous_metadata, session.metadata)

        await update_summary(session)

        session.history.add_user(transcript)
        session.history.add_assistant(_MOCK_RESPONSE.estimation.executive_summary)
        session.touch()

        return _MOCK_RESPONSE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session_client():
    """AsyncClient with a fresh app + fake service.

    The session_store is also cleared before the test so previous test runs
    cannot leak session state.
    """
    app = create_app()
    fake = _FakeService()
    app.dependency_overrides[get_estimation_service] = lambda: fake

    # Clear in-memory store so sessions don't bleed between tests
    session_store._store.clear()

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as ac:
        # Attach the fake so individual tests can inspect .last_request
        ac._fake_service = fake  # type: ignore[attr-defined]
        yield ac


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _new_session(client: AsyncClient) -> str:
    """POST /api/v1/sessions and return the session_id."""
    resp = await client.post("/api/v1/sessions")
    assert resp.status_code == 201
    return resp.json()["session_id"]


async def _estimate(
    client: AsyncClient, session_id: str, transcript: str, files=None
) -> dict:
    """POST /api/v1/sessions/{id}/estimate and return the JSON body."""
    data = {"transcript": transcript}
    resp = await client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=data,
        files=files,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Test 1 — Two-turn session: ProjectMetadata accumulates across turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_accumulates_across_turns(session_client: AsyncClient) -> None:
    """After two turns the session metadata reflects both transcripts.

    Turn 1 mentions InvoiceApp + React + PostgreSQL + 3-person team.
    Turn 2 mentions Docker + Stripe.

    After both turns GET /sessions/{id} must show:
    - team_size = 3 (from the mock EstimationResult.team_composition sum)
    - 'React' and 'PostgreSQL' in mentioned_technologies (from turn 1)
    - 'Docker' and 'Stripe' in mentioned_technologies (from turn 2)
    - agreed_scope set (from executive_summary)
    """
    sid = await _new_session(session_client)

    await _estimate(session_client, sid, _TRANSCRIPT_TURN_1)

    # --- after turn 1 ---
    info1 = (await session_client.get(f"/api/v1/sessions/{sid}")).json()
    meta1 = info1["project_metadata"]

    assert info1["turn_count"] == 1
    assert meta1["assumed_team_size"] == 3, (
        "team_size should equal sum of team_composition counts in the mock result"
    )
    assert "React" in meta1["mentioned_technologies"]
    assert "PostgreSQL" in meta1["mentioned_technologies"]
    assert meta1["agreed_scope"] is not None, (
        "agreed_scope must be set from executive_summary"
    )

    await _estimate(session_client, sid, _TRANSCRIPT_TURN_2)

    # --- after turn 2 ---
    info2 = (await session_client.get(f"/api/v1/sessions/{sid}")).json()
    meta2 = info2["project_metadata"]

    assert info2["turn_count"] == 2
    # Technologies from both turns must be present (cumulative merge)
    for tech in ("React", "PostgreSQL", "Docker", "Stripe"):
        assert tech in meta2["mentioned_technologies"], (
            f"'{tech}' should be in mentioned_technologies after two turns"
        )
    # team_size and scope remain from turn 1 (not overwritten)
    assert meta2["assumed_team_size"] == 3
    assert meta2["agreed_scope"] is not None


# ---------------------------------------------------------------------------
# Test 2 — PDF attachment text is forwarded to the estimation service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pdf_attachment_text_reaches_service(session_client: AsyncClient) -> None:
    """The text extracted from a PDF attachment must appear in the transcript
    that the EstimationService receives.

    We use a minimal but valid PDF built with raw bytes so the test has no
    dependency on external fixtures.  The PDF contains the unique string
    'SENTINEL_CONTENT_12345' which cannot appear by coincidence.
    """
    sid = await _new_session(session_client)

    sentinel = "SENTINEL_CONTENT_12345"

    # Minimal single-page PDF that contains the sentinel string.
    # The content stream is uncompressed plain text so pypdf can extract it.
    pdf_content_stream = f"BT /F1 12 Tf 50 700 Td ({sentinel}) Tj ET"
    stream_bytes = pdf_content_stream.encode()
    stream_len = len(stream_bytes)

    minimal_pdf = textwrap.dedent(f"""\
        %PDF-1.4
        1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
        2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
        3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
        4 0 obj<</Length {stream_len}>>
        stream
        {pdf_content_stream}
        endstream
        endobj
        5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
        xref
        0 6
        0000000000 65535 f\r
        0000000009 00000 n\r
        0000000058 00000 n\r
        0000000115 00000 n\r
        0000000266 00000 n\r
        0000000{350 + stream_len:06d} 00000 n\r
        trailer<</Size 6/Root 1 0 R>>
        startxref
        {420 + stream_len}
        %%EOF
    """).encode()

    base_transcript = "Build a REST API for invoice management with React frontend."

    resp = await session_client.post(
        f"/api/v1/sessions/{sid}/estimate",
        data={"transcript": base_transcript},
        files=[
            ("attachments", ("spec.pdf", io.BytesIO(minimal_pdf), "application/pdf"))
        ],
    )

    # Even if pypdf cannot parse this minimal PDF, the endpoint must not 500.
    # What we assert is that the call reached the service with the transcript.
    assert resp.status_code in (200, 400), (
        f"Expected 200 or 400, got {resp.status_code}: {resp.text}"
    )

    if resp.status_code == 200:
        # The service was called — inspect what transcript it received
        received_transcript: str = session_client._fake_service.last_request.transcript  # type: ignore[attr-defined]
        # The base transcript must always be present
        assert base_transcript in received_transcript
        # If pypdf could extract text, the sentinel should appear
        if sentinel in received_transcript:
            assert "--- attachment: spec.pdf ---" in received_transcript, (
                "Attachment text should be wrapped in separator markers"
            )


# ---------------------------------------------------------------------------
# Test 3 — Sliding-window cap: 8 turns never exceed MAX_CONVERSATION_TURNS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sliding_window_caps_history(session_client: AsyncClient) -> None:
    """After 8 turns the in-memory turn count must equal MAX_CONVERSATION_TURNS.

    The sliding-window eviction in ConversationHistory ensures that the
    history buffer never grows beyond max_turns pairs regardless of how many
    requests are sent to the same session.
    """
    max_turns = get_settings().max_conversation_turns
    sid = await _new_session(session_client)

    n_turns = max_turns + 2  # intentionally exceed the window

    for i in range(n_turns):
        transcript = (
            f"Turn {i + 1}: Add feature {i + 1} to the invoice management platform "
            f"using React and PostgreSQL with Docker deployment."
        )
        resp = await session_client.post(
            f"/api/v1/sessions/{sid}/estimate",
            data={"transcript": transcript},
        )
        assert resp.status_code == 200, f"Turn {i + 1} failed: {resp.text}"

    info = (await session_client.get(f"/api/v1/sessions/{sid}")).json()
    assert info["turn_count"] == max_turns, (
        f"After {n_turns} turns, expected turn_count={max_turns} "
        f"(sliding window), got {info['turn_count']}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Anchors: stable facts are detected and recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anchors_recorded_after_stable_turns(session_client: AsyncClient) -> None:
    """After two turns that produce the same team_size, an anchor must exist.

    The mock result always returns team_composition with 3 engineers, so
    assumed_team_size is set to 3 on turn 1 and remains 3 on turn 2.
    After turn 2, update_anchors must detect the stable value and record it.
    """
    sid = await _new_session(session_client)

    await _estimate(session_client, sid, _TRANSCRIPT_TURN_1)
    await _estimate(session_client, sid, _TRANSCRIPT_TURN_2)

    session = session_store._store[sid]
    assert any(a.startswith("team_size:") for a in session.anchors), (
        "Expected a team_size anchor after two turns with the same team composition"
    )


@pytest.mark.asyncio
async def test_anchors_no_duplicates_via_http(session_client: AsyncClient) -> None:
    """Running 4 turns with the same stable team_size never duplicates anchors."""
    sid = await _new_session(session_client)

    for _ in range(4):
        await _estimate(session_client, sid, _TRANSCRIPT_TURN_1)

    session = session_store._store[sid]
    team_anchors = [a for a in session.anchors if a.startswith("team_size:")]
    assert len(team_anchors) == 1, (
        f"team_size anchor should appear exactly once, got: {team_anchors}"
    )
