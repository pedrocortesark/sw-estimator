"""Unit tests for the update_summary summarizer service.

Two scenarios:
1. Below threshold — the summarizer must NOT call the LLM when the history
   has not yet reached max_turns * 2 non-system messages.
2. At threshold — the summarizer MUST call the LLM and update
   session.accumulated_summary when the window is full.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.sessions import Session
from src.services.summarizer import update_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(max_turns: int = 3) -> Session:
    return Session(session_id="summarizer-test", max_turns=max_turns)


def _fill_history(session: Session, n_pairs: int) -> None:
    """Add n_pairs of user+assistant messages directly to the history buffer
    (bypassing _evict_if_needed so we can fill to exactly the threshold)."""
    for i in range(n_pairs):
        session.history._messages.append(
            # Import inline to avoid circular import in test module
            __import__(
                "src.services.sessions", fromlist=["Message"]
            ).Message(role="user", content=f"user turn {i}")
        )
        session.history._messages.append(
            __import__(
                "src.services.sessions", fromlist=["Message"]
            ).Message(role="assistant", content=f"assistant turn {i}")
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_not_called_below_threshold() -> None:
    """With max_turns=3 and only 1 prior turn (2 messages), the summarizer
    must NOT invoke the LLM — the threshold is 3*2=6 non-system messages."""
    session = _make_session(max_turns=3)
    _fill_history(session, n_pairs=1)  # 2 non-system messages < 6

    with patch("src.services.summarizer.complete", new_callable=AsyncMock) as mock_complete:
        await update_summary(session)

    mock_complete.assert_not_called()
    assert session.accumulated_summary == ""


@pytest.mark.asyncio
async def test_summary_not_called_one_below_threshold() -> None:
    """With max_turns=3 and 2 prior pairs (4 messages), still below threshold
    of 6 — the LLM must NOT be called."""
    session = _make_session(max_turns=3)
    _fill_history(session, n_pairs=2)  # 4 non-system messages < 6

    with patch("src.services.summarizer.complete", new_callable=AsyncMock) as mock_complete:
        await update_summary(session)

    mock_complete.assert_not_called()


@pytest.mark.asyncio
async def test_summary_called_at_threshold() -> None:
    """With max_turns=3 and exactly 3 pairs (6 messages), the summarizer
    MUST call the LLM and update session.accumulated_summary."""
    session = _make_session(max_turns=3)
    _fill_history(session, n_pairs=3)  # 6 non-system messages == threshold

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "  This is the accumulated summary.  "

    with patch("src.services.summarizer.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = fake_response
        await update_summary(session)

    mock_complete.assert_called_once()
    assert session.accumulated_summary == "This is the accumulated summary."
    assert session.summary_chars == len("This is the accumulated summary.")


@pytest.mark.asyncio
async def test_summary_called_above_threshold() -> None:
    """With more than max_turns pairs in the buffer, the summarizer still
    fires and updates the summary."""
    session = _make_session(max_turns=3)
    _fill_history(session, n_pairs=5)  # 10 messages > threshold

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "Extended summary text."

    with patch("src.services.summarizer.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = fake_response
        await update_summary(session)

    mock_complete.assert_called_once()
    assert session.accumulated_summary == "Extended summary text."


@pytest.mark.asyncio
async def test_summary_exception_is_swallowed() -> None:
    """If the LLM call raises, update_summary must NOT propagate the exception
    and must leave accumulated_summary unchanged."""
    session = _make_session(max_turns=3)
    _fill_history(session, n_pairs=3)

    with patch(
        "src.services.summarizer.complete",
        new_callable=AsyncMock,
        side_effect=RuntimeError("network error"),
    ):
        await update_summary(session)  # must not raise

    assert session.accumulated_summary == ""


@pytest.mark.asyncio
async def test_summary_accumulates_previous() -> None:
    """The rendered prompt includes the previous summary when it is non-empty."""
    session = _make_session(max_turns=3)
    session.accumulated_summary = "Previous summary from last window."
    _fill_history(session, n_pairs=3)

    captured_messages: list = []

    async def _capture(messages):
        captured_messages.extend(messages)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Updated summary."
        return resp

    with patch("src.services.summarizer.complete", side_effect=_capture):
        await update_summary(session)

    assert session.accumulated_summary == "Updated summary."
    # The prompt sent to the LLM must mention the previous summary
    assert any(
        "Previous summary from last window." in m["content"]
        for m in captured_messages
    )


def test_summary_chars_property() -> None:
    """summary_chars always reflects the current length of accumulated_summary."""
    session = _make_session()
    assert session.summary_chars == 0

    session.accumulated_summary = "Hello world"
    assert session.summary_chars == 11
