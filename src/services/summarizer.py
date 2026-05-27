"""Summarizer service — accumulates a rolling summary of the conversation.

The summary is updated only when the sliding window is full (i.e., about to
evict the oldest turn).  At that point the current messages are compressed
into a concise paragraph so the evicted context is not lost entirely.

This is intentionally fire-and-forget: any LLM or network error is caught,
logged, and swallowed so a summary failure never disrupts the main
estimation flow.
"""

from __future__ import annotations

from src.core.logging import logger
from src.prompts.loader import render_summarizer_prompt
from src.services.llm_wrapper import complete
from src.services.sessions import Session


async def update_summary(session: Session) -> None:
    """Update the session's accumulated summary when the window is full.

    Triggers only when the number of stored messages (excluding the system
    prompt) is about to hit the sliding-window cap — that is, when the
    current non-system message count equals ``max_turns * 2``.  At this
    point the oldest pair is about to be evicted, so we compress the
    current window into a paragraph before the turn is added.

    The summary is built by calling the LiteLLM Router (module-level
    ``complete`` function) with a rendered Jinja2 prompt.  This is a
    plain text call — no structured output required.

    Args:
        session: The active session whose history and summary to inspect
            and potentially update.
    """
    # Count only user/assistant messages (exclude system prompt).
    non_system = [m for m in session.history._messages if m.role != "system"]
    threshold = session.history.max_turns * 2

    if len(non_system) < threshold:
        return

    messages_for_prompt = [m.to_dict() for m in non_system]
    prompt_text = render_summarizer_prompt(session.accumulated_summary, messages_for_prompt)

    try:
        response = await complete(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ]
        )
        # Extract the text content from the LiteLLM response object.
        summary_text: str = response.choices[0].message.content or ""
        session.accumulated_summary = summary_text.strip()
        logger.info(
            "session_summary_updated",
            session_id=session.session_id,
            summary_chars=session.summary_chars,
        )
    except Exception:
        logger.warning(
            "session_summary_failed",
            session_id=session.session_id,
            exc_info=True,
        )
