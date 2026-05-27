"""Session state management for multi-turn conversations.

Design decision — in-memory only (no DB, no Redis)
----------------------------------------------------
At this stage the service is single-process and stateless restarts are
acceptable: losing a conversation on redeploy is a known trade-off we take
consciously in exchange for zero infrastructure overhead.  When the product
matures and we need horizontal scaling or session persistence across
deployments, replacing ``_store`` with a Redis backend or a DB-backed
repository will be a localised change confined to this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.core.config import get_settings

if TYPE_CHECKING:
    from src.schemas.estimation import EstimationRequest


# ---------------------------------------------------------------------------
# Message — atomic unit stored inside ConversationHistory
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single LLM message with its role and text content."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Return the wire format expected by every LLM provider."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# ConversationHistory — sliding-window message buffer
# ---------------------------------------------------------------------------


class ConversationHistory:
    """Fixed-depth message buffer with a sliding-window eviction policy.

    Turns are counted as *user+assistant pairs*.  When the buffer exceeds
    ``max_turns`` pairs the oldest pair is discarded.  The system prompt
    (role="system") is **always** preserved at position 0 — it is never
    subject to eviction because removing it would silently degrade model
    behaviour without any visible error.

    Args:
        max_turns: Maximum number of user/assistant pairs to keep.
            Defaults to ``settings.max_conversation_turns`` (6).
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._messages: list[Message] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_system_prompt(self, content: str) -> None:
        """Insert or replace the system prompt at position 0.

        Called once at session creation; may be refreshed if the prompt
        changes (e.g. when ProjectMetadata is first populated).
        """
        system_msg = Message(role="system", content=content)
        if self._messages and self._messages[0].role == "system":
            self._messages[0] = system_msg
        else:
            self._messages.insert(0, system_msg)

    def add_user(self, content: str) -> None:
        """Append a user message and evict stale turns if needed."""
        self._messages.append(Message(role="user", content=content))
        self._evict_if_needed()

    def add_assistant(self, content: str) -> None:
        """Append an assistant message."""
        self._messages.append(Message(role="assistant", content=content))

    def as_dicts(self) -> list[dict[str, str]]:
        """Return the full message list in LLM wire format."""
        return [m.to_dict() for m in self._messages]

    def to_messages_list(self, system_content: str) -> list[dict[str, str]]:
        """Return the message list with a freshly rendered system prompt.

        Replaces whatever system prompt was stored with *system_content*,
        keeping all user/assistant turns intact.  This lets callers
        regenerate the system prompt from the latest ``ProjectMetadata``
        without mutating the buffer state.

        Args:
            system_content: Pre-rendered system prompt string.

        Returns:
            List of ``{role, content}`` dicts ready for any LLM provider.
        """
        non_sys = [m.to_dict() for m in self._messages if m.role != "system"]
        return [{"role": "system", "content": system_content}, *non_sys]

    def __len__(self) -> int:
        """Number of messages currently stored (including system prompt)."""
        return len(self._messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _non_system_messages(self) -> list[Message]:
        return [m for m in self._messages if m.role != "system"]

    def _turn_count(self) -> int:
        """Count completed or in-progress user/assistant pairs."""
        non_sys = self._non_system_messages()
        # Each user message starts a new turn regardless of whether an
        # assistant reply has arrived yet.
        return sum(1 for m in non_sys if m.role == "user")

    def _evict_if_needed(self) -> None:
        """Remove the oldest user+assistant pair when over the limit."""
        while self._turn_count() > self.max_turns:
            # Find the index of the first non-system message (oldest user msg)
            for i, msg in enumerate(self._messages):
                if msg.role == "user":
                    # Remove it
                    self._messages.pop(i)
                    # Remove the immediately following assistant reply, if any
                    if (
                        i < len(self._messages)
                        and self._messages[i].role == "assistant"
                    ):
                        self._messages.pop(i)
                    break


# ---------------------------------------------------------------------------
# ProjectMetadata — facts extracted from the conversation
# ---------------------------------------------------------------------------


class ProjectMetadata(BaseModel):
    """Structured facts inferred or agreed upon during the conversation.

    All fields are optional because metadata is populated incrementally as
    the user provides information across multiple turns.  Callers should
    treat ``None`` as "not yet known" rather than a missing value.
    """

    project_name: str | None = Field(
        default=None,
        description="Human-readable name or codename for the project.",
    )
    assumed_team_size: int | None = Field(
        default=None,
        ge=1,
        description="Number of engineers assumed for the estimate.",
    )
    mentioned_technologies: list[str] = Field(
        default_factory=list,
        description="Tech stack items mentioned by the user (e.g. 'React', 'PostgreSQL').",
    )
    agreed_scope: str | None = Field(
        default=None,
        description="Free-text summary of the scope as confirmed with the user.",
    )


# ---------------------------------------------------------------------------
# Session — container for one user conversation
# ---------------------------------------------------------------------------


class Session:
    """All conversational state scoped to a single session identifier.

    Bundles a :class:`ConversationHistory` and :class:`ProjectMetadata`
    under one roof so that service methods receive a single object instead
    of two separate lookups.

    Args:
        session_id: Caller-supplied opaque identifier (UUID recommended).
        max_turns: Forwarded to the underlying ConversationHistory buffer.
    """

    def __init__(self, session_id: str, max_turns: int | None = None) -> None:
        self.session_id = session_id
        resolved_turns = (
            max_turns
            if max_turns is not None
            else get_settings().max_conversation_turns
        )
        self.history = ConversationHistory(max_turns=resolved_turns)
        self.metadata = ProjectMetadata()
        self.anchors: list[str] = []
        self.accumulated_summary: str = ""
        self.last_resolved_tier: str = "unknown"
        self.last_tier_rule: str = "no_match"
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_active: datetime = self.created_at

    @property
    def summary_chars(self) -> int:
        """Number of characters in the accumulated summary."""
        return len(self.accumulated_summary)

    def touch(self) -> None:
        """Update the last-active timestamp (call on every turn)."""
        self.last_active = datetime.now(timezone.utc)

    def update_anchors(self, previous: ProjectMetadata, new: ProjectMetadata) -> None:
        """Detect stable facts between two consecutive turns and record them.

        An "anchor" is a fact that has remained unchanged between *previous*
        and *new* metadata.  Only non-None values are considered stable.
        Anchors are deduplicated: adding the same anchor twice is a no-op.

        Args:
            previous: Metadata snapshot from *before* this turn.
            new: Metadata snapshot from *after* this turn.
        """

        def _add(anchor: str) -> None:
            if anchor not in self.anchors:
                self.anchors.append(anchor)

        if (
            previous.project_name is not None
            and previous.project_name == new.project_name
        ):
            _add(f"project_name:{new.project_name}")

        if (
            previous.assumed_team_size is not None
            and previous.assumed_team_size == new.assumed_team_size
        ):
            _add(f"team_size:{new.assumed_team_size}")

        common_techs = set(t.lower() for t in previous.mentioned_technologies) & set(
            t.lower() for t in new.mentioned_technologies
        )
        for tech in sorted(common_techs):
            _add(f"tech:{tech}")

    def to_messages_list(
        self,
        request: EstimationRequest,
        version: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages array for the LLM with a fresh system prompt.

        Regenerates the system prompt from the *current* ``self.metadata``
        so that every turn benefits from accumulated context (project name,
        team size, technologies, agreed scope) without requiring the caller
        to manage prompt state separately.

        The lazy import of ``render_estimation_prompt`` breaks the
        ``sessions → loader → sessions`` circular dependency at import time
        while keeping this convenience method on the domain object.

        Args:
            request: The current ``EstimationRequest`` (provides transcript
                and any per-request overrides used by the Jinja2 templates).
            version: Prompt template version.  Defaults to
                ``settings.prompt_version`` when ``None``.

        Returns:
            List of ``{role, content}`` dicts in LLM wire format:
            ``[system, ...user/assistant turns...]``
        """
        # Lazy import to avoid circular dependency:
        #   sessions.py → loader.py → sessions.py (ProjectMetadata)
        from src.prompts.loader import render_estimation_prompt  # noqa: PLC0415

        resolved_version = version or get_settings().prompt_version
        system_content, _ = render_estimation_prompt(
            request,
            version=resolved_version,
            project_metadata=self.metadata,
        )
        return self.history.to_messages_list(system_content)


# ---------------------------------------------------------------------------
# SessionStore — in-memory registry (module-level singleton)
# ---------------------------------------------------------------------------


class SessionStore:
    """Thread-unsafe in-memory registry of active sessions.

    Volatility accepted at this phase
    ----------------------------------
    Process restart or horizontal scaling will lose all sessions.  This is
    intentional: the goal right now is to prove the multi-turn UX, not to
    build production infrastructure.  When persistence is needed, swap this
    class for one backed by Redis (using the same ``get`` / ``get_or_create``
    interface) without touching any caller.

    The store is not thread-safe.  FastAPI runs on an async event loop in a
    single OS thread by default, so concurrent modification is not a concern
    for the current deployment model.
    """

    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def get(self, session_id: str) -> Session | None:
        """Return an existing session or ``None`` if it does not exist."""
        return self._store.get(session_id)

    def get_or_create(self, session_id: str, max_turns: int | None = None) -> Session:
        """Return the session for *session_id*, creating it if absent."""
        if session_id not in self._store:
            self._store[session_id] = Session(session_id, max_turns=max_turns)
        return self._store[session_id]

    def delete(self, session_id: str) -> bool:
        """Remove a session. Returns ``True`` if it existed."""
        return self._store.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._store)


# Module-level singleton — import this in service layer code.
session_store: SessionStore = SessionStore()
