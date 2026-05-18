"""Input guardrails — validate raw user input before it reaches the LLM.

Three sublayers run in order:
1. Prompt-injection detection (regex)
2. PII heuristics (regex)
3. OpenAI Moderation API (optional — skipped when openai_client is None)

All sublayers follow the *exception, not fix-and-retry* policy: they raise
``InputGuardrailViolation`` immediately on the first violation found.
"""

from __future__ import annotations

import re
from typing import Literal

import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class InputGuardrailViolation(Exception):
    """Raised when input fails a guardrail check."""

    def __init__(
        self,
        message: str,
        reason: Literal["prompt_injection", "pii", "moderation"],
    ) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason


# ---------------------------------------------------------------------------
# Sublayer 1 — Prompt injection (regex)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|prior|all)\s+instructions", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(system|instructions)\s*>", re.IGNORECASE),
    re.compile(r"forget\s+everything", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"new\s+instructions\s*:", re.IGNORECASE),
    re.compile(r"disregard\s+\S+\s*(instructions|rules|context)", re.IGNORECASE),
]


def _check_prompt_injection(text: str) -> None:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "guardrail_violation",
                reason="prompt_injection",
                snippet=text[:80],
            )
            raise InputGuardrailViolation(
                message="Input contains a prompt-injection pattern and was rejected.",
                reason="prompt_injection",
            )


# ---------------------------------------------------------------------------
# Sublayer 2 — PII heuristics (regex)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# IBAN: 2 letters + 2 digits, then groups of digits/letters (with optional spaces)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){4,7}\b")
# International phone: +XX followed by digits/spaces, at least 9 significant digits
_PHONE_INTL_RE = re.compile(r"\+\d{1,3}[\s\-]?\d[\d\s\-]{7,}")
# National phone: 9 or more consecutive digits
_PHONE_NATIONAL_RE = re.compile(r"\b\d{9,}\b")


def _check_pii(text: str) -> None:
    checks = [
        (_EMAIL_RE, "email address"),
        (_IBAN_RE, "IBAN"),
        (_PHONE_INTL_RE, "international phone number"),
        (_PHONE_NATIONAL_RE, "phone number"),
    ]
    for pattern, label in checks:
        if pattern.search(text):
            logger.warning(
                "guardrail_violation",
                reason="pii",
                pii_type=label,
                snippet=text[:80],
            )
            raise InputGuardrailViolation(
                message=f"Input contains a {label} and was rejected to protect privacy.",
                reason="pii",
            )


# ---------------------------------------------------------------------------
# Sublayer 3 — OpenAI Moderation API
# ---------------------------------------------------------------------------


async def _check_moderation(text: str, openai_client: AsyncOpenAI) -> None:
    response = await openai_client.moderations.create(input=text)
    result = response.results[0]
    if result.flagged:
        flagged_categories = [
            cat for cat, flagged in result.categories.model_dump().items() if flagged
        ]
        logger.warning(
            "guardrail_violation",
            reason="moderation",
            categories=flagged_categories,
            snippet=text[:80],
        )
        raise InputGuardrailViolation(
            message="Input was flagged by the content moderation API and was rejected.",
            reason="moderation",
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def check_input(
    description: str, *, openai_client: AsyncOpenAI | None = None
) -> None:
    """Run all input guardrail sublayers against *description*.

    Args:
        description: Raw user-supplied text (e.g. a meeting transcript).
        openai_client: Optional async OpenAI client. When ``None``, the
            moderation sublayer is skipped — useful in tests and offline
            environments.

    Raises:
        InputGuardrailViolation: On the first violation found, with
            ``reason`` set to ``"prompt_injection"``, ``"pii"``, or
            ``"moderation"``.
    """
    _check_prompt_injection(description)
    _check_pii(description)
    if openai_client is not None:
        await _check_moderation(description, openai_client)
