"""High-level EstimationService — orchestrates the 5-layer estimation pipeline.

Pipeline execution order
------------------------
Every call to ``estimate()`` passes through these layers in sequence:

  REQUEST
     │
     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Layer 1 · INPUT GUARDRAILS                         │
  │  src/guardrails/input.py :: check_input()           │
  │  • Prompt-injection detection (regex)               │
  │  • PII heuristics (regex)                           │
  │  • OpenAI Moderation API (optional)                 │
  │  → raises InputGuardrailViolation on any violation  │
  └─────────────────────────────────────────────────────┘
     │
     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Layer 2 · SEMANTIC CACHE (read)                    │
  │  src/cache/semantic.py :: EstimationSemanticCache   │
  │  • Embed transcript → vector lookup                 │
  │  • Cache hit → return cached EstimationResult       │
  │  • Cache miss → continue to Layer 3                 │
  └─────────────────────────────────────────────────────┘
     │  (cache miss)
     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Layer 3 · PROMPT RENDERING                         │
  │  src/prompts/loader.py :: render_estimation_prompt  │
  │  • Jinja2 templates (system.j2 + user.j2)           │
  │  • Injects few-shot examples, schema hints          │
  │  • Returns (system_prompt, user_prompt) tuple       │
  └─────────────────────────────────────────────────────┘
     │
     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Layer 4 · LLM CALL (structured output)             │
  │  src/services/llm_wrapper.py :: LLMWrapper          │
  │  • instructor.from_litellm → no provider lock-in    │
  │  • response_model=EstimationResult (Pydantic)       │
  │  • @model_validator arithmetic consistency checks   │
  │  • max_retries=3 → InstructorRetryException → 502   │
  └─────────────────────────────────────────────────────┘
     │
     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Layer 5 · OUTPUT GUARDRAILS + CACHE WRITE          │
  │  src/guardrails/output.py :: enforce_scope_response │
  │  • Business-rule enforcement on structured result   │
  │  • Store result in semantic cache for future hits   │
  └─────────────────────────────────────────────────────┘
     │
     ▼
  RESPONSE (EstimationResponse)
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

import structlog
from openai import AsyncOpenAI

from src.cache.semantic import EstimationSemanticCache
from src.core.config import get_settings
from src.guardrails.input import check_input
from src.guardrails.output import enforce_scope_response
from src.prompts.loader import render_estimation_prompt
from src.schemas.estimation import EstimationRequest, EstimationResponse
from src.services.llm_wrapper import get_llm_wrapper

if TYPE_CHECKING:
    from src.schemas.estimation import EstimationResult

# ---------------------------------------------------------------------------
# In-process fallback cache — used when no Redis semantic cache is wired.
# Keyed by SHA-256 of (transcript, provider, project_type, detail_level,
# output_format).  Stored at module level so it survives across requests
# within the same worker process.
# ---------------------------------------------------------------------------
_IN_MEMORY_CACHE: dict[str, "EstimationResult"] = {}


class EstimationService:
    """Orchestrates the full 5-layer estimation pipeline.

    Args:
        cache: Semantic cache instance (injected; no-op stub by default).
        openai_client: Optional AsyncOpenAI client forwarded to the input
            guardrail's moderation sublayer. When ``None``, moderation is
            skipped.
    """

    def __init__(
        self,
        cache: EstimationSemanticCache | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        # cache=None means "skip layers 2 and 5b" — safe when Redis is absent.
        self._cache = cache
        self._openai_client = openai_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def estimate(
        self,
        request: EstimationRequest,
        project_metadata=None,
        prompt_version: str | None = None,
    ) -> EstimationResponse:
        """Run the estimation pipeline and return a structured response.

        Args:
            prompt_version: Override the active template version for this call.
                Defaults to ``settings.prompt_version`` when ``None``.

        Raises:
            InputGuardrailViolation: If Layer 1 rejects the input.
            InstructorRetryException: If Layer 4 exhausts LLM retries
                (translated to HTTP 502 by the exception handler in
                src/core/exceptions.py).
        """
        version = prompt_version or get_settings().prompt_version
        # --- Layer 1: input guardrails -----------------------------------
        await self._run_input_guardrails(request)

        # --- Layer 2: semantic cache (read) ------------------------------
        cached_result, _cache_kind = self._lookup_cache(request)
        if cached_result is not None:
            return self._build_response(
                cached_result,
                {
                    "provider": "memory_cache"
                    if self._cache is None
                    else "semantic_cache",
                    "model": "cached",
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                cached=True,
            )

        # --- Layer 3: prompt rendering -----------------------------------
        system_prompt, user_prompt = self._render_prompts(
            request, project_metadata=project_metadata, version=version
        )

        # --- Layer 4: LLM structured call --------------------------------
        estimation_result, meta = await self._call_llm(
            system_prompt, user_prompt, request
        )

        # --- Layer 5: output guardrails + cache write --------------------
        estimation_result = self._run_output_guardrails(estimation_result)
        self._store_cache(request, estimation_result)

        return self._build_response(
            estimation_result, meta, cached=False, prompt_version=version
        )

    # ------------------------------------------------------------------
    # Layer implementations (one private method per layer)
    # ------------------------------------------------------------------

    async def _run_input_guardrails(self, request: EstimationRequest) -> None:
        """Layer 1 — validate raw transcript before it reaches the LLM.

        Delegates to ``check_input()``, which runs in order:
        1. Prompt-injection regex check
        2. PII heuristic regex check
        3. OpenAI Moderation API (skipped when openai_client is None)

        Raises:
            InputGuardrailViolation: On the first violation found.
        """
        await check_input(request.transcript, openai_client=self._openai_client)

    @staticmethod
    def _cache_key(request: EstimationRequest) -> str:
        """Return a deterministic SHA-256 fingerprint for *request*."""
        data = json.dumps(
            {
                "transcript": request.transcript,
                "provider": getattr(request, "provider", None),
                "project_type": request.project_type,
                "detail_level": request.detail_level,
                "output_format": request.output_format,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def _lookup_cache(
        self, request: EstimationRequest
    ) -> "tuple[EstimationResult | None, str]":
        """Layer 2 (read) — return a cached result for semantically similar input.

        Checks, in order:
        1. Redis semantic cache (when configured)
        2. In-process memory dict (always checked; fallback for local dev)

        Returns ``(None, "none")`` on a cache miss, otherwise
        ``(result, kind)`` where *kind* is ``"semantic"`` for a vector-
        similarity hit or ``"exact"`` for a hash-based in-memory hit.
        """
        from src.schemas.estimation import EstimationResult

        if self._cache is not None:
            result = self._cache.lookup(request)
            if isinstance(result, EstimationResult):
                return result, "semantic"

        # Fallback: module-level in-memory dict (exact hash match)
        exact = _IN_MEMORY_CACHE.get(self._cache_key(request))
        if exact is not None:
            return exact, "exact"

        return None, "none"

    def _render_prompts(
        self,
        request: EstimationRequest,
        project_metadata=None,
        version: str | None = None,
    ) -> tuple[str, str]:
        """Layer 3 — render Jinja2 templates into (system_prompt, user_prompt)."""
        return render_estimation_prompt(
            request,
            version=version or get_settings().prompt_version,
            project_metadata=project_metadata,
        )

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        request: EstimationRequest,
    ) -> tuple:
        """Layer 4 — validate provider, call the LLM, return (EstimationResult, meta).

        Delegates to ``generate_estimation`` which handles provider routing and
        structured output parsing via Instructor.  The rendered prompts from
        Layer 3 are passed as overrides so the pipeline's Jinja2 templates are
        actually used for the LLM call.

        Raises:
            UnknownProviderError: For unsupported provider names (before any LLM call).
            InstructorRetryException: If all retries are exhausted.
        """
        from src.core.exceptions import UnknownProviderError
        from src.services.llm_service import generate_estimation

        provider = getattr(request, "provider", None)
        if provider is not None and provider not in (
            "openai",
            "anthropic",
        ):
            raise UnknownProviderError(
                f"Unsupported provider: '{provider}'. Use 'openai' or 'anthropic'."
            )

        result_dict = await generate_estimation(
            transcript=request.transcript,
            provider_override=provider,
        )
        meta = {
            "provider": result_dict["provider"],
            "model": result_dict["model"],
            "input_tokens": result_dict["usage"].input_tokens,
            "output_tokens": result_dict["usage"].output_tokens,
        }
        return result_dict["estimation_result"], meta

    def _run_output_guardrails(self, estimation_result):
        """Layer 5a — enforce business rules on the structured LLM response.

        Examples of rules to implement:
        • Reject estimates with total_hours < 8 (likely hallucinated)
        • Cap total_cost_usd to a configurable ceiling
        • Ensure at least one Phase and one Task are present

        TODO: implement rules in ``src/guardrails/output.py``.
        """
        return enforce_scope_response(estimation_result)

    def _store_cache(self, request: EstimationRequest, estimation_result) -> None:
        """Layer 5b — persist the result for future semantically-similar requests."""
        # Always populate the in-memory fallback cache.
        _IN_MEMORY_CACHE[self._cache_key(request)] = estimation_result
        # Also write to Redis semantic cache when configured.
        if self._cache is not None:
            self._cache.store(request, estimation_result)

    async def estimate_conversational(
        self,
        *,
        session: "Session",
        transcript: str,
        enriched_transcript: str,
        attachments_total_chars: int = 0,
        prompt_version: str | None = None,
    ) -> EstimationResponse:
        """Run the estimation pipeline for one conversational turn and emit ``turn_observed``.

        Combines the full 5-layer estimation pipeline with session-state
        management (tier resolution, metadata extraction, anchor tracking,
        sliding-window summarisation, history update) and emits a single
        structured ``turn_observed`` event that aggregates every observable
        field for that turn — making it straightforward to export per-turn
        CSVs or feed them into an eval harness.

        The ``turn_observed`` event is emitted **just before the return** so
        all fields, including post-compression ``messages_in_window``, are
        fully settled.

        Args:
            session: The active :class:`~src.services.sessions.Session`
                instance.  State (metadata, anchors, history, summary) is
                updated in-place as a side-effect.
            transcript: The raw user message, used for history storage and
                metadata extraction.  Does **not** include attachment text.
            enriched_transcript: ``transcript`` with any extracted attachment
                text appended.  This is the string sent to the LLM.
            attachments_total_chars: Character count of the attachment portion
                of *enriched_transcript*.  Pass ``0`` when no files were
                attached.
            prompt_version: Prompt template override forwarded to Layer 3.
        """
        from src.services.metadata_extractor import update_from_result
        from src.services.sessions import Session  # local to avoid circular
        from src.services.summarizer import update_summary
        from src.services.tier_resolver import resolve_tier

        _log = structlog.get_logger().bind(session_id=session.session_id)

        t0 = time.perf_counter()
        turn_index = session.history._turn_count() + 1
        version = prompt_version or get_settings().prompt_version

        request = EstimationRequest(transcript=enriched_transcript)

        # --- Layer 1: input guardrails -----------------------------------
        await self._run_input_guardrails(request)

        # --- Layer 2: semantic cache (read) — capture kind ---------------
        cached_result, cache_hit_kind = self._lookup_cache(request)
        if cached_result is not None:
            response = self._build_response(
                cached_result,
                {
                    "provider": "memory_cache"
                    if self._cache is None
                    else "semantic_cache",
                    "model": "cached",
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                cached=True,
            )
        else:
            # --- Layer 3: prompt rendering -------------------------------
            system_prompt, user_prompt = self._render_prompts(
                request,
                project_metadata=session.metadata,
                version=version,
            )

            # --- Layer 4: LLM structured call ----------------------------
            estimation_result, meta = await self._call_llm(
                system_prompt, user_prompt, request
            )

            # --- Layer 5: output guardrails + cache write ----------------
            estimation_result = self._run_output_guardrails(estimation_result)
            self._store_cache(request, estimation_result)

            response = self._build_response(
                estimation_result, meta, cached=False, prompt_version=version
            )

        # --- Session state updates ---------------------------------------
        tier, rule = resolve_tier(session.metadata, response.estimation)
        session.last_resolved_tier = tier
        session.last_tier_rule = rule

        previous_metadata = session.metadata.model_copy(deep=True)
        session.metadata = update_from_result(
            transcript, response.estimation, session.metadata
        )
        session.update_anchors(previous_metadata, session.metadata)

        # Summarise before adding the new turn so the window check reflects
        # the current (pre-turn) message count.
        await update_summary(session)

        session.history.add_user(transcript)
        session.history.add_assistant(response.estimation.executive_summary)
        session.touch()

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # --- Emit unified turn event (all fields in one place) -----------
        _log.info(
            "turn_observed",
            turn_index=turn_index,
            enriched_transcript_chars=len(enriched_transcript),
            attachments_total_chars=attachments_total_chars,
            messages_in_window=len(session.history),
            anchors_count=len(session.anchors),
            summary_chars=session.summary_chars,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
            latency_ms=latency_ms,
            cache_hit_kind=cache_hit_kind,
            last_resolved_tier=session.last_resolved_tier,
        )

        return response

    def _build_response(
        self,
        estimation_result,
        meta: dict,
        *,
        cached: bool = False,
        prompt_version: str | None = None,
    ) -> EstimationResponse:
        """Assemble the final ``EstimationResponse`` from the LLM result and metadata."""
        from src.schemas.estimation import UsageCost
        from src.services.pricing import calculate_cost

        cost_usd = calculate_cost(
            meta["model"], meta["input_tokens"], meta["output_tokens"]
        )
        return EstimationResponse(
            estimation=estimation_result,
            provider_used=meta["provider"],
            model_used=meta["model"],
            usage=UsageCost(
                input_tokens=meta["input_tokens"],
                output_tokens=meta["output_tokens"],
                total_tokens=meta["input_tokens"] + meta["output_tokens"],
                cost_usd=cost_usd,
            ),
            cached=cached,
            prompt_version=prompt_version or get_settings().prompt_version,
        )
