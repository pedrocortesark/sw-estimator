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

from openai import AsyncOpenAI

from src.cache.semantic import EstimationSemanticCache
from src.guardrails.input import check_input
from src.guardrails.output import enforce_scope_response
from src.prompts.loader import render_estimation_prompt
from src.schemas.estimation import EstimationRequest, EstimationResponse
from src.services.llm_wrapper import get_llm_wrapper


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
        self._cache = cache or EstimationSemanticCache()
        self._openai_client = openai_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def estimate(self, request: EstimationRequest) -> EstimationResponse:
        """Run the estimation pipeline and return a structured response.

        Raises:
            InputGuardrailViolation: If Layer 1 rejects the input.
            InstructorRetryException: If Layer 4 exhausts LLM retries
                (translated to HTTP 502 by the exception handler in
                src/core/exceptions.py).
        """
        # --- Layer 1: input guardrails -----------------------------------
        await self._run_input_guardrails(request)

        # --- Layer 2: semantic cache (read) ------------------------------
        cached = self._lookup_cache(request)
        if cached is not None:
            return cached

        # --- Layer 3: prompt rendering -----------------------------------
        system_prompt, user_prompt = self._render_prompts(request)

        # --- Layer 4: LLM structured call --------------------------------
        estimation_result, meta = await self._call_llm(
            system_prompt, user_prompt, request
        )

        # --- Layer 5: output guardrails + cache write --------------------
        estimation_result = self._run_output_guardrails(estimation_result)
        self._store_cache(request, estimation_result)

        return self._build_response(estimation_result, meta)

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

    def _lookup_cache(self, request: EstimationRequest) -> EstimationResponse | None:
        """Layer 2 (read) — return a cached response for semantically similar input.

        Uses cosine similarity over transcript embeddings. Returns ``None`` on a
        cache miss so the pipeline continues to Layer 3.

        TODO: wire in an actual vector store (e.g. Redis + pgvector / Qdrant).
        """
        return self._cache.lookup(request)  # type: ignore[return-value]

    def _render_prompts(self, request: EstimationRequest) -> tuple[str, str]:
        """Layer 3 — render Jinja2 templates into (system_prompt, user_prompt).

        Template resolution: ``src/prompts/estimation/v1/{system,user}.j2``

        TODO: pass template variables (schema hints, few-shot examples, etc.)
        once the Jinja2 templates are filled in.
        """
        return render_estimation_prompt(request)

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        request: EstimationRequest,
    ) -> tuple:
        """Layer 4 — call the LLM and parse a structured ``EstimationResult``.

        Uses ``LLMWrapper.complete_structured()`` which wraps
        ``instructor.from_litellm(litellm.completion)`` — no provider lock-in.
        Instructor retries up to ``max_retries=3`` times if the LLM output
        fails the ``@model_validator`` arithmetic consistency checks.

        Returns:
            ``(EstimationResult, meta_dict)`` where ``meta_dict`` contains
            ``model``, ``provider``, ``input_tokens``, ``output_tokens``, and
            ``latency_ms``.

        Raises:
            InstructorRetryException: If all retries are exhausted.
        """
        from src.schemas.estimation import EstimationResult

        return await get_llm_wrapper().complete_structured(
            system_prompt=system_prompt,
            user_message=user_prompt,
            response_model=EstimationResult,
            model_override=None,  # TODO: resolve from request.provider
        )

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
        """Layer 5b — persist the result for future semantically-similar requests.

        TODO: wire in the same vector store used by ``_lookup_cache``.
        """
        self._cache.store(request, estimation_result)

    def _build_response(self, estimation_result, meta: dict) -> EstimationResponse:
        """Assemble the final ``EstimationResponse`` from the LLM result and metadata.

        TODO: move cost calculation here (currently lives in llm_service.py).
        """
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
        )
