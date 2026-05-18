"""LiteLLM wrapper — two complementary interfaces for LLM calls.

1. LLMWrapper (class)
   Instructor + litellm.completion for structured outputs.
   Calls litellm.completion directly — no Router, no round-robin — so the
   model used for a structured call is always deterministic.

2. complete / stream_complete (module-level async functions)
   LiteLLM Router with automatic primary→fallback cascade.
   Used by stream_estimation where structured output is not required.
"""

import asyncio
import functools
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, TypeVar

import instructor
import litellm
from instructor.exceptions import InstructorRetryException  # noqa: F401  (re-exported)
from litellm import Router

from src.core.config import get_settings
from src.core.exceptions import ProviderAuthError, ProviderRateLimitError
from src.core.logging import logger

# Silence verbose LiteLLM logs in production
litellm.suppress_debug_info = True

T = TypeVar("T")


# ---------------------------------------------------------------------------
# LLMWrapper — structured output via instructor.from_litellm
# ---------------------------------------------------------------------------


class LLMWrapper:
    """Thin wrapper around Instructor + LiteLLM for structured LLM calls.

    instructor.from_litellm(litellm.completion) is the only transport used
    here, so the same code path works for every provider LiteLLM supports
    — no SDK-level lock-in to OpenAI or Anthropic.

    Args:
        openai_api_key: OpenAI API key.
        anthropic_api_key: Anthropic API key.
        primary_model: Default model used when no model_override is given.
        fallback_model: Stored for reference; callers may pass it as
            model_override after catching an exception on the primary.
        timeout: Per-request timeout in seconds forwarded to litellm.
        num_retries: Network-level retries forwarded to litellm (distinct
            from Instructor's validation retries).
    """

    def __init__(
        self,
        openai_api_key: str,
        anthropic_api_key: str,
        primary_model: str,
        fallback_model: str,
        timeout: int,
        num_retries: int,
    ) -> None:
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._timeout = timeout
        self._num_retries = num_retries

        # instructor.from_litellm wraps the sync litellm.completion callable.
        # Instructor handles schema extraction, JSON parsing, and validation
        # retries; LiteLLM handles provider routing, auth, and transport.
        self._instructor = instructor.from_litellm(litellm.completion)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _api_key_for(self, model: str) -> str:
        """Return the correct API key by inspecting the model name.

        Models whose name starts with 'claude' are Anthropic models.
        All others are assumed to be OpenAI-compatible.
        """
        if model.startswith("claude"):
            return self._anthropic_api_key
        return self._openai_api_key

    def _provider_for(self, model: str) -> str:
        return "anthropic" if model.startswith("claude") else "openai"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
        model_override: str | None = None,
        max_retries: int = 6,
    ) -> tuple[T, dict]:
        """Call an LLM via Instructor + LiteLLM and return a validated Pydantic instance.

        Runs litellm.completion in a thread pool (asyncio.to_thread) because
        instructor.from_litellm wraps the synchronous litellm.completion.

        Args:
            system_prompt: Content for the system role.
            user_message: Content for the user role.
            response_model: Pydantic class the LLM must fill.
            model_override: When set, overrides self._primary_model.
            max_retries: Instructor-level retries on Pydantic validation failure.
                Each retry re-sends the ValidationError to the LLM as context.

        Returns:
            Tuple of (pydantic_instance, metadata).
            metadata keys: model (str), provider (str), latency_ms (float),
                           input_tokens (int), output_tokens (int).

        Raises:
            InstructorRetryException: All max_retries exhausted without a valid
                response. Propagated as-is so the service layer maps it to HTTP.
        """
        model = model_override or self._primary_model
        api_key = self._api_key_for(model)
        provider = self._provider_for(model)

        call_logger = logger.bind(model=model, provider=provider)
        call_logger.info("structured_call_started")
        start = time.monotonic()

        fn = functools.partial(
            self._instructor.chat.completions.create_with_completion,
            response_model=response_model,
            model=model,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            api_key=api_key,
            timeout=self._timeout,
            num_retries=self._num_retries,
        )

        try:
            result, completion = await asyncio.to_thread(fn)
        except InstructorRetryException:
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            call_logger.error(
                "structured_call_retries_exhausted",
                latency_ms=latency_ms,
                max_retries=max_retries,
            )
            raise

        latency_ms = round((time.monotonic() - start) * 1000, 1)
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        call_logger.info(
            "structured_call_completed",
            latency_ms=latency_ms,
            tokens_in=input_tokens,
            tokens_out=output_tokens,
        )

        return result, {
            "model": getattr(completion, "model", model),
            "provider": provider,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    """Return a cached LLMWrapper instance (singleton).

    primary_model is the provider configured in settings.llm_provider;
    fallback_model is the other provider's model.
    """
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        primary = settings.anthropic_model
        fallback = settings.openai_model
    else:
        primary = settings.openai_model
        fallback = settings.anthropic_model

    return LLMWrapper(
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        primary_model=primary,
        fallback_model=fallback,
        timeout=60,
        num_retries=1,
    )


# ---------------------------------------------------------------------------
# Router-based functions — kept for stream_estimation
# ---------------------------------------------------------------------------


@lru_cache
def _get_router() -> Router:
    """Build and cache the LiteLLM Router (singleton).

    Used only for streaming, where structured output is not needed and
    automatic primary→fallback cascade is desirable.
    """
    settings = get_settings()
    return Router(
        model_list=[
            {
                "model_name": "estimation-model",
                "litellm_params": {"model": model},
            }
            for model in settings.llm_models
        ],
        num_retries=1,
        timeout=60,
    )


def get_router() -> Router:
    """Return the cached LiteLLM Router instance."""
    return _get_router()


async def complete(messages: list[dict[str, str]]) -> Any:
    """Send a single completion request through the LiteLLM Router."""
    router = _get_router()
    try:
        return await router.acompletion(model="estimation-model", messages=messages)
    except litellm.AuthenticationError as exc:
        raise ProviderAuthError() from exc
    except litellm.RateLimitError as exc:
        raise ProviderRateLimitError() from exc


async def stream_complete(
    messages: list[dict[str, str]],
) -> AsyncGenerator[Any, None]:
    """Send a streaming completion request through the LiteLLM Router."""
    router = _get_router()
    try:
        stream = await router.acompletion(
            model="estimation-model",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
    except litellm.AuthenticationError as exc:
        raise ProviderAuthError() from exc
    except litellm.RateLimitError as exc:
        raise ProviderRateLimitError() from exc

    async for chunk in stream:
        yield chunk
