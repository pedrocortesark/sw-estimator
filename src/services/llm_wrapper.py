"""LiteLLM wrapper — low-level adapter for the LiteLLM Router.

Encapsulates all LiteLLM-specific concerns:
- Router construction and caching.
- Raw completion calls (sync and streaming).
- Mapping LiteLLM exceptions to domain exceptions.

Higher-level services should depend on this module, not on litellm directly.
"""

from functools import lru_cache
from typing import AsyncGenerator, Any

import litellm
from litellm import Router

from src.core.config import get_settings
from src.core.exceptions import ProviderAuthError, ProviderRateLimitError

# Silence verbose LiteLLM logs in production
litellm.suppress_debug_info = True


@lru_cache
def _get_router() -> Router:
    """Build and cache the LiteLLM Router (singleton).

    The Router is created once and reused for all requests. It iterates
    over settings.llm_models in order: the first is the primary, the rest
    are fallbacks. All share the same alias so LiteLLM handles the cascade.
    LiteLLM reads ANTHROPIC_API_KEY and OPENAI_API_KEY from the environment.
    """
    settings = get_settings()
    return Router(
        model_list=[
            {
                "model_name": "estimation-model",  # same alias = automatic cascade
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
    """Send a single completion request through the LiteLLM Router.

    Args:
        messages: List of chat messages in OpenAI format.

    Returns:
        LiteLLM completion response object.

    Raises:
        ProviderAuthError: When authentication fails for all configured providers.
        ProviderRateLimitError: When all providers are rate-limited.
    """
    router = _get_router()
    try:
        return await router.acompletion(
            model="estimation-model",
            messages=messages,
        )
    except litellm.AuthenticationError as exc:
        raise ProviderAuthError() from exc
    except litellm.RateLimitError as exc:
        raise ProviderRateLimitError() from exc


async def stream_complete(
    messages: list[dict[str, str]],
) -> AsyncGenerator[Any, None]:
    """Send a streaming completion request through the LiteLLM Router.

    Args:
        messages: List of chat messages in OpenAI format.

    Yields:
        Raw LiteLLM stream chunks.

    Raises:
        ProviderAuthError: When authentication fails for all configured providers.
        ProviderRateLimitError: When all providers are rate-limited.
    """
    router = _get_router()
    try:
        stream = await router.acompletion(
            model="estimation-model",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},  # enables usage in the last chunk
        )
    except litellm.AuthenticationError as exc:
        raise ProviderAuthError() from exc
    except litellm.RateLimitError as exc:
        raise ProviderRateLimitError() from exc

    async for chunk in stream:
        yield chunk
