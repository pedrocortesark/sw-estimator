"""LLM aggregator service — the core of the CAG architecture.

This module is responsible for:
1. Selecting the correct LLM provider (OpenAI or Anthropic) based on config
   or a per-request override.
2. Building the system prompt with the static few-shot examples injected
   (Context-Augmented Generation).
3. Delegating the completion to the chosen provider and returning a
   structured EstimationResponse.
"""

import anthropic
import openai

from src.context.examples import ESTIMATION_EXAMPLES
from src.core.config import get_settings
from src.core.exceptions import (
    ProviderAuthError,
    ProviderRateLimitError,
    UnknownProviderError,
)
from src.core.logging import logger
from src.schemas.estimation import EstimationResponse, UsageCost
from src.services.anthropic_provider import AnthropicProvider
from src.services.base_llm import BaseLLMProvider
from src.services.openai_provider import OpenAIProvider
from src.services.pricing import calculate_cost


def _build_system_prompt() -> str:
    """Build the system prompt with few-shot examples injected.

    This is the CAG step: all reference context travels in every request.
    The LLM receives both the instructions AND the examples in the system role,
    so it can learn the expected output format and level of detail.
    """
    examples_text = ""
    for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        examples_text += f"""
                        --- Example {i} ---
                        Meeting transcript:
                        {example["meeting_summary"]}

                        Estimation:
                        {example["estimation"]}
                        """

    return f"""You are an expert software estimation consultant with deep experience across \
            web development, cloud infrastructure, integrations, and computational design projects.

            Your task is to analyse a meeting transcript and produce a professional, structured \
            software effort estimation in Markdown format.

            Follow these rules strictly:
            - Break down the estimation by functional module or technical area.
            - For each module, provide a table with tasks, estimated hours, and required profiles.
            - Include a final summary table with total hours per module and a grand total.
            - List the required team profiles and their total hours.
            - Include a "Main Risks" section with at least 2 risks.
            - Provide an estimated range (e.g. 110–125 hours) accounting for uncertainty.
            - Be specific and technical. Avoid vague statements.
            - Write entirely in English.

            Here are examples of previous estimations to use as reference for format and depth:
            {examples_text}
            Now produce the estimation for the new transcript provided by the user."""


def _get_provider(provider_name: str) -> BaseLLMProvider:
    """Instantiate and return the correct provider.

    Args:
        provider_name: 'openai' or 'anthropic'

    Raises:
        ValueError: if an unsupported provider name is given.
    """
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "anthropic":
        return AnthropicProvider()
    raise UnknownProviderError(
        f"Unsupported LLM provider: '{provider_name}'. "
        "Valid options are: 'openai', 'anthropic'."
    )


async def generate_estimation(
    transcript: str,
    provider_override: str | None = None,
) -> EstimationResponse:
    """Generate a software effort estimation from a meeting transcript.

    Args:
        transcript:        Raw text of the meeting transcription.
        provider_override: Optional provider name to use instead of the one
                           configured in Settings. Useful for A/B testing.

    Returns:
        EstimationResponse with the generated estimation, provider, and model used.
    """
    settings = get_settings()
    provider_name = provider_override or settings.llm_provider

    logger.info(f"Generating estimation | provider={provider_name}")

    provider = _get_provider(provider_name)
    system_prompt = _build_system_prompt()

    try:
        estimation_text, model_used, provider_usage = await provider.complete(
            system_prompt=system_prompt,
            user_message=f"Meeting transcript:\n{transcript}",
        )
    except (openai.RateLimitError, anthropic.RateLimitError) as exc:
        logger.warning(f"Rate limit hit | provider={provider_name}")
        raise ProviderRateLimitError() from exc
    except (openai.AuthenticationError, anthropic.AuthenticationStatusError) as exc:
        logger.error(f"Authentication failed | provider={provider_name}")
        raise ProviderAuthError() from exc

    cost_usd = calculate_cost(
        model_used, provider_usage.input_tokens, provider_usage.output_tokens
    )
    logger.info(
        f"Estimation generated successfully"
        f" | provider={provider_name}"
        f" | model={model_used}"
        f" | input_tokens={provider_usage.input_tokens}"
        f" | output_tokens={provider_usage.output_tokens}"
        f" | cost_usd={cost_usd:.6f}"
    )

    return EstimationResponse(
        estimation=estimation_text,
        provider_used=provider_name,
        model_used=model_used,
        usage=UsageCost(
            input_tokens=provider_usage.input_tokens,
            output_tokens=provider_usage.output_tokens,
            total_tokens=provider_usage.input_tokens + provider_usage.output_tokens,
            cost_usd=cost_usd,
        ),
    )
