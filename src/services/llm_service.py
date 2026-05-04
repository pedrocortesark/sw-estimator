"""LLM aggregator service — the core of the CAG architecture.

This module is responsible for:
1. Building the system prompt with the static few-shot examples injected
   (Context-Augmented Generation).
2. Routing requests through LiteLLM Router with automatic fallback.
3. Returning a structured EstimationResponse.

LiteLLM Router handles provider selection, retries, and fallback automatically.
Primary model is tried first; if it fails, the fallback model takes over.
"""

from functools import lru_cache
from typing import AsyncGenerator

import litellm
from litellm import Router

from src.context.examples import ESTIMATION_EXAMPLES
from src.core.config import get_settings
from src.core.exceptions import ProviderAuthError, ProviderRateLimitError
from src.core.logging import logger
from src.schemas.estimation import EstimationResponse, UsageCost
from src.services.pricing import calculate_cost

# Silencia los logs verbosos de LiteLLM en producción
litellm.suppress_debug_info = True


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

    return f"""You are a senior software estimation consultant with 15+ years of experience \
delivering accurate effort and cost estimates across web development, cloud infrastructure, \
third-party integrations, and computational design projects. \
You have a strong track record of producing estimates that match actual delivery within a 15% margin.

Your task is to analyse a meeting transcript and produce a professional, structured software \
effort estimation in Markdown format.

The following section contains real estimation examples from previous projects at this company. \
Use them to calibrate your response in three specific ways:
- Hourly rates and cost structure must be consistent with these examples.
- Task granularity and breakdown depth should match the level of detail shown.
- Output format and section structure must follow the same pattern.

{examples_text}

Now produce the estimation for the new transcript. Your response must include:
1. A 2–3 sentence project summary.
2. A breakdown by functional module or technical area. For each module, provide a table \
with tasks, estimated hours, required profiles, and cost.
3. A final summary table with total hours and cost per module, plus a grand total.
4. The recommended team composition with total hours per profile.
5. A "Main Risks & Assumptions" section with at least 3 items.
6. An estimated delivery range in weeks and an hour range (e.g. 110–125 h) accounting for uncertainty.

Additional rules:
- Use EUR as currency.
- Round hours to multiples of 4.
- Be specific and technical. Avoid vague statements.
- Write entirely in English.

Meeting transcript provided by the user:"""


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
                "model_name": "estimation-model",  # mismo alias = cascada automática
                "litellm_params": {"model": model},
            }
            for model in settings.llm_models
        ],
        num_retries=1,
        timeout=60,
    )


async def generate_estimation(transcript: str) -> EstimationResponse:
    """Generate a software effort estimation from a meeting transcript.

    Args:
        transcript: Raw text of the meeting transcription.

    Returns:
        EstimationResponse with the generated estimation, model used, and usage cost.
    """
    router = _get_router()
    system_prompt = _build_system_prompt()

    logger.info("Generating estimation via LiteLLM Router")

    try:
        response = await router.acompletion(
            model="estimation-model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Meeting transcript:\n{transcript}"},
            ],
        )
    except litellm.AuthenticationError as exc:
        logger.error("Authentication failed")
        raise ProviderAuthError() from exc
    except litellm.RateLimitError as exc:
        logger.warning("Rate limit hit")
        raise ProviderRateLimitError() from exc

    model_used = response.model
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_usd = calculate_cost(model_used, input_tokens, output_tokens)

    logger.info(
        f"Estimation generated | model={model_used}"
        f" | input_tokens={input_tokens} | output_tokens={output_tokens}"
        f" | cost_usd={cost_usd:.6f}"
    )

    return EstimationResponse(
        estimation=response.choices[0].message.content,
        provider_used=model_used.split("/")[0] if "/" in model_used else model_used,
        model_used=model_used,
        usage=UsageCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
        ),
    )


async def stream_estimation(
    transcript: str,
) -> AsyncGenerator[str | EstimationResponse, None]:
    """Generate a software effort estimation using streaming.

    Yields:
        Text chunks as they arrive from the model.
        The final yielded item is an EstimationResponse with full metadata.
    """
    router = _get_router()
    system_prompt = _build_system_prompt()

    logger.info("Starting estimation stream via LiteLLM Router")

    try:
        stream = await router.acompletion(
            model="estimation-model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Meeting transcript:\n{transcript}"},
            ],
            stream=True,
            stream_options={"include_usage": True},  # activa uso en el último chunk
        )
    except litellm.AuthenticationError as exc:
        logger.error("Authentication failed during stream")
        raise ProviderAuthError() from exc
    except litellm.RateLimitError as exc:
        logger.warning("Rate limit hit during stream")
        raise ProviderRateLimitError() from exc

    full_text = ""
    model_used = ""
    input_tokens = 0
    output_tokens = 0

    async for chunk in stream:
        # Cada chunk lleva el nombre del modelo
        if chunk.model:
            model_used = chunk.model

        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_text += delta
            yield delta

        # El último chunk (con stream_options include_usage) trae el uso
        usage = getattr(chunk, "usage", None)
        if usage:
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0

    cost_usd = calculate_cost(model_used, input_tokens, output_tokens)
    logger.info(
        f"Estimation stream finished | model={model_used}"
        f" | input_tokens={input_tokens} | output_tokens={output_tokens}"
        f" | cost_usd={cost_usd:.6f}"
    )

    yield EstimationResponse(
        estimation=full_text,
        provider_used=model_used.split("/")[0] if "/" in model_used else model_used,
        model_used=model_used,
        usage=UsageCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
        ),
    )
