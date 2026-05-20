"""LLM service — builds prompts and delegates LLM calls to LLMWrapper.

This module is responsible for:
1. Building the system prompt with few-shot examples injected (CAG).
2. Delegating structured calls to LLMWrapper.complete_structured (Instructor + LiteLLM).
3. Delegating streaming calls to the LiteLLM Router (stream_complete).
"""

from typing import AsyncGenerator

import time

from src.core.config import get_settings
from src.core.exceptions import (
    ProviderAuthError,
    ProviderRateLimitError,
    UnknownProviderError,
)
from src.core.logging import logger
from src.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    UsageCost,
)
from src.services.llm_wrapper import get_llm_wrapper, get_router, stream_complete
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


async def generate_estimation(
    transcript: str,
    provider_override: str | None = None,
) -> dict:
    """Generate a structured software effort estimation from a meeting transcript.

    Delegates to LLMWrapper.complete_structured which uses
    instructor.from_litellm(litellm.completion) — no direct SDK lock-in.

    Args:
        transcript: Raw text of the meeting transcription.
        provider_override: 'openai' or 'anthropic'. Overrides settings.llm_provider.

    Returns:
        dict with keys: estimation_result, model, provider, usage (UsageCost).
    """
    settings = get_settings()

    # Resolve which model to use, honouring the per-request provider override.
    if provider_override == "anthropic":
        model = settings.anthropic_model
    elif provider_override == "openai":
        model = settings.openai_model
    elif provider_override is None:
        model = None  # LLMWrapper will use its configured primary_model
    else:
        raise UnknownProviderError(
            f"Unsupported provider: '{provider_override}'. Use 'openai' or 'anthropic'."
        )

    system_prompt = _build_system_prompt()
    wrapper = get_llm_wrapper()

    estimation_result, meta = await wrapper.complete_structured(
        system_prompt=system_prompt,
        user_message=f"Meeting transcript:\n{transcript}",
        response_model=EstimationResult,
        model_override=model,
    )

    cost_usd = calculate_cost(
        meta["model"], meta["input_tokens"], meta["output_tokens"]
    )

    return {
        "estimation_result": estimation_result,
        "model": meta["model"],
        "provider": meta["provider"],
        "usage": UsageCost(
            input_tokens=meta["input_tokens"],
            output_tokens=meta["output_tokens"],
            total_tokens=meta["input_tokens"] + meta["output_tokens"],
            cost_usd=cost_usd,
        ),
    }


async def stream_estimation(
    request: EstimationRequest,
    prompt_version: str | None = None,
) -> AsyncGenerator[str | EstimationResponse, None]:
    """Generate a software effort estimation using streaming.

    Yields:
        Text chunks as they arrive from the model.
        The final yielded item is an EstimationResponse with full metadata.
    """
    settings = get_settings()
    primary_model = settings.llm_models[0] if settings.llm_models else "unknown"

    system_prompt, user_message = render_estimation_prompt(
        request, version=prompt_version or get_settings().prompt_version, model=primary_model
    )

    call_logger = logger.bind(endpoint="/estimate/stream", mode="stream")
    call_logger.info("llm_call_started", models=get_router().model_list)
    start = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    full_text = ""
    model_used = ""
    first_model = ""
    input_tokens = 0
    output_tokens = 0
    finish_reason = None

    try:
        chunks = stream_complete(messages=messages)
    except (ProviderAuthError, ProviderRateLimitError) as exc:
        latency = round((time.time() - start) * 1000, 1)
        error_type = type(exc).__name__
        log_fn = (
            call_logger.error
            if isinstance(exc, ProviderAuthError)
            else call_logger.warning
        )
        log_fn("llm_call_failed", error_type=error_type, latency_ms=latency)
        raise

    async for chunk in chunks:
        # Cada chunk lleva el nombre del modelo
        if chunk.model:
            if not first_model:
                # Guardamos el modelo del primer chunk para detectar fallback.
                # Si a mitad del stream LiteLLM cambia de proveedor, model_used
                # será distinto de first_model al final.
                first_model = chunk.model
            model_used = chunk.model

        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_text += delta
            yield delta

        # finish_reason llega en el penúltimo chunk (antes del chunk de usage)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

        # El último chunk (con stream_options include_usage) trae el uso
        usage = getattr(chunk, "usage", None)
        if usage:
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0

    cost_usd = calculate_cost(model_used, input_tokens, output_tokens)
    latency = round((time.time() - start) * 1000, 1)

    # Si LiteLLM hizo fallback a otro modelo, lo registramos explícitamente.
    if first_model and model_used and first_model != model_used:
        call_logger.warning(
            "fallback_triggered",
            original_provider=first_model,
            fallback_provider=model_used,
        )

    call_logger.info(
        "llm_call_completed",
        model=model_used,
        tokens_in=input_tokens,
        tokens_out=output_tokens,
        latency_ms=latency,
        cost_usd=round(cost_usd, 6),
        finish_reason=finish_reason,
    )

    yield EstimationResponse(
        text=full_text,
        prompt_version=prompt_version,
        provider_used=model_used.split("/")[0] if "/" in model_used else model_used,
        model_used=model_used,
        usage=UsageCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
        ),
    )
