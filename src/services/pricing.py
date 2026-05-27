"""LLM model pricing table and cost calculation helpers.

Prices are expressed in USD per 1 million tokens and must be kept in sync
with the official provider pricing pages:
- OpenAI:    https://openai.com/api/pricing
- Anthropic: https://www.anthropic.com/pricing#anthropic-api

Update this file whenever a model's pricing changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Pricing entry for a single model."""

    input_per_million: float  # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens


# ---------------------------------------------------------------------------
# Pricing table  (prices in USD per 1M tokens — last updated 2025-04)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00),
    "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
    "o3": ModelPrice(input_per_million=10.00, output_per_million=40.00),
    "o4-mini": ModelPrice(input_per_million=1.10, output_per_million=4.40),
    # Anthropic
    "claude-opus-4-5": ModelPrice(input_per_million=15.00, output_per_million=75.00),
    "claude-sonnet-4-5": ModelPrice(input_per_million=3.00, output_per_million=15.00),
    "claude-haiku-4-5": ModelPrice(input_per_million=0.80, output_per_million=4.00),
    "claude-haiku-4-5-20251001": ModelPrice(
        input_per_million=1.00, output_per_million=5.00
    ),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the total cost in USD for a single API call.

    If the model is not in the pricing table the cost is returned as 0.0
    rather than raising an exception, so a missing entry never breaks the
    response. A warning is emitted via the logger so it can be fixed.

    Args:
        model:         The model identifier as returned by the provider.
        input_tokens:  Number of input tokens consumed.
        output_tokens: Number of output tokens generated.

    Returns:
        Total cost in USD as a float.
    """
    from src.core.logging import logger  # local import to avoid circular deps

    # Normalise LiteLLM-style "provider/model" names (e.g. "openai/gpt-4o-mini")
    # and versioned model names returned by the API (e.g. "gpt-4o-mini-2024-07-18")
    normalised = model.split("/")[-1] if "/" in model else model
    pricing = MODEL_PRICING.get(model) or MODEL_PRICING.get(normalised)
    if pricing is None:
        # Try prefix match: "gpt-4o-mini-2024-07-18" → "gpt-4o-mini"
        # Sort keys longest-first to avoid "gpt-4o" matching before "gpt-4o-mini"
        for key in sorted(MODEL_PRICING, key=len, reverse=True):
            if normalised.startswith(key):
                pricing = MODEL_PRICING[key]
                break
    if pricing is None:
        logger.warning(
            f"No pricing entry found for model '{model}'. Cost will be reported as 0.0."
        )
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return input_cost + output_cost
