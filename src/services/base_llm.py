"""Abstract base class defining the contract for all LLM providers.

Any provider (OpenAI, Anthropic, etc.) must inherit from BaseLLMProvider
and implement the `complete` method. This enforces a consistent interface
across providers and makes the aggregator provider-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderUsage:
    """Token usage reported by the provider for a single API call."""

    input_tokens: int
    output_tokens: int


class BaseLLMProvider(ABC):
    """Interface that every LLM provider must implement."""

    @abstractmethod
    async def complete(
        self, system_prompt: str, user_message: str
    ) -> tuple[str, str, ProviderUsage]:
        """Send a chat completion request to the LLM.

        Args:
            system_prompt: Instructions and context injected as the system role.
                           This is where the CAG few-shot examples live.
            user_message:  The user's input — in our case, the meeting transcript.

        Returns:
            A tuple of (response_text, model_name_used, usage).
            - response_text: the generated text.
            - model_name_used: lets the API response report exactly which model ran.
            - usage: input/output token counts for cost calculation.
        """
        ...
