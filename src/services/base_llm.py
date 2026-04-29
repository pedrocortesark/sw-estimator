"""Abstract base class defining the contract for all LLM providers.

Any provider (OpenAI, Anthropic, etc.) must inherit from BaseLLMProvider
and implement the `complete` method. This enforces a consistent interface
across providers and makes the aggregator provider-agnostic.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Interface that every LLM provider must implement."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_message: str) -> tuple[str, str]:
        """Send a chat completion request to the LLM.

        Args:
            system_prompt: Instructions and context injected as the system role.
                           This is where the CAG few-shot examples live.
            user_message:  The user's input — in our case, the meeting transcript.

        Returns:
            A tuple of (response_text, model_name_used).
            Returning the model name lets the API response report exactly
            which model generated the estimation.
        """
        ...
