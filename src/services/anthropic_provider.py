"""Anthropic implementation of BaseLLMProvider."""

import anthropic

from src.core.config import get_settings
from src.core.exceptions import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderInternalError,
    ProviderRateLimitError,
)
from src.core.logging import logger
from src.services.base_llm import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Sends chat completion requests to the Anthropic API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.anthropic_model
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def complete(self, system_prompt: str, user_message: str) -> tuple[str, str]:
        logger.debug(f"Sending request to Anthropic | model={self._model}")

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            raise ProviderBadRequestError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(str(exc)) from exc
        except anthropic.InternalServerError as exc:
            raise ProviderInternalError(str(exc)) from exc

        text = response.content[0].text
        logger.debug(
            f"Anthropic response received | stop_reason={response.stop_reason}"
        )
        return text, self._model
