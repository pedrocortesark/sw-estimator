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
from src.services.base_llm import BaseLLMProvider, ProviderUsage


class AnthropicProvider(BaseLLMProvider):
    """Sends chat completion requests to the Anthropic API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.anthropic_model
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def complete(
        self, system_prompt: str, user_message: str
    ) -> tuple[str, str, ProviderUsage]:
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
        except anthropic.APIStatusError as exc:
            # Catch-all for any other HTTP error codes not explicitly handled above
            # (e.g. 409 Conflict). Re-raise as generic EstimatorError so the
            # fallback handler in exceptions.py returns a 500 with a safe message.
            raise ProviderInternalError(str(exc)) from exc

        usage = ProviderUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        logger.debug(
            f"Anthropic response received"
            f" | stop_reason={response.stop_reason}"
            f" | input_tokens={usage.input_tokens}"
            f" | output_tokens={usage.output_tokens}"
        )
        return response.content[0].text, self._model, usage

    async def stream_complete(
        self, system_prompt: str, user_message: str
    ) -> "AsyncGenerator[str | ProviderUsage | str, None]":
        from typing import AsyncGenerator
        logger.debug(f"Sending stream request to Anthropic | model={self._model}")

        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                
                final_message = await stream.get_final_message()
                
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
        except anthropic.APIStatusError as exc:
            raise ProviderInternalError(str(exc)) from exc

        usage = ProviderUsage(
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
        )
        logger.debug(
            f"Anthropic stream completed"
            f" | input_tokens={usage.input_tokens}"
            f" | output_tokens={usage.output_tokens}"
        )
        yield {"model": self._model, "usage": usage}

