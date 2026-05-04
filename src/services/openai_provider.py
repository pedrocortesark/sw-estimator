"""OpenAI implementation of BaseLLMProvider."""

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

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


class OpenAIProvider(BaseLLMProvider):
    """Sends requests to the OpenAI Responses API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def complete(
        self, system_prompt: str, user_message: str
    ) -> tuple[str, str, ProviderUsage]:
        logger.debug(f"Sending request to OpenAI | model={self._model}")

        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=user_message,
                temperature=0.2,  # Low temperature → more consistent, less creative estimations
                store=False,  # Do not store transcripts on OpenAI servers
            )
        except AuthenticationError as exc:
            raise ProviderAuthError(str(exc)) from exc
        except RateLimitError as exc:
            raise ProviderRateLimitError(str(exc)) from exc
        except BadRequestError as exc:
            raise ProviderBadRequestError(str(exc)) from exc
        except APIConnectionError as exc:
            raise ProviderConnectionError(str(exc)) from exc
        except InternalServerError as exc:
            raise ProviderInternalError(str(exc)) from exc

        usage = ProviderUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        logger.debug(
            f"OpenAI response received"
            f" | input_tokens={usage.input_tokens}"
            f" | output_tokens={usage.output_tokens}"
        )
        return response.output_text, self._model, usage

    async def stream_complete(
        self, system_prompt: str, user_message: str
    ) -> "AsyncGenerator[str | ProviderUsage | str, None]":
        from typing import AsyncGenerator
        logger.debug(f"Sending stream request to OpenAI | model={self._model}")

        try:
            stream = await self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=user_message,
                temperature=0.2,
                store=False,
                stream=True,
            )
            
            final_usage = None
            async for event in stream:
                if event.type == 'response.output_text.delta':
                    yield event.delta
                elif event.type == 'response.completed':
                    final_usage = event.response.usage
                    
        except AuthenticationError as exc:
            raise ProviderAuthError(str(exc)) from exc
        except RateLimitError as exc:
            raise ProviderRateLimitError(str(exc)) from exc
        except BadRequestError as exc:
            raise ProviderBadRequestError(str(exc)) from exc
        except APIConnectionError as exc:
            raise ProviderConnectionError(str(exc)) from exc
        except InternalServerError as exc:
            raise ProviderInternalError(str(exc)) from exc

        usage = ProviderUsage(
            input_tokens=final_usage.input_tokens if final_usage else 0,
            output_tokens=final_usage.output_tokens if final_usage else 0,
        )
        logger.debug(
            f"OpenAI stream completed"
            f" | input_tokens={usage.input_tokens}"
            f" | output_tokens={usage.output_tokens}"
        )
        yield {"model": self._model, "usage": usage}

