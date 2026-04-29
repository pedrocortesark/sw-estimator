"""OpenAI implementation of BaseLLMProvider."""

from openai import AsyncOpenAI

from src.core.config import get_settings
from src.core.logging import logger
from src.services.base_llm import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Sends chat completion requests to the OpenAI API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def complete(self, system_prompt: str, user_message: str) -> tuple[str, str]:
        logger.debug(f"Sending request to OpenAI | model={self._model}")

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,  # Low temperature → more consistent, less creative estimations
        )

        text = response.choices[0].message.content or ""
        logger.debug(f"OpenAI response received | tokens={response.usage.total_tokens}")
        return text, self._model
