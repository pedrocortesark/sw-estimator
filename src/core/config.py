from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración central de la aplicación.

    pydantic-settings lee automáticamente las variables del archivo .env
    y las valida. Si falta una variable sin valor por defecto, la app
    falla al arrancar (fail-fast), lo cual es el comportamiento correcto.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # OPENAI_API_KEY == openai_api_key
        extra="ignore",        # ignora variables del .env que no estén aquí
    )

    # --- App ---
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # --- LLM Aggregator ---
    llm_provider: Literal["openai", "anthropic"] = "openai"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Anthropic ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"


@lru_cache
def get_settings() -> Settings:
    """Retorna la instancia de Settings (singleton en memoria).

    @lru_cache garantiza que el .env se lee UNA sola vez en toda la vida
    de la aplicación, no en cada request. Es el patrón estándar para esto
    en FastAPI.
    """
    return Settings()
