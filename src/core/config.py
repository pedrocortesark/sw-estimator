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
        extra="ignore",  # ignora variables del .env que no estén aquí
    )

    # --- App ---
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    api_base_url: str = "http://localhost:8000"

    # --- LLM Router (LiteLLM) ---
    # Lista de modelos en orden de prioridad, formato "proveedor/modelo".
    # El Router intenta el primero; si falla, pasa al siguiente, y así sucesivamente.
    # Desde .env: LLM_MODELS=["anthropic/claude-opus-4-5","anthropic/claude-haiku-4-5-20251001","openai/gpt-4o-mini"]
    llm_models: list[str] = [
        "anthropic/claude-haiku-4-5-20251001",
        "openai/gpt-4o-mini",
    ]

    # --- API Keys ---
    # LiteLLM las lee automáticamente desde el entorno, pero las declaramos
    # aquí para que pydantic-settings valide que existen en el .env.
    openai_api_key: str = ""
    anthropic_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Retorna la instancia de Settings (singleton en memoria).

    @lru_cache garantiza que el .env se lee UNA sola vez en toda la vida
    de la aplicación, no en cada request. Es el patrón estándar para esto
    en FastAPI.
    """
    return Settings()
