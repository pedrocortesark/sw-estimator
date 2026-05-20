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

    # --- Prompt ---
    # Active Jinja2 prompt version.  Must match a directory under
    # src/prompts/estimation/ (e.g. "v1", "v2").
    # Override in .env to switch globally; the per-request ?prompt_version=
    # query param takes precedence when explicitly supplied.
    prompt_version: str = "v1"

    # --- LLM Provider ---
    # Which provider to use for Instructor-based structured calls.
    # Can be overridden per-request via the EstimationRequest.provider field.
    llm_provider: Literal["openai", "anthropic"] = "openai"

    # Model names used when calling the provider SDKs directly via Instructor.
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # --- LLM Router (LiteLLM) — kept for streaming fallback ---
    # Lista de modelos en orden de prioridad, formato "proveedor/modelo".
    llm_models: list[str] = [
        "anthropic/claude-haiku-4-5-20251001",
        "openai/gpt-4o-mini",
    ]

    # --- Cache ---
    # Set to a Redis Stack URL (e.g. redis://localhost:6379) to enable
    # semantic caching.  Leave empty to use the in-process memory cache only.
    redis_url: str = ""
    # Minimum cosine similarity to consider two transcripts a semantic cache hit.
    # Lower = more cache hits but higher risk of returning a wrong estimate.
    # 0.85 catches phrasing variants (e.g. "Rhino.Compute" vs "Compute");
    # raise toward 0.92 for stricter matching in production.
    semantic_cache_threshold: float = 0.85

    # --- API Keys ---
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
