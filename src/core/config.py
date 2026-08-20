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
    prompt_version: str = "v2"

    # --- Conversation ---
    # Maximum number of user/assistant turn *pairs* to keep in the sliding
    # window.  The system prompt is always preserved and never counts toward
    # this limit.  Override in .env with MAX_CONVERSATION_TURNS=<n>.
    max_conversation_turns: int = 6

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

    # --- Persistence (Session 8 — pgvector) ---
    # Canonical URL for the async engine. The RAG store is on the real-time
    # request path, so it uses asyncpg; Alembic derives the sync psycopg URL
    # in ``alembic/env.py`` from this same setting.
    database_url: str = "postgresql+asyncpg://estimator:estimator@localhost:5432/estimator"

    # --- Session 10 — cross-encoder reranking ---
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # --- Session 12 — agent search configuration ---
    agent_search_top_k: int = 5
    agent_search_distance_threshold: float = 0.7

    # --- Session 13 — graph models and configuration ---
    # Models for the multi-agent graph nodes
    graph_classifier_model: str = "gpt-4o-mini"
    graph_analysis_model: str = "gpt-4o"
    graph_proposal_model: str = "gpt-4o"
    graph_extraction_model: str = "gpt-4o-mini"
    graph_generation_model: str = "gpt-4o"
    
    # Graph feature flags
    graph_proposal_enabled: bool = True
    graph_personas_enabled: bool = True
    
    # Classifier complexity → structure agent reasoning effort mapping
    graph_structure_effort_by_complexity: dict[str, str] = {
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    
    # Logfire configuration
    logfire_service_name: str = "sw-estimator"

    # --- Session 15 — inter-service authentication ----------------------------
    # Shared secret for service-to-service calls (e.g. business-backend -> AI service).
    # When set, endpoints require X-Service-Token header. Empty = auth disabled (dev).
    service_token: str = ""

    # --- Session 14 — multi-agent human-in-the-loop ---
    confidence_threshold: float = 0.7
    supervisor_router_model: str = "gpt-4o-mini"
    supervisor_max_steps: int = 8
    supervisor_confidence_threshold: float = 0.6
    supervisor_min_grounded_ratio: float = 0.5
    supervisor_privilege_strict: bool = False
    supervisor_audit_args_preview_chars: int = 200
    supervisor_competition_enabled: bool = False
    supervisor_divergence_penalty: float = 0.4
    supervisor_persistence_enabled: bool = False

    # --- Session 9/10 — generation and retrieval settings ---
    # Embedding model for vector search
    embedding_model: str = "text-embedding-3-small"
    # Available models for LLM calls
    available_models: list[str] = [
        "gpt-4o-mini",
        "claude-haiku-4-5-20251001",
    ]
    # Generation settings
    generation_model: str = "gpt-4o-mini"
    generation_max_tokens: int = 4096
    generation_reasoning_effort: str = "medium"
    # Reformulation model for query understanding
    reformulation_model: str = "gpt-4o-mini"
    # Retrieval settings
    retrieval_top_k: int = 10
    retrieval_distance_threshold: float = 0.6
    # Context assembly
    max_context_tokens: int = 8000
    # PII/Presidio
    presidio_spacy_model: str = "es_core_news_md"


@lru_cache
def get_settings() -> Settings:
    """Retorna la instancia de Settings (singleton en memoria).

    @lru_cache garantiza que el .env se lee UNA sola vez en toda la vida
    de la aplicación, no en cada request. Es el patrón estándar para esto
    en FastAPI.
    """
    return Settings()
