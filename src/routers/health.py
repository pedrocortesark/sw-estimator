from fastapi import APIRouter

from src.core.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Endpoint de healthcheck.

    Los sistemas de orquestación (Docker, Kubernetes, load balancers)
    llaman a este endpoint periódicamente para saber si la app está viva.
    Debe ser rápido y no depender de servicios externos.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "env": settings.app_env,
        "llm_provider": settings.llm_provider,
    }
