from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import get_settings
from src.core.logging import logger, setup_logging
from src.routers import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación.

    El código ANTES del `yield` se ejecuta al arrancar el servidor.
    El código DESPUÉS del `yield` se ejecuta al apagarlo.

    Es el lugar correcto para inicializar recursos (logger, clientes HTTP,
    conexiones a BD...) y limpiarlos al cerrar.
    """
    # --- Startup ---
    setup_logging()
    settings = get_settings()
    logger.info(
        f"Arrancando sw-estimator | env={settings.app_env} | provider={settings.llm_provider}"
    )

    yield  # <-- aquí FastAPI atiende requests

    # --- Shutdown ---
    logger.info("Apagando sw-estimator...")


def create_app() -> FastAPI:
    """Factory function que crea y configura la instancia de FastAPI.

    Usar una factory en lugar de una variable global facilita mucho los tests:
    cada test puede crear su propia instancia limpia de la app.
    """
    settings = get_settings()

    app = FastAPI(
        title="SW Estimator",
        description="Estima el esfuerzo de software a partir de transcripciones de reuniones usando LLMs (CAG).",
        version="0.1.0",
        lifespan=lifespan,
        # En producción ocultamos los docs de la API
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
    )

    app.include_router(health.router)
    # El router de estimaciones se añadirá en la Fase 7:
    # app.include_router(estimation_router, prefix="/api/v1")

    return app


# Instancia global que uvicorn usa para arrancar el servidor:
# uvicorn src.main:app
app = create_app()
