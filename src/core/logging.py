import logging

import structlog

from src.core.config import get_settings


def configure_logging() -> None:
    """Configura structlog como logger global de la aplicación.

    Se llama UNA sola vez al arrancar la app (en el lifespan de FastAPI).

    La configuración es dual:
    - development: ConsoleRenderer con colores, legible por humanos en terminal.
    - production:  JSONRenderer, para ingestión por herramientas como
                   Elasticsearch, Loki o CloudWatch.

    La clave de structlog es la cadena de procesadores (processors). Cada
    procesador recibe el diccionario del evento, lo enriquece o transforma,
    y lo pasa al siguiente. El último siempre es el renderer.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # --- Procesadores compartidos por todos los entornos ---
    # Se ejecutan en orden de izquierda a derecha antes del renderer final.
    shared_processors = [
        # Añade el campo "level" (INFO, WARNING, ERROR…) al diccionario del evento.
        structlog.processors.add_log_level,
        # Añade "timestamp" en formato ISO-8601 (ej. "2026-05-06T10:30:00Z").
        structlog.processors.TimeStamper(fmt="iso"),
        # Renombra el campo por defecto "event" a "msg" (convenio más legible).
        structlog.processors.EventRenamer("msg"),
    ]

    if settings.app_env == "production":
        # En producción: JSON puro, una línea por evento.
        # Las plataformas de observabilidad esperan este formato.
        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
        )
    else:
        # En desarrollo: salida con colores y formato tabular, fácil de leer
        # en la terminal durante el desarrollo local.
        structlog.configure(
            processors=shared_processors
            + [
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
        )

    # Usamos el logger ya configurado para confirmar el arranque.
    structlog.get_logger().info(
        "logging_configured",
        env=settings.app_env,
        level=settings.log_level.upper(),
    )


# logger es el objeto que importa el resto del código con:
#   from src.core.logging import logger
#
# Cada módulo llama a structlog.get_logger() y obtiene un logger que
# ya está configurado por configure_logging(). Al hacer .bind() sobre él
# se añaden campos contextuales que viajan en todos los logs de ese scope.
logger = structlog.get_logger()

__all__ = ["logger", "configure_logging"]
