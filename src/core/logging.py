import sys

from loguru import logger

from src.core.config import get_settings


def setup_logging() -> None:
    """Configura loguru como logger global de la aplicación.

    Se llama UNA sola vez al arrancar la app (en el lifespan de FastAPI).
    Elimina el handler por defecto de loguru y añade uno configurado
    según el nivel definido en Settings.
    """
    settings = get_settings()

    # Eliminar el handler por defecto de loguru (ya viene con uno configurado)
    logger.remove()

    # Añadir nuestro handler: salida a stdout con el nivel configurado
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.info(f"Logging configurado. Nivel: {settings.log_level.upper()}")


# Re-exportamos logger para que el resto del código pueda hacer:
# from src.core.logging import logger
__all__ = ["logger", "setup_logging"]
