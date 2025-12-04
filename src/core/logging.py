# ==============================================================================
# Logging Configuration
# ==============================================================================
#
# Loguru-based logging with optional JSON formatting for production.
#
# IMPORTANT: This module intentionally avoids OpenTelemetry imports to prevent
# circular dependencies. OTEL log correlation is handled separately in tracing.py.
#
# Features:
#   - Colored console output for development
#   - JSON structured logging for production
#   - Name binding for log source identification
#   - Section headers for visual separation
#
# Usage:
#   from src.core.logging import get_logger, setup_logging, log_section
#
#   setup_logging(level="DEBUG", json_format=False)
#   logger = get_logger(__name__)
#   logger.info("Hello world")
#   log_section("Starting Pipeline", emoji="🚀")
#
# =============================================================================="""

import sys

from loguru import logger


def get_logger(name: str = __name__):
    """Get a configured logger instance with name binding."""
    return logger.bind(name=name)


def setup_logging(level: str = "INFO", json_format: bool = False):
    """Configure logging for the application."""
    logger.remove()

    if json_format:
        logger.add(
            sys.stdout,
            format="{message}",
            level=level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[name]}</cyan> | "
                "<level>{message}</level>"
            ),
            level=level,
            colorize=True,
        )

    return logger


def log_section(message: str, emoji: str = "📌", name: str = "main"):
    """Log a section header for visual separation."""
    separator = "=" * 60
    bound_logger = logger.bind(name=name)
    bound_logger.info(f"\n{separator}")
    bound_logger.info(f"{emoji} {message}")
    bound_logger.info(separator)


# Auto-setup with defaults on import
setup_logging()
