"""Logging configuration with structured output."""

import sys

from loguru import logger


def get_logger(name: str = __name__):
    """Get a configured logger instance."""
    return logger.bind(name=name)


def setup_logging(level: str = "INFO", json_format: bool = False):
    """Configure logging for the application."""
    logger.remove()

    if json_format:
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
            level=level,
            colorize=True,
        )

    return logger


def log_section(message: str, emoji: str = "📌"):
    """Log a section header for visual separation."""
    separator = "=" * 60
    logger.info(f"\n{separator}")
    logger.info(f"{emoji} {message}")
    logger.info(separator)
