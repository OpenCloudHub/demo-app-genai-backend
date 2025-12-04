"""Core module exports - NO tracing imports here."""

from src.core.config import CONFIG
from src.core.logging import get_logger, log_section, setup_logging

__all__ = [
    "CONFIG",
    "get_logger",
    "setup_logging",
    "log_section",
]
