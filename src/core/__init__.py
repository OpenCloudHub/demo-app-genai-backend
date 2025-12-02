"""Core module - config, logging, tracing."""

from src.core.config import CONFIG
from src.core.logging import get_logger, log_section, setup_logging
from src.core.tracing import (
    OTEL_AVAILABLE,
    TRACING_ENABLED,
    get_tracer,
    setup_mlflow_autolog,
    setup_tracing,
    trace_if_enabled,
)

__all__ = [
    "CONFIG",
    "get_logger",
    "log_section",
    "setup_logging",
    "get_tracer",
    "setup_tracing",
    "trace_if_enabled",
    "setup_mlflow_autolog",
    "TRACING_ENABLED",
    "OTEL_AVAILABLE",
]
