"""OpenTelemetry tracing setup for FastAPI with log correlation."""

import os
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)

# Master switch for all tracing (OTEL + MLflow)
TRACING_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"

# Track if OpenTelemetry packages are available
OTEL_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    if TRACING_ENABLED:
        logger.warning("OpenTelemetry packages not installed, OTEL tracing disabled")


def trace_if_enabled(name: str):
    """
    Decorator that applies mlflow.trace only if tracing is enabled.

    Usage:
        @trace_if_enabled(name="my_function")
        def my_function():
            ...
    """

    def decorator(func):
        if not TRACING_ENABLED:
            return func

        try:
            import mlflow

            return mlflow.trace(name=name)(func)
        except Exception:
            return func

    return decorator


def setup_mlflow_autolog():
    """Enable MLflow LangChain autologging if tracing is enabled."""
    if not TRACING_ENABLED:
        logger.debug("MLflow autologging disabled (OTEL_ENABLED=false)")
        return False

    try:
        import mlflow

        mlflow.langchain.autolog()
        logger.info("✓ MLflow LangChain autologging enabled")
        return True
    except Exception as e:
        logger.warning(f"Failed to enable MLflow autologging: {e}")
        return False


def setup_tracing(
    app,
    service_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    enabled: bool = True,
    log_correlation: bool = True,
):
    """
    Setup OpenTelemetry tracing for FastAPI app.

    Args:
        app: FastAPI application instance
        service_name: Service name for traces
        endpoint: OTLP endpoint (e.g., tempo:4317)
        enabled: Whether tracing is enabled
        log_correlation: Add trace_id/span_id to logs (for Loki → Tempo linking)
    """
    if not enabled or not TRACING_ENABLED or not OTEL_AVAILABLE:
        logger.info("OpenTelemetry tracing disabled")
        return

    service = service_name or os.getenv("OTEL_SERVICE_NAME", "rag-backend")
    otlp_endpoint = endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"
    )

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: service,
                "service.namespace": "opencloudhub",
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

        if log_correlation:
            try:
                from opentelemetry.instrumentation.logging import LoggingInstrumentor

                LoggingInstrumentor().instrument(set_logging_format=True)
                logger.info("✓ Log correlation enabled")
            except ImportError:
                pass

        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        HTTPXClientInstrumentor().instrument()

        try:
            from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

            PsycopgInstrumentor().instrument()
        except ImportError:
            pass

        logger.info(f"✓ Tracing enabled: service={service}, endpoint={otlp_endpoint}")

    except Exception as e:
        logger.warning(f"Failed to setup tracing: {e}")


def get_tracer(name: str = __name__):
    """Get a tracer for manual span creation."""
    if OTEL_AVAILABLE and TRACING_ENABLED:
        from opentelemetry import trace

        return trace.get_tracer(name)
    return None


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID (returns None if tracing disabled)."""
    if not OTEL_AVAILABLE or not TRACING_ENABLED:
        return None

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return trace.format_trace_id(span.get_span_context().trace_id)
    except Exception:
        pass
    return None
