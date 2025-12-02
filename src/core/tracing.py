"""OpenTelemetry tracing setup for FastAPI."""

from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


def setup_tracing(
    app,
    service_name: str,
    endpoint: str,
    enabled: bool = True,
) -> bool:
    """
    Setup OpenTelemetry tracing for FastAPI app.

    Returns True if tracing was successfully enabled.
    """
    if not enabled:
        logger.info("⏭️ Tracing disabled via config")
        return False

    # Check if packages are available
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        logger.error(f"❌ OpenTelemetry packages not installed: {e}")
        return False

    try:
        # Create resource with service info
        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                "service.namespace": "opencloudhub",
            }
        )

        # Setup provider
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Setup exporter
        logger.info(f"🔗 Connecting to OTLP endpoint: {endpoint}")
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("✓ FastAPI instrumented")

        # Instrument HTTPX (for outgoing HTTP calls)
        HTTPXClientInstrumentor().instrument()
        logger.info("✓ HTTPX instrumented")

        # Instrument psycopg (database)
        try:
            from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

            PsycopgInstrumentor().instrument()
            logger.info("✓ Psycopg instrumented")
        except ImportError:
            logger.debug("Psycopg instrumentation not available")

        logger.info(f"✅ Tracing enabled: service={service_name}, endpoint={endpoint}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to setup tracing: {e}", exc_info=True)
        return False


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID for error responses."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return trace.format_trace_id(span.get_span_context().trace_id)
    except Exception:
        pass
    return None
