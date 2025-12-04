# ==============================================================================
# OpenTelemetry Tracing & Prometheus Metrics
# ==============================================================================
#
# Observability setup for FastAPI based on fastapi-observability pattern.
# Reference: https://github.com/blueswen/fastapi-observability
#
# This module provides:
#   1. OpenTelemetry tracing → Tempo via OTLP gRPC
#   2. Prometheus metrics with trace exemplars (for Grafana correlation)
#   3. Automatic log correlation (trace_id/span_id injection)
#   4. Request/response metrics (latency, counts, in-progress)
#
# Metrics Exposed:
#   - fastapi_requests_total: Counter by method/path
#   - fastapi_responses_total: Counter by method/path/status
#   - fastapi_requests_duration_seconds: Histogram with TraceID exemplars
#   - fastapi_exceptions_total: Counter by exception type
#   - fastapi_requests_in_progress: Gauge of active requests
#
# Usage:
#   from src.core.tracing import setup_otlp, PrometheusMiddleware, metrics
#
#   setup_otlp(app, app_name="my-service", endpoint="tempo:4317")
#   app.add_middleware(PrometheusMiddleware, app_name="my-service")
#   app.get("/metrics")(lambda req: metrics(req))
#
# =============================================================================="""

import logging
import time
from typing import Tuple

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.openmetrics.exposition import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.types import ASGIApp

# =============================================================================
# Prometheus Metrics
# =============================================================================

APP_INFO = Gauge("fastapi_app_info", "FastAPI application information.", ["app_name"])

REQUESTS = Counter(
    "fastapi_requests_total",
    "Total count of requests by method and path.",
    ["method", "path", "app_name"],
)

RESPONSES = Counter(
    "fastapi_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path", "status_code", "app_name"],
)

REQUESTS_PROCESSING_TIME = Histogram(
    "fastapi_requests_duration_seconds",
    "Histogram of requests processing time by path (in seconds)",
    ["method", "path", "app_name"],
)

EXCEPTIONS = Counter(
    "fastapi_exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path", "exception_type", "app_name"],
)

REQUESTS_IN_PROGRESS = Gauge(
    "fastapi_requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method", "path", "app_name"],
)


# =============================================================================
# Prometheus Middleware
# =============================================================================


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics with trace exemplars."""

    def __init__(self, app: ASGIApp, app_name: str = "fastapi-app") -> None:
        super().__init__(app)
        self.app_name = app_name
        APP_INFO.labels(app_name=self.app_name).inc()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        method = request.method
        path, is_handled_path = self.get_path(request)

        if not is_handled_path:
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(
            method=method, path=path, app_name=self.app_name
        ).inc()
        REQUESTS.labels(method=method, path=path, app_name=self.app_name).inc()

        before_time = time.perf_counter()
        try:
            response = await call_next(request)
        except BaseException as e:
            status_code = HTTP_500_INTERNAL_SERVER_ERROR
            EXCEPTIONS.labels(
                method=method,
                path=path,
                exception_type=type(e).__name__,
                app_name=self.app_name,
            ).inc()
            raise e from None
        else:
            status_code = response.status_code
            after_time = time.perf_counter()

            # Retrieve trace id for exemplar
            span = trace.get_current_span()
            trace_id = trace.format_trace_id(span.get_span_context().trace_id)

            REQUESTS_PROCESSING_TIME.labels(
                method=method, path=path, app_name=self.app_name
            ).observe(after_time - before_time, exemplar={"TraceID": trace_id})
        finally:
            RESPONSES.labels(
                method=method,
                path=path,
                status_code=status_code,
                app_name=self.app_name,
            ).inc()
            REQUESTS_IN_PROGRESS.labels(
                method=method, path=path, app_name=self.app_name
            ).dec()

        return response

    @staticmethod
    def get_path(request: Request) -> Tuple[str, bool]:
        """Get the path template for the request."""
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                return route.path, True
        return request.url.path, False


def metrics(request: Request) -> Response:
    """Endpoint to expose Prometheus metrics with exemplars."""
    return Response(
        generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST}
    )


# =============================================================================
# OpenTelemetry Setup
# =============================================================================


def setup_otlp(
    app: ASGIApp,
    app_name: str,
    endpoint: str,
    log_correlation: bool = True,
) -> None:
    """
    Setup OpenTelemetry tracing for FastAPI.

    Based on fastapi-observability pattern.

    Args:
        app: FastAPI/ASGI application instance
        app_name: Service name for traces
        endpoint: OTLP gRPC endpoint (e.g., "tempo:4317" or "localhost:4317")
        log_correlation: Whether to inject trace_id into logs
    """
    # Set the service name to show in traces
    resource = Resource.create(attributes={"service.name": app_name})

    # Set the tracer provider
    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)

    # Add OTLP exporter
    tracer.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )

    # Instrument logging to add trace_id and span_id to logs
    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=True)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)


# =============================================================================
# Helper Functions
# =============================================================================


def get_current_trace_id() -> str | None:
    """Get current trace ID for error responses."""
    try:
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return trace.format_trace_id(span.get_span_context().trace_id)
    except Exception:
        pass
    return None


def get_current_span_id() -> str | None:
    """Get current span ID."""
    try:
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return trace.format_span_id(span.get_span_context().span_id)
    except Exception:
        pass
    return None


class EndpointFilter(logging.Filter):
    """Filter out /metrics endpoint from access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /metrics") == -1
