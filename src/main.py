# ==============================================================================
# OpenCloudHub RAG API - FastAPI Application Entry Point
# ==============================================================================
#
# RAG API with streaming responses, session management,
# and OpenTelemetry observability.
#
# This module:
#   1. Initializes the FastAPI application with lifespan management
#   2. Sets up OpenTelemetry tracing to Tempo via OTLP gRPC
#   3. Configures Prometheus metrics middleware
#   4. Loads RAGChain with prompt from MLflow registry (@champion alias)
#   5. Registers all API routes (query, session, health, admin, debug)
#
# Usage:
#   # Development
#   fastapi dev src/main.py
#
#   # Production
#   uvicorn src.main:app --host 0.0.0.0 --port 8000
#
# Environment:
#   Requires: DB_*, MLFLOW_TRACKING_URI, LLM_*, EMBEDDING_MODEL
#   Optional: OTEL_* for tracing configuration
#
# =============================================================================="""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import admin, debug, health, query, session
from src.core import CONFIG, get_logger, setup_logging
from src.core.database import DatabaseManager  # Add this
from src.core.tracing import (
    PrometheusMiddleware,
    get_current_trace_id,
    metrics,
    setup_otlp,
)
from src.rag import RAGChain
from src.schemas import APIStatus

setup_logging(level=CONFIG.log_level, json_format=CONFIG.log_json)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG chain on startup."""
    logger.info("🚀 Starting OpenCloudHub RAG API")
    app.state.status = APIStatus.LOADING
    app.state.chain = None
    app.state.db = None
    app.state.prompt_name = CONFIG.prompt_name
    app.state.top_k = CONFIG.db_top_k
    app.state.prompt_version = None
    app.state.start_time = datetime.now(timezone.utc)

    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)

    try:
        # Load prompt version
        try:
            prod_prompt = mlflow.genai.load_prompt(
                f"prompts:/{app.state.prompt_name}@champion"
            )
            app.state.prompt_version = prod_prompt.version
            logger.info(
                f"ℹ️ Loaded production prompt version: v{app.state.prompt_version}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to load production prompt version: {e}")
            app.state.status = APIStatus.UNHEALTHY
            yield
            return

        # Initialize database (singleton for serving)
        app.state.db = DatabaseManager.get_instance(CONFIG.db_connection_string)

        # Initialize RAG chain with injected engine
        app.state.chain = RAGChain(
            pg_engine=app.state.db.engine,
            db_connection_string_psycopg=CONFIG.db_connection_string_psycopg,
            table_name=CONFIG.db_table_name,
            embedding_model=CONFIG.embedding_model,
            llm_base_url=CONFIG.llm_base_url,
            llm_model=CONFIG.llm_model,
            api_key=CONFIG.api_key,
            prompt_name=app.state.prompt_name,
            prompt_version=app.state.prompt_version,
            top_k=app.state.top_k,
        )
        app.state.status = APIStatus.HEALTHY
        logger.success("✅ RAG chain loaded successfully")

    except Exception as e:
        logger.error(f"⚠️ Failed to initialize RAG chain: {e}", exc_info=True)
        app.state.status = APIStatus.UNHEALTHY

    yield

    # Cleanup on shutdown
    logger.info("👋 Shutting down OpenCloudHub RAG API")
    DatabaseManager.reset_instance()
    app.state.chain = None


app = FastAPI(
    title="🤖 OpenCloudHub RAG API",
    description="Demo RAG System for OpenCloudHub MLOps Platform",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api",
)

# Setup OTEL tracing → Tempo
setup_otlp(
    app=app,
    app_name=CONFIG.otel_service_name,
    endpoint=CONFIG.otel_exporter_otlp_endpoint,
    log_correlation=True,
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware, app_name=CONFIG.otel_service_name)


@app.get("/metrics")
async def get_metrics(request: Request):
    return metrics(request)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    trace_id = get_current_trace_id()
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {"error_type": type(exc).__name__, "message": str(exc)},
            "trace_id": trace_id,
        },
    )


app.include_router(health.router)
app.include_router(query.router)
app.include_router(session.router)
app.include_router(debug.router)
app.include_router(admin.router)
