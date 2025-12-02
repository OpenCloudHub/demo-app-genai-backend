"""OpenCloudHub RAG API - FastAPI application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.routes import admin, debug, health, query, session
from src.core import CONFIG, get_logger, setup_tracing
from src.core.tracing import get_current_trace_id
from src.rag import RAGChain
from src.schemas import APIStatus

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG chain on startup."""
    logger.info("🚀 Starting OpenCloudHub RAG API")

    # Initialize app state
    app.state.status = APIStatus.LOADING
    app.state.chain = None
    app.state.prompt_name = CONFIG.prompt_name
    app.state.top_k = CONFIG.db_top_k
    app.state.prompt_version = None
    app.state.start_time = datetime.now(timezone.utc)

    # Setup MLflow
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)

    try:
        # Load production prompt version
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
            logger.warning("Service will start unhealthy until prompt is available")
            app.state.status = APIStatus.UNHEALTHY
            yield
            return

        # Initialize RAG chain
        try:
            app.state.chain = RAGChain(
                db_connection_string=CONFIG.db_connection_string,
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
            logger.warning(
                "Service will start unhealthy. Use /admin/reload-prompt to retry."
            )
            app.state.status = APIStatus.UNHEALTHY

    except Exception as e:
        logger.error(f"⚠️ Unexpected error during startup: {e}", exc_info=True)
        app.state.status = APIStatus.UNHEALTHY

    yield

    logger.info("👋 Shutting down OpenCloudHub RAG API")
    app.state.chain = None


description = """
**Demo RAG System for OpenCloudHub MLOps Platform Documentation 🚀**

This demo Retrieval-Augmented Generation (RAG) system demonstrates modern MLOps
practices applied to Generative AI workloads.

## 🔍 Features

- **Chat History**: Maintain conversation context across multiple queries
- **Streaming**: Real-time token-by-token response generation
- **Debug Endpoints**: Inspect retrieval results and configuration
- **Hot Reload**: Update prompts without service restart
- **OpenTelemetry**: Distributed tracing support

## 🔄 Session Management

1. Create a session with `POST /session/create`
2. Use the `session_id` in subsequent queries
3. View history with `GET /session/{session_id}/history`
4. Clear with `DELETE /session/{session_id}`

Built with ❤️ by OpenCloudHub
"""

app = FastAPI(
    title="🤖 OpenCloudHub RAG API",
    description=description,
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api",
)

# Setup tracing
setup_tracing(
    app,
    service_name=CONFIG.otel_service_name,
    endpoint=CONFIG.otel_exporter_endpoint,
    enabled=CONFIG.otel_enabled,
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    trace_id = get_current_trace_id()
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {"error_type": type(exc).__name__, "message": str(exc)},
            "trace_id": trace_id,  # Helps debugging
        },
    )


# Register Routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(session.router)
app.include_router(debug.router)
app.include_router(admin.router)
