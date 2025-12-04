# ==============================================================================
# Admin Endpoints - Runtime Configuration
# ==============================================================================
#
# Administrative operations for runtime configuration changes.
#
# Endpoints:
#   POST /api/admin/reload-prompt - Hot-reload prompt without restart
#
# Request Body:
#   {
#     "prompt_version": 3,  // Specific version, or null for @champion
#     "top_k": 5            // Number of documents to retrieve
#   }
#
# Use Case:
#   After promoting a new prompt version to @champion in MLflow,
#   call this endpoint to reload the running API without downtime.
#
# Note:
#   Detaches from OTEL trace context during reload to avoid interference
#   with MLflow API calls.
#
# =============================================================================="""

from datetime import datetime, timezone

import mlflow
import urllib3
from fastapi import APIRouter, HTTPException, Request, status

from src.core import CONFIG, get_logger
from src.rag import RAGChain
from src.schemas import APIStatus, ReloadPromptRequest, ReloadPromptResponse

logger = get_logger(__name__)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post(
    "/reload-prompt",
    response_model=ReloadPromptResponse,
    summary="Reload Prompt Template",
)
async def reload_prompt(body: ReloadPromptRequest, request: Request):
    """Reload the RAG chain with a different prompt version."""
    from opentelemetry import context as otel_context
    from opentelemetry.context import Context

    state = request.app.state
    state.status = APIStatus.LOADING
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)

    try:
        logger.info(
            f"🔄 Reloading RAG chain with prompt v{body.prompt_version or 'champion'}..."
        )

        # Detach from current trace context to avoid OTEL interfering with MLflow
        token = otel_context.attach(Context())
        try:
            state.chain = RAGChain(
                pg_engine=state.db.engine,
                db_connection_string_psycopg=CONFIG.db_connection_string_psycopg,
                table_name=CONFIG.db_table_name,
                embedding_model=CONFIG.embedding_model,
                llm_base_url=CONFIG.llm_base_url,
                llm_model=CONFIG.llm_model,
                api_key=CONFIG.api_key,
                prompt_name=state.prompt_name,
                prompt_version=body.prompt_version,
                top_k=body.top_k,
            )
        finally:
            otel_context.detach(token)

        state.prompt_version = state.chain.prompt_version
        state.top_k = body.top_k
        state.status = APIStatus.HEALTHY
        logger.success(f"✅ Reloaded with prompt v{state.prompt_version}")

        return ReloadPromptResponse(
            status="reloaded",
            prompt_name=state.prompt_name,
            prompt_version=state.prompt_version,
            top_k=body.top_k,
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as e:
        state.status = APIStatus.UNHEALTHY
        logger.error(f"❌ Failed to reload prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload prompt: {str(e)}",
        )
