"""Admin endpoints for runtime configuration."""

from datetime import datetime, timezone

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
    state = request.app.state
    state.status = APIStatus.LOADING

    try:
        logger.info(
            f"🔄 Reloading RAG chain with prompt v{body.prompt_version or 'champion'}..."
        )

        state.chain = RAGChain(
            db_connection_string=CONFIG.db_connection_string,
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

        # Get actual loaded version from chain (handles None -> champion resolution)
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
