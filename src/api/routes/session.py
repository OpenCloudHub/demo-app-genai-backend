"""Session management endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import get_chain
from src.core import get_logger
from src.rag import RAGChain
from src.schemas import (
    SessionClearResponse,
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionMessage,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/session", tags=["Session"])


@router.post("/create", response_model=SessionCreateResponse, summary="Create Session")
async def create_session(chain: RAGChain = Depends(get_chain)):
    """Create a new chat session."""
    session_id = chain.create_session()

    return SessionCreateResponse(
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get Session History",
)
async def get_session_history(
    session_id: str,
    chain: RAGChain = Depends(get_chain),
):
    """Get the chat history for a session."""
    try:
        messages = chain.get_session_history(session_id)

        formatted_messages = []
        for msg in messages:
            role = "human" if msg.type == "human" else "ai"
            formatted_messages.append(SessionMessage(role=role, content=msg.content))

        return SessionHistoryResponse(
            session_id=session_id,
            messages=formatted_messages,
            message_count=len(formatted_messages),
        )
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session history: {str(e)}",
        )


@router.delete(
    "/{session_id}", response_model=SessionClearResponse, summary="Clear Session"
)
async def clear_session(
    session_id: str,
    chain: RAGChain = Depends(get_chain),
):
    """Clear the chat history for a session."""
    try:
        chain.clear_session(session_id)

        return SessionClearResponse(
            session_id=session_id,
            status="cleared",
        )
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session: {str(e)}",
        )
