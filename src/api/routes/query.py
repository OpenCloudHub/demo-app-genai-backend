# ==============================================================================
# Query Endpoint - Main RAG Interface
# ==============================================================================
#
# Handles question-answering requests with optional streaming support.
#
# Endpoints:
#   POST /api/query - Ask a question
#     - Non-streaming: Returns full response in JSON
#     - Streaming: Returns Server-Sent Events (SSE) with token-by-token output
#
# Request Body:
#   {
#     "question": "What is GitOps?",
#     "session_id": "optional-uuid",  // For chat history context
#     "stream": false                  // Enable SSE streaming
#   }
#
# Response (non-streaming):
#   {
#     "answer": "GitOps is...",
#     "session_id": "uuid",
#     "prompt_version": 3,
#     "processing_time_ms": 1234.5
#   }
#
# =============================================================================="""

import asyncio
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.api.deps import get_chain
from src.core import get_logger
from src.rag import RAGChain
from src.schemas import QueryRequest, QueryResponse

logger = get_logger(__name__)

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse, summary="Ask a Question")
async def query(
    request: QueryRequest,
    req: Request,
    chain: RAGChain = Depends(get_chain),
):
    """
    Ask a question about the OpenCloudHub MLOps platform.

    Optionally provide a session_id for chat history context.
    If no session_id is provided, a new session will be created.
    """
    state = req.app.state

    if state.prompt_version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prompt not loaded.",
        )

    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(
            f"Query: {request.question[:100]}... (session={session_id[:8]}, stream={request.stream})"
        )

        if request.stream:
            return await _stream_response(
                chain, request.question, session_id, start_time
            )

        # Non-streaming response
        answer = chain.invoke(request.question, session_id)
        processing_time_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=answer,
            prompt_name=state.prompt_name,
            prompt_version=state.prompt_version,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


async def _stream_response(
    chain: RAGChain, question: str, session_id: str, start_time: float
):
    """Handle streaming response."""

    async def generate():
        try:
            full_response = []

            def sync_stream():
                for chunk in chain.stream(question, session_id):
                    yield chunk

            loop = asyncio.get_event_loop()
            gen = sync_stream()

            while True:
                try:
                    chunk = await loop.run_in_executor(None, lambda: next(gen, None))
                    if chunk is None:
                        break
                    if chunk:
                        full_response.append(chunk)
                        yield f"data: {chunk}\n\n"
                except StopIteration:
                    break

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Stream complete: {len(full_response)} chunks in {processing_time:.0f}ms"
            )
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )
