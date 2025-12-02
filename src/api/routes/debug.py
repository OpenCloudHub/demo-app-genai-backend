"""Debug endpoints for inspecting RAG internals."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.deps import get_chain
from src.core import CONFIG, get_logger
from src.rag import RAGChain
from src.schemas import (
    DebugConfigResponse,
    DebugRetrievalRequest,
    DebugRetrievalResponse,
    RetrievedDocument,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/debug", tags=["Debug"])


@router.get("/config", response_model=DebugConfigResponse, summary="Get Configuration")
async def get_debug_config(
    request: Request,
    chain: RAGChain = Depends(get_chain),
):
    """Get current RAG configuration for debugging."""
    state = request.app.state

    try:
        prompt_template = chain.prompt_template.messages[0].prompt.template

        return DebugConfigResponse(
            prompt_name=state.prompt_name,
            prompt_version=state.prompt_version,
            prompt_template=prompt_template,
            embedding_model=CONFIG.embedding_model,
            llm_model=CONFIG.llm_model,
            top_k=state.top_k,
            table_name=CONFIG.db_table_name,
        )
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get config: {str(e)}",
        )


@router.post(
    "/retrieval", response_model=DebugRetrievalResponse, summary="Test Retrieval"
)
async def debug_retrieval(
    body: DebugRetrievalRequest,
    request: Request,
    chain: RAGChain = Depends(get_chain),
):
    """
    Test document retrieval without generating an answer.

    Shows exactly what documents would be retrieved for a given question
    and how they would be formatted as context for the LLM.
    """
    top_k = request.app.state.top_k

    start_time = time.time()

    try:
        docs = chain.retriever.invoke(body.question)[:top_k]

        retrieved_docs = []
        for i, doc in enumerate(docs, 1):
            retrieved_docs.append(
                RetrievedDocument(
                    rank=i,
                    content=doc.page_content,
                    source_repo=doc.metadata.get("source_repo"),
                    source_file=doc.metadata.get("source_file"),
                    section_h1=doc.metadata.get("section_h1"),
                    section_h2=doc.metadata.get("section_h2"),
                    section_h3=doc.metadata.get("section_h3"),
                    data_version=doc.metadata.get("data_version"),
                    similarity_score=doc.metadata.get("score"),
                )
            )

        formatted_context = chain._format_docs(docs)
        retrieval_time_ms = (time.time() - start_time) * 1000

        return DebugRetrievalResponse(
            question=body.question,
            top_k=top_k,
            documents=retrieved_docs,
            formatted_context=formatted_context,
            retrieval_time_ms=retrieval_time_ms,
        )

    except Exception as e:
        logger.error(f"Retrieval debug error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )
