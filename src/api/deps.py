"""Shared API dependencies."""

from fastapi import HTTPException, Request, status


def get_chain(request: Request):
    """Dependency to get RAG chain from app state."""
    if request.app.state.chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not loaded. Service may still be starting up.",
        )
    return request.app.state.chain
