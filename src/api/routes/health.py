"""Health and root endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status

from src.schemas import APIStatus, HealthResponse, RootResponse

router = APIRouter(tags=["Health"])


@router.get("/", response_model=RootResponse, summary="Root endpoint")
async def root(request: Request):
    """Root endpoint with basic service information."""
    return RootResponse(
        service="OpenCloudHub RAG API",
        version="1.0.0",
        status=request.app.state.status.value,
        docs="/docs",
        health="/health",
    )


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health(request: Request):
    """Health check endpoint."""
    state = request.app.state
    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()

    response = HealthResponse(
        status=state.status,
        chain_loaded=state.chain is not None,
        prompt_name=state.prompt_name,
        prompt_version=state.prompt_version,
        uptime_seconds=int(uptime),
    )

    if state.status != APIStatus.HEALTHY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response
