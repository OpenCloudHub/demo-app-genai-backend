"""Health and system schema definitions."""

from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.common import APIStatus


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: APIStatus = Field(..., description="API health status")
    chain_loaded: bool = Field(..., description="Whether the RAG chain is loaded")
    prompt_name: Optional[str] = Field(None, description="Current prompt name")
    prompt_version: Optional[int] = Field(None, description="Current prompt version")
    uptime_seconds: Optional[int] = Field(None, description="Service uptime in seconds")


class RootResponse(BaseModel):
    """Response model for root endpoint."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    docs: str = Field(..., description="URL to API documentation")
    health: str = Field(..., description="URL to health check endpoint")
