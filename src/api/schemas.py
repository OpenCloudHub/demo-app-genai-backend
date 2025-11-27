"""Schema definitions for the RAG API."""

from datetime import datetime
from enum import StrEnum, auto
from typing import Optional

from pydantic import BaseModel, Field


class APIStatus(StrEnum):
    """API status enumeration."""

    LOADING = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    NOT_READY = auto()


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask the RAG system",
        examples=["What is OpenCloudHub?", "How does ArgoCD work in the platform?"],
    )
    session_id: Optional[str] = Field(
        None, description="Optional session ID for chat history tracking"
    )
    stream: bool = Field(default=False, description="Enable streaming response")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: str = Field(..., description="Generated answer from the RAG system")
    prompt_name: str = Field(..., description="Name of the prompt template used")
    prompt_version: int = Field(..., description="Version of the prompt template used")
    session_id: Optional[str] = Field(
        None, description="Session ID for chat history tracking"
    )
    timestamp: datetime = Field(..., description="Response timestamp (UTC)")
    processing_time_ms: float = Field(
        ..., description="Time taken to process the request in milliseconds"
    )


class PromptInfo(BaseModel):
    """Current prompt information."""

    prompt_name: str = Field(..., description="Name of the prompt template")
    prompt_version: int = Field(..., description="Current prompt version")
    prompt_template: str = Field(..., description="The actual prompt template text")


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


class ReloadPromptRequest(BaseModel):
    """Request model for reloading prompt."""

    prompt_version: int | None = Field(
        None, description="Specific prompt version to load, None for latest", ge=1
    )
    top_k: int = Field(
        5,
        description="Number of top documents to retrieve for context",
        ge=1,
        le=10,
    )


class ReloadPromptResponse(BaseModel):
    """Response model for prompt reload."""

    status: str = Field(..., description="Reload status")
    prompt_name: str = Field(..., description="Name of the prompt template")
    prompt_version: int = Field(..., description="Loaded prompt version")
    top_k: int = Field(
        ..., description="Number of top documents to retrieve for context"
    )
    timestamp: datetime = Field(..., description="Reload timestamp (UTC)")


class ErrorDetail(BaseModel):
    """Error detail model."""

    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str | ErrorDetail = Field(..., description="Error details")
