"""Query-related schema definitions."""

from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask the RAG system",
        examples=["What is OpenCloudHub?", "How does ArgoCD work in the platform?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for chat history. Omit or set to null for new session.",
        examples=[None, "550e8400-e29b-41d4-a716-446655440000"],
        json_schema_extra={"nullable": True},
    )
    stream: bool = Field(default=False, description="Enable streaming response")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: str = Field(..., description="Generated answer from the RAG system")
    prompt_name: str = Field(..., description="Name of the prompt template used")
    prompt_version: int = Field(..., description="Version of the prompt template used")
    session_id: str = Field(..., description="Session ID for chat history tracking")
    timestamp: datetime = Field(..., description="Response timestamp (UTC)")
    processing_time_ms: float = Field(
        ..., description="Time taken to process the request in milliseconds"
    )
