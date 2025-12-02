"""Admin-related schema definitions."""

from datetime import datetime

from pydantic import BaseModel, Field


class ReloadPromptRequest(BaseModel):
    """Request model for reloading prompt."""

    prompt_version: int | None = Field(
        None, description="Specific prompt version to load, None for latest", ge=1
    )
    top_k: int = Field(
        5,
        description="Number of top documents to retrieve for context",
        ge=1,
        le=20,
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
