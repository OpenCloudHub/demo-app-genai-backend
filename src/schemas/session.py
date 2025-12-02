"""Session-related schema definitions."""

from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    """Response model for session creation."""

    session_id: str = Field(..., description="New session ID")
    created_at: datetime = Field(..., description="Session creation timestamp")


class SessionMessage(BaseModel):
    """A single message in chat history."""

    role: str = Field(..., description="Message role (human/ai)")
    content: str = Field(..., description="Message content")


class SessionHistoryResponse(BaseModel):
    """Response model for session history."""

    session_id: str = Field(..., description="Session ID")
    messages: list[SessionMessage] = Field(
        ..., description="List of messages in session"
    )
    message_count: int = Field(..., description="Total number of messages")


class SessionClearResponse(BaseModel):
    """Response model for clearing session."""

    session_id: str = Field(..., description="Session ID that was cleared")
    status: str = Field(..., description="Clear status")
