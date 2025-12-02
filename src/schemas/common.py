"""Common schema definitions shared across the API."""

from enum import StrEnum, auto
from typing import Optional

from pydantic import BaseModel, Field


class APIStatus(StrEnum):
    """API status enumeration."""

    LOADING = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    NOT_READY = auto()


class ErrorDetail(BaseModel):
    """Error detail model."""

    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str | ErrorDetail = Field(..., description="Error details")
