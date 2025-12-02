"""Schema definitions for the RAG API."""

from src.schemas.admin import ReloadPromptRequest, ReloadPromptResponse
from src.schemas.common import APIStatus, ErrorDetail, ErrorResponse
from src.schemas.debug import (
    DebugConfigResponse,
    DebugRetrievalRequest,
    DebugRetrievalResponse,
    RetrievedDocument,
)
from src.schemas.health import HealthResponse, RootResponse
from src.schemas.query import QueryRequest, QueryResponse
from src.schemas.session import (
    SessionClearResponse,
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionMessage,
)

__all__ = [
    # Common
    "APIStatus",
    "ErrorDetail",
    "ErrorResponse",
    # Query
    "QueryRequest",
    "QueryResponse",
    # Session
    "SessionCreateResponse",
    "SessionMessage",
    "SessionHistoryResponse",
    "SessionClearResponse",
    # Debug
    "RetrievedDocument",
    "DebugRetrievalRequest",
    "DebugRetrievalResponse",
    "DebugConfigResponse",
    # Health
    "HealthResponse",
    "RootResponse",
    # Admin
    "ReloadPromptRequest",
    "ReloadPromptResponse",
]
