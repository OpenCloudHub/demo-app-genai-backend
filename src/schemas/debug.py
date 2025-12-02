"""Debug-related schema definitions."""

from typing import Optional

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """A single retrieved document from vector search."""

    rank: int = Field(..., description="Retrieval rank (1 = most relevant)")
    content: str = Field(..., description="Document content")
    source_repo: Optional[str] = Field(None, description="Source repository")
    source_file: Optional[str] = Field(None, description="Source file name")
    section_h1: Optional[str] = Field(None, description="H1 header context")
    section_h2: Optional[str] = Field(None, description="H2 header context")
    section_h3: Optional[str] = Field(None, description="H3 header context")
    data_version: Optional[str] = Field(None, description="Data version tag")
    similarity_score: Optional[float] = Field(
        None, description="Similarity score if available"
    )


class DebugRetrievalRequest(BaseModel):
    """Request model for debug retrieval."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to retrieve documents for",
    )


class DebugRetrievalResponse(BaseModel):
    """Response model for debug retrieval."""

    question: str = Field(..., description="The query used for retrieval")
    top_k: int = Field(..., description="Number of documents requested")
    documents: list[RetrievedDocument] = Field(..., description="Retrieved documents")
    formatted_context: str = Field(
        ..., description="Context as it would be sent to LLM"
    )
    retrieval_time_ms: float = Field(
        ..., description="Time taken to retrieve documents"
    )


class DebugConfigResponse(BaseModel):
    """Response model for debug config."""

    prompt_name: str = Field(..., description="Current prompt name")
    prompt_version: int = Field(..., description="Current prompt version")
    prompt_template: str = Field(..., description="Current prompt template")
    embedding_model: str = Field(..., description="Embedding model in use")
    llm_model: str = Field(..., description="LLM model in use")
    top_k: int = Field(..., description="Default number of documents to retrieve")
    table_name: str = Field(..., description="Vector store table name")
