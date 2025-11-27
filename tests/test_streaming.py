"""Test retrieval and streaming with deployed model"""

import os
import time

import pytest
import urllib3

from src._config import CONFIG
from src.rag.chain import RAGChain

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["MLFLOW_TRACKING_URI"] = CONFIG.mlflow_tracking_uri


@pytest.fixture
def rag_chain():
    """Initialize RAG chain for tests."""
    return RAGChain(
        db_connection_string=CONFIG.connection_string.db_connection_string,
        table_name=CONFIG.db_table_name,
        embedding_model=CONFIG.embedding_model,
        llm_base_url=CONFIG.llm_base_url,
        llm_model=CONFIG.llm_model,
        prompt_name="readme-rag-prompt",
        prompt_version=1,
        top_k=5,
    )


@pytest.mark.parametrize(
    "question",
    [
        "What is GitOps in OpenCloudHub?",
        "How does Ray work for distributed training?",
        "What observability tools are used?",
    ],
)
def test_retrieval_returns_documents(rag_chain, question):
    """Test that retrieval returns documents for each question."""
    docs = rag_chain.vectorstore.similarity_search(question, k=5)
    assert len(docs) > 0, f"No documents retrieved for: {question}"
    assert all(hasattr(doc, "metadata") for doc in docs)
    assert all("source_repo" in doc.metadata for doc in docs)


def test_streaming_response(rag_chain):
    """Test streaming token generation."""
    question = "What is GitOps in OpenCloudHub?"

    chunks = []
    start_time = time.time()

    for chunk in rag_chain.stream(question):
        chunks.append(chunk)
        print(chunk, end="", flush=True)

    elapsed_time = time.time() - start_time

    print(f"\n\nStreaming completed in {elapsed_time:.2f}s")
    print(f"Total chunks: {len(chunks)}")

    assert len(chunks) > 0, "No chunks received from streaming"
    full_response = "".join(chunks)
    assert len(full_response) > 0, "Empty response from streaming"


def test_streaming_vs_invoke(rag_chain):
    """Compare streaming vs regular invoke for consistency."""
    question = "What observability tools are used?"

    # Get regular response
    regular_response = rag_chain.invoke(question)

    # Get streamed response
    streamed_chunks = list(rag_chain.stream(question))
    streamed_response = "".join(streamed_chunks)

    print(f"\nRegular length: {len(regular_response)}")
    print(f"Streamed length: {len(streamed_response)}")

    # Both should produce non-empty responses
    assert len(regular_response) > 0
    assert len(streamed_response) > 0


def test_streaming_multiple_questions(rag_chain):
    """Test streaming works for multiple sequential questions."""
    questions = [
        "What is Ray in OpenCloudHub?",
        "How does ArgoCD work?",
    ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        print(f"{'=' * 60}\n")

        chunks = []
        for chunk in rag_chain.stream(question):
            chunks.append(chunk)
            print(chunk, end="", flush=True)

        print()  # newline
        assert len(chunks) > 0, f"No chunks for: {question}"


def test_retrieval_quality(rag_chain):
    """Test retrieval quality with relevant keywords."""
    question = "What is GitOps in OpenCloudHub?"
    docs = rag_chain.vectorstore.similarity_search(question, k=5)

    # Check if retrieved docs contain relevant terms
    combined_content = " ".join(doc.page_content.lower() for doc in docs)

    relevant_terms = ["gitops", "argocd", "deployment", "kubernetes"]
    found_terms = [term for term in relevant_terms if term in combined_content]

    print(f"\nRelevant terms found: {found_terms}")
    assert len(found_terms) > 0, "No relevant terms found in retrieved documents"


def test_streaming_performance(rag_chain):
    """Test streaming response time."""
    question = "What is OpenCloudHub?"

    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0

    for chunk in rag_chain.stream(question):
        chunk_count += 1
        if first_chunk_time is None:
            first_chunk_time = time.time() - start_time

    total_time = time.time() - start_time

    print(f"\nTime to first chunk: {first_chunk_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Chunks received: {chunk_count}")

    # First chunk should arrive reasonably fast (< 5 seconds)
    assert first_chunk_time < 5.0, f"First chunk took too long: {first_chunk_time:.2f}s"
    assert chunk_count > 0, "No chunks received"
