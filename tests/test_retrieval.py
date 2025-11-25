"""Test retrieval to debug performance issues"""

import os

import urllib3

from src._config import CONFIG
from src.rag.chain import RAGChain

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["MLFLOW_TRACKING_URI"] = CONFIG.mlflow_tracking_uri

# Test questions
test_questions = [
    "What is GitOps in OpenCloudHub?",
    "How does Ray work for distributed training?",
    "What observability tools are used?",
]

print("Initializing RAG chain...")
rag = RAGChain(
    db_connection_string=CONFIG.connection_string,
    table_name=CONFIG.table_name,
    embedding_model=CONFIG.embedding_model,
    llm_base_url=CONFIG.llm_base_url,
    llm_model=CONFIG.llm_model,
    prompt_name="readme-rag-prompt",
    prompt_version=1,
    top_k=5,
)

print("\nTesting retrieval...")
for question in test_questions:
    print(f"\n{'=' * 60}")
    print(f"Q: {question}")
    print(f"{'=' * 60}")

    # Get retrieved docs
    docs = rag.vectorstore.similarity_search(question, k=5)

    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. [{doc.metadata['source_repo']}]")
        print(f"   Preview: {doc.page_content[:150]}...")

    # Get answer
    answer = rag.invoke(question)
    print(f"\nAnswer: {answer}")
