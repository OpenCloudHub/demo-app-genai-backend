"""Test retrieval to debug performance issues"""

import os

import urllib3

from src._config import CONFIG
from src._logging import get_logger
from src.rag.chain import RAGChain

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["MLFLOW_TRACKING_URI"] = CONFIG.mlflow_tracking_uri

logger = get_logger("debug_retrieval")

logger.log_section("Debug Retrieval Performance", emoji="🔍")

# Test questions
test_questions = [
    "What is GitOps in OpenCloudHub?",
    "How does Ray work for distributed training?",
    "What observability tools are used?",
]

logger.info("Initializing RAG chain...")
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

logger.info("\nTesting retrieval...")
for question in test_questions:
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Q: {question}")
    logger.info(f"{'=' * 60}")

    # Get retrieved docs
    docs = rag.vectorstore.similarity_search(question, k=5)

    logger.info(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n{i}. [{doc.metadata['source_repo']}]")
        logger.info(f"   Preview: {doc.page_content[:150]}...")

    # Get answer
    answer = rag.invoke(question)
    logger.info(f"\nAnswer: {answer}")
