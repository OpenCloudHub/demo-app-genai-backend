"""
LangChain-based RAG chain with proper schema mapping and tracing.
"""

from typing import Optional

import httpx
import mlflow
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from sentence_transformers import SentenceTransformer

from src._logging import get_logger

logger = get_logger(__name__)

# Enable LangChain autologging
mlflow.langchain.autolog()


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer to work with LangChain."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class RAGChain:
    """RAG chain with LangChain standard schema."""

    def __init__(
        self,
        db_connection_string: str,
        table_name: str,
        embedding_model: str,
        llm_base_url: str,
        llm_model: str,
        prompt_name: str,
        prompt_version: Optional[int] = None,  # None loads @production
        top_k: int = 5,
    ):
        # Initialize LLM
        http_client = httpx.Client(verify=False)
        self.llm = ChatOpenAI(
            model=llm_model,
            base_url=llm_base_url,
            api_key="dummy",
            http_client=http_client,
            temperature=0.7,
        )

        # Initialize embedder
        self.embedder = SentenceTransformerEmbeddings(embedding_model)

        # Connect to existing table with LangChain schema
        try:
            pg_engine = PGEngine.from_connection_string(
                url=db_connection_string,
                # Test connection
                connect_args={"connect_timeout": 10},
            )
            logger.info("✓ Connected to Postgres vectorstore")
        except Exception as e:
            logger.error(f"Failed to connect to Postgres: {e}")
            raise

        # Initialize vectorstore with hybrid search config

        hybrid_config = HybridSearchConfig(
            tsv_lang="pg_catalog.english",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={"rrf_k": 60, "fetch_top_k": top_k},
        )

        # Now using standard LangChain schema - no custom mapping needed!
        self.vectorstore = PGVectorStore.create_sync(
            engine=pg_engine,
            table_name=table_name,
            embedding_service=self.embedder,
            hybrid_search_config=hybrid_config,
            # Map to our custom column names
            id_column="id",
            content_column="content",
            embedding_column="embedding",
            metadata_json_column="metadata",
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.top_k = top_k
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version

        # Load prompt
        self.prompt_template = self._load_prompt(prompt_name, prompt_version)

        # Build chain
        self.chain = self._build_chain()

    def _load_prompt(self, prompt_name: str, version: int):
        """Load prompt from MLflow"""
        try:
            prompt_uri = f"prompts:/{prompt_name}/{version}"

            mlflow_prompt = mlflow.genai.load_prompt(prompt_uri)
            logger.info(f"✓ Loaded prompt: {prompt_uri}")

            return ChatPromptTemplate.from_messages(
                [
                    ("system", mlflow_prompt.to_single_brace_format()),
                ]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt {prompt_name}: {e}")

    @mlflow.trace(name="format_context")
    def _format_docs(self, docs) -> str:
        """Format retrieved documents with metadata from JSONB"""
        if not docs:
            return "No relevant context found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            # Extract from metadata JSONB
            source_repo = doc.metadata.get("source_repo", "unknown")
            source_file = doc.metadata.get("source_file", "unknown")

            formatted.append(
                f"[Source {i}: {source_repo}/{source_file}]\n{doc.page_content}"
            )

        return "\n\n".join(formatted)

    def _build_chain(self):
        """Build RAG chain with tracing"""
        chain = (
            RunnableParallel(
                {
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough(),
                }
            )
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        return chain

    @mlflow.trace(name="rag_query")
    def invoke(self, question: str) -> str:
        """Invoke chain with tracing"""
        return self.chain.invoke(question)

    @mlflow.trace(name="rag_stream")
    def stream(self, question: str):
        """Stream the response token by token."""
        try:
            logger.debug(f"Starting stream for: {question[:50]}...")
            chunk_count = 0

            for chunk in self.chain.stream(
                question
            ):  # Pass question directly, not as dict
                chunk_count += 1
                logger.debug(
                    f"Chunk {chunk_count}: type={type(chunk)}, repr={repr(chunk)[:100]}"
                )

                # Handle different chunk types from LangChain
                if hasattr(chunk, "content"):
                    # AIMessageChunk
                    if chunk.content:
                        yield chunk.content
                elif isinstance(chunk, str):
                    # String chunk
                    if chunk:
                        yield chunk
                elif isinstance(chunk, dict):
                    # Dict chunk - try different keys
                    content = (
                        chunk.get("answer") or chunk.get("output") or chunk.get("text")
                    )
                    if content:
                        yield content
                else:
                    logger.warning(f"Unknown chunk type: {type(chunk)}, value: {chunk}")

            logger.info(f"Stream completed with {chunk_count} chunks")

        except Exception as e:
            logger.error(f"Stream error in chain: {e}", exc_info=True)
            raise

    def get_langchain_chain(self):
        """Return underlying chain"""
        return self.chain
