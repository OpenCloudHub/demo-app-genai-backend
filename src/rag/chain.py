"""LangChain-based RAG chain with MLflow tracing and chat history."""

import uuid
from typing import Optional

import httpx
import mlflow
import psycopg
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from langchain_postgres import PGEngine, PGVectorStore, PostgresChatMessageHistory
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)

from src.core import CONFIG, get_logger
from src.core.tracing import setup_mlflow_autolog, trace_if_enabled
from src.rag.embeddings import SentenceTransformerEmbeddings

logger = get_logger(__name__)

# Enable MLflow autologging (respects OTEL_ENABLED setting)
setup_mlflow_autolog()


class RAGChain:
    """RAG chain with LangChain, chat history, and MLflow tracing."""

    def __init__(
        self,
        db_connection_string: str,
        db_connection_string_psycopg: str,
        table_name: str,
        embedding_model: str,
        llm_base_url: str,
        llm_model: str,
        api_key: str,
        prompt_name: str,
        prompt_version: Optional[int] = None,
        top_k: int = 5,
        chat_history_table: str = "chat_history",
    ):
        self.db_connection_string = db_connection_string
        self.db_connection_string_psycopg = db_connection_string_psycopg
        self.chat_history_table = chat_history_table
        self.top_k = top_k
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version

        # Initialize LLM
        http_client = httpx.Client(verify=False)
        self.llm = ChatOpenAI(
            model=llm_model,
            base_url=llm_base_url,
            api_key=api_key,
            http_client=http_client,
            temperature=0.7,
        )

        # Initialize embedder
        self.embedder = SentenceTransformerEmbeddings(embedding_model)

        # Connect to vectorstore
        self._init_vectorstore(db_connection_string, table_name, top_k)

        # Create chat history table
        self._init_chat_history_table()

        # Load prompt
        self.prompt_template = self._load_prompt(prompt_name, prompt_version)

    def _init_vectorstore(self, connection_string: str, table_name: str, top_k: int):
        """Initialize PGVector store with hybrid search."""
        try:
            pg_engine = PGEngine.from_connection_string(
                url=connection_string,
                connect_args={"connect_timeout": 10},
            )
            logger.info("✓ Connected to Postgres vectorstore")
        except Exception as e:
            logger.error(f"Failed to connect to Postgres: {e}")
            raise

        hybrid_config = HybridSearchConfig(
            tsv_lang="pg_catalog.english",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={"rrf_k": 60, "fetch_top_k": top_k},
        )

        self.vectorstore = PGVectorStore.create_sync(
            engine=pg_engine,
            table_name=table_name,
            embedding_service=self.embedder,
            hybrid_search_config=hybrid_config,
            id_column="id",
            content_column="content",
            embedding_column="embedding",
            metadata_json_column="metadata",
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

    def _init_chat_history_table(self):
        """Create chat history table if it doesn't exist."""
        try:
            with psycopg.connect(
                self.db_connection_string_psycopg, connect_timeout=30
            ) as conn:
                PostgresChatMessageHistory.create_tables(conn, self.chat_history_table)
            logger.info(f"✓ Chat history table '{self.chat_history_table}' ready")
        except Exception as e:
            logger.error(f"Failed to create chat history table: {e}")
            raise

    def _get_chat_history(self, session_id: str) -> PostgresChatMessageHistory:
        """Get chat history for a session."""
        conn = psycopg.connect(self.db_connection_string_psycopg, connect_timeout=10)
        return PostgresChatMessageHistory(
            self.chat_history_table,
            session_id,
            sync_connection=conn,
        )

    def _load_prompt(
        self, prompt_name: str, version: Optional[int] = None
    ) -> ChatPromptTemplate:
        """Load prompt from MLflow with chat history placeholder.

        Args:
            prompt_name: Name of the prompt in MLflow
            version: Specific version to load, or None to load @champion alias
        """
        mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
        try:
            if version is not None:
                prompt_uri = f"prompts:/{prompt_name}/{version}"
            else:
                # Load production/champion version when no specific version requested
                prompt_uri = f"prompts:/{prompt_name}@champion"

            mlflow_prompt = mlflow.genai.load_prompt(prompt_uri)
            # Update instance version from loaded prompt
            self.prompt_version = mlflow_prompt.version
            logger.info(f"✓ Loaded prompt: {prompt_uri} (v{self.prompt_version})")

            return ChatPromptTemplate.from_messages(
                [
                    ("system", mlflow_prompt.to_single_brace_format()),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt {prompt_name}: {e}")

    @trace_if_enabled(name="format_context")
    def _format_docs(self, docs) -> str:
        """Format retrieved documents with metadata."""
        if not docs:
            return "No relevant context found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source_repo = doc.metadata.get("source_repo", "unknown")
            source_file = doc.metadata.get("source_file", "unknown")

            # Build section path from headers
            section_parts = []
            if doc.metadata.get("section_h1"):
                section_parts.append(doc.metadata["section_h1"])
            if doc.metadata.get("section_h2"):
                section_parts.append(doc.metadata["section_h2"])
            if doc.metadata.get("section_h3"):
                section_parts.append(doc.metadata["section_h3"])

            section_path = " > ".join(section_parts) if section_parts else ""

            if section_path:
                header = f"[Source {i}: {source_repo} | Section: {section_path}]"
            else:
                header = f"[Source {i}: {source_repo}/{source_file}]"

            formatted.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(formatted)

    def _build_chain_with_history(self, chat_history: PostgresChatMessageHistory):
        """Build RAG chain with chat history."""

        def get_history(_):
            return chat_history.messages

        chain = (
            RunnableParallel(
                {
                    "context": self.retriever | self._format_docs,
                    "chat_history": RunnableLambda(get_history),
                    "question": RunnablePassthrough(),
                }
            )
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        return chain

    @trace_if_enabled(name="rag_query")
    def invoke(self, question: str, session_id: Optional[str] = None) -> str:
        """Invoke chain with tracing and chat history."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        chat_history = self._get_chat_history(session_id)
        chain = self._build_chain_with_history(chat_history)

        try:
            response = chain.invoke(question)
        except Exception as e:
            logger.error(f"Chain invoke failed: {e}", exc_info=True)
            raise

        chat_history.add_messages(
            [
                HumanMessage(content=question),
                AIMessage(content=response),
            ]
        )

        return response

    @trace_if_enabled(name="rag_stream")
    def stream(self, question: str, session_id: Optional[str] = None):
        """Stream the response token by token with chat history."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        chat_history = self._get_chat_history(session_id)
        chain = self._build_chain_with_history(chat_history)

        try:
            logger.debug(f"Starting stream for: {question[:50]}...")
            full_response = []

            for chunk in chain.stream(question):
                if hasattr(chunk, "content"):
                    if chunk.content:
                        full_response.append(chunk.content)
                        yield chunk.content
                elif isinstance(chunk, str):
                    if chunk:
                        full_response.append(chunk)
                        yield chunk
                elif isinstance(chunk, dict):
                    content = (
                        chunk.get("answer") or chunk.get("output") or chunk.get("text")
                    )
                    if content:
                        full_response.append(content)
                        yield content

            chat_history.add_messages(
                [
                    HumanMessage(content=question),
                    AIMessage(content="".join(full_response)),
                ]
            )

            logger.info(f"Stream completed, saved to session {session_id}")

        except Exception as e:
            logger.error(f"Stream error in chain: {e}", exc_info=True)
            raise

    def create_session(self) -> str:
        """Create a new chat session and return the session ID."""
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session_history(self, session_id: str) -> list:
        """Get all messages for a session."""
        chat_history = self._get_chat_history(session_id)
        return chat_history.messages

    def clear_session(self, session_id: str):
        """Clear chat history for a session."""
        chat_history = self._get_chat_history(session_id)
        chat_history.clear()
        logger.info(f"Cleared session: {session_id}")
