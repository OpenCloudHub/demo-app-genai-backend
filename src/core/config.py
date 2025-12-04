# ==============================================================================
# Application Configuration
# ==============================================================================
#
# Centralized configuration using Pydantic Settings with environment variables.
#
# Environment Variables (required):
#   DB_USER, DB_PASSWORD      - PostgreSQL credentials
#   MLFLOW_TRACKING_URI       - MLflow server URL
#   LLM_BASE_URL, LLM_MODEL   - OpenAI-compatible LLM endpoint
#   EMBEDDING_MODEL           - SentenceTransformer model name
#
# Environment Variables (optional):
#   DB_HOST (127.0.0.1), DB_PORT (5432), DB_NAME (demo_app)
#   DB_TABLE_NAME (readme_embeddings), DB_TOP_K (10)
#   PROMPT_NAME (readme-rag-prompt)
#   OTEL_ENABLED (true), OTEL_EXPORTER_OTLP_ENDPOINT (localhost:4317)
#
# Usage:
#   from src.core.config import CONFIG
#   print(CONFIG.db_connection_string)
#
# =============================================================================="""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration for RAG API and evaluation."""

    # API config
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    # Database credentials
    db_host: str = Field(default="127.0.0.1")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="demo_app")
    db_user: str
    db_password: str

    # PgVector settings
    db_table_name: str = Field(default="readme_embeddings")
    db_top_k: int = Field(default=10)

    # MLflow / Prompt config
    mlflow_tracking_uri: str
    dvc_repo: str = Field(default="https://github.com/OpenCloudHub/data-registry")
    eval_data_path: str = Field(
        default="data/opencloudhub-readmes/rag-evaluation/questions.csv"
    )
    prompt_name: str = Field(default="readme-rag-prompt")

    # LLM settings
    llm_base_url: str
    llm_model: str
    api_key: str

    # Embedding model
    embedding_model: str

    # ===================
    # Tracing Configuration
    # ===================
    # OpenTelemetry settings (using standard OTEL env var names)
    otel_enabled: bool = Field(default=True)
    otel_service_name: str = Field(default="demo-app-genai-backend")
    # Standard OTEL env var: OTEL_EXPORTER_OTLP_ENDPOINT
    otel_exporter_otlp_endpoint: str = Field(default="localhost:4317")

    # MLflow tracing (separate from OTEL - traces LangChain to MLflow UI)
    mlflow_tracing_enabled: bool = Field(default=True)

    @computed_field
    @property
    def db_connection_string(self) -> str:
        """SQLAlchemy format for PGEngine/vectorstore."""
        return f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @computed_field
    @property
    def db_connection_string_psycopg(self) -> str:
        """Standard format for psycopg direct connections (chat history)."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton config instance
CONFIG = Config()
