"""Application configuration using Pydantic settings."""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration for RAG API and evaluation."""

    # API config
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    # Database credentials
    db_host: str = Field(default="127.0.0.1")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="demo_app")
    db_user: str
    db_password: str

    # PgVector settings
    db_table_name: str = "readme_embeddings"
    db_top_k: int = 10

    # Prompt config
    mlflow_tracking_uri: str
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    eval_data_path: str = "data/opencloudhub-readmes/rag-evaluation/questions.csv"
    prompt_name: str = "readme-rag-prompt"

    # LLM settings
    llm_base_url: str
    llm_model: str
    api_key: str

    # Embedding model
    embedding_model: str

    # Tracing
    otel_enabled: bool = Field(default=True)
    otel_service_name: str = Field(default="demo-app-genai-backend")
    otel_exporter_otlp_endpoint: str = Field(default="localhost:4317")

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


CONFIG = Config()
