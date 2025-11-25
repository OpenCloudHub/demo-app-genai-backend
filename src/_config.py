from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    """Allows extra environment variables to be ignored."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class APIConfig(BaseServiceSettings):
    """Configuration for RAG API"""

    prompt_version: int | None = None  # None = @production
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)


class DatabaseConfig(BaseServiceSettings):
    """Configuration for database connection"""

    # PgVector settings
    connection_string: str = (
        "postgresql+psycopg://demo-app:1234@localhost:5432/demo_app"
    )
    table_name: str = "readme_embeddings"
    top_k: int = 10


class PromptConfig(BaseServiceSettings):
    """Configuration for prompt management"""

    mlflow_tracking_uri: str = "https://mlflow.internal.opencloudhub.org"
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    eval_data_path: str = "data/opencloudhub-readmes/rag-evaluation/questions.csv"
    prompt_name: str = "readme-rag-prompt"
    prompt_version: int | None = None  # Use @production alias


class ModelConfig(BaseServiceSettings):
    """Configuration for RAG chain"""

    # LLM settings
    llm_base_url: str = "https://api.opencloudhub.org/models/base/qwen-0.5b/v1"
    llm_model: str = "qwen2.5-0.5b-instruct"
    api_key: str = "dummy"  # Not used but required by ChatOpenAI
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class EvalConfig(BaseServiceSettings):
    """Configuration for RAG evaluation"""

    eval_data_version: str = "opencloudhub-readmes-rag-evaluation-v1.0.0"


class Config(BaseModel):
    """Configuration for RAG API and evaluation"""

    api: APIConfig = APIConfig()
    db: DatabaseConfig = DatabaseConfig()
    prompt: PromptConfig = PromptConfig()
    models: ModelConfig = ModelConfig()
    eval: EvalConfig = EvalConfig()


CONFIG = Config()
