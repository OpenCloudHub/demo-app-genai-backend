from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration for RAG API and evaluation"""

    # API config
    prompt_version: int | None = None  # None = @production
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    # PgVector settings
    db_connection_string: str = (
        "postgresql+psycopg://demo-app:1234@localhost:5432/demo_app"
    )
    table_name: str = "readme_embeddings"
    top_k: int = 10

    # Prompt config
    mlflow_tracking_uri: str = "https://mlflow.internal.opencloudhub.org"
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    eval_data_path: str = "data/opencloudhub-readmes/rag-evaluation/questions.csv"
    prompt_name: str = "readme-rag-prompt"
    prompt_version: int | None = None  # Use @production alias

    # LLM settings
    llm_base_url: str = "https://api.opencloudhub.org/models/base/qwen-0.5b/v1"
    llm_model: str = "qwen2.5-0.5b-instruct"
    api_key: str = "dummy"  # Not used but required by ChatOpenAI
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    eval_data_version: str = "opencloudhub-readmes-rag-evaluation-v1.0.0"


CONFIG = Config()
