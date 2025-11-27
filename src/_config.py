from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration for RAG API and evaluation"""

    # API config
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    # PgVector settings
    db_connection_string: str
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


CONFIG = Config()
