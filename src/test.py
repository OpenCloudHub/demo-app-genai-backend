import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_openai import ChatOpenAI
from langchain_postgres import PGEngine, PGVectorStore

# Disable SSL verification for your local setup
http_client = httpx.Client(verify=False)

# Initialize your Qwen model
llm = ChatOpenAI(
    model="qwen2.5-0.5b-instruct",
    base_url="https://api.opencloudhub.org/models/qwen-0.5b/v1",
    api_key="dummy",
    http_client=http_client,
)

# Correct connection string format for psycopg
CONNECTION_STRING = "postgresql+psycopg://demo-app:1234@localhost:5432/demo_app"
VECTOR_SIZE = 768
embedding = DeterministicFakeEmbedding(size=VECTOR_SIZE)

# Initialize the engine
pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

# Initialize the vector store table
TABLE_NAME = "test_collection"
pg_engine.init_vectorstore_table(
    table_name=TABLE_NAME,
    vector_size=VECTOR_SIZE,
)

# Create the vector store
vector_store = PGVectorStore.create_sync(
    engine=pg_engine,
    table_name=TABLE_NAME,
    embedding_service=embedding,
)

# Test with some documents
docs = [
    Document(page_content="This is a test document about MLOps"),
    Document(page_content="OpenCloudHub is a Kubernetes platform"),
]

vector_store.add_documents(docs)
print("✓ Successfully connected to pgvector database and added documents!")

# Test similarity search
results = vector_store.similarity_search("Kubernetes", k=1)
print(f"Search results: {results}")
