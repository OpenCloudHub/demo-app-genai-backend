# ==============================================================================
# Embedding Models for RAG
# ==============================================================================
#
# LangChain-compatible embedding wrapper for SentenceTransformers.
#
# This wrapper allows using any SentenceTransformer model with LangChain's
# PGVectorStore and other retrieval components.
#
# Supported Models (examples):
#   - sentence-transformers/all-MiniLM-L6-v2 (fast, 384 dims)
#   - sentence-transformers/all-mpnet-base-v2 (better quality, 768 dims)
#   - BAAI/bge-small-en-v1.5 (good balance)
#
# Usage:
#   embedder = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
#   vectors = embedder.embed_documents(["doc1", "doc2"])
#   query_vec = embedder.embed_query("search query")
#
# =============================================================================="""

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer to work with LangChain."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
