"""Dense query orchestration with caller-owned embedding and storage services."""

from resolv.repositories.milvus import MilvusVectorStore
from resolv.schemas.knowledge import SearchHit
from resolv.services.embedding import EmbeddingService


class Retriever:
    def __init__(
        self, embedding_service: EmbeddingService, vector_store: MilvusVectorStore
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Embed a valid query and return storage results, preserving failures."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonblank string")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        vector = self.embedding_service.embed_query(query)
        return self.vector_store.search(vector, top_k)
