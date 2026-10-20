"""Reuse pinned local BGE inference; v3 profile and fingerprints stay independent."""

from collections.abc import Sequence
from pathlib import Path

from resolv.schemas.hybrid_knowledge import HybridIndexProfile, HybridRetrievalError
from resolv.schemas.knowledge import KnowledgeIndexProfile
from resolv.services.hybrid_reranker import INFERENCE_LOCK
from resolv.services.knowledge_embedding import KnowledgeEmbeddingService

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class HybridEmbeddingAdapter:
    def __init__(self, profile: HybridIndexProfile, *, cache_dir: Path | None = None):
        self.profile = HybridIndexProfile.model_validate(profile.model_dump())
        if self.profile.model_revision != REVISION:
            raise HybridRetrievalError("hybrid_dense_revision_mismatch")
        dense = KnowledgeIndexProfile(
            model_revision=self.profile.model_revision,
            collection_name=self.profile.collection_name,
            model_name=self.profile.model_name,
            dimension=self.profile.dimension,
            normalize_embeddings=self.profile.normalize_embeddings,
            use_fp16=self.profile.use_fp16,
            passage_max_length=self.profile.passage_max_length,
            text_rule=self.profile.text_rule,
        )
        self._service = KnowledgeEmbeddingService(dense, cache_dir=cache_dir)

    @property
    def dimension(self) -> int:
        return self._service.dimension

    def embed_query(self, text: str) -> list[float]:
        with INFERENCE_LOCK:
            return self._service.embed_query(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with INFERENCE_LOCK:
            return self._service.embed_documents(texts)

    def close(self) -> None:
        with INFERENCE_LOCK:
            self._service.close()
