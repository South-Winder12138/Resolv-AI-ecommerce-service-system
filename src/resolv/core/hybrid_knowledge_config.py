"""Explicit independent profiles; never infer versions or select a collection."""
from resolv.schemas.hybrid_knowledge import HybridIndexProfile, HybridRetrievalProfile


def default_index_profile(collection_name: str) -> HybridIndexProfile:
    return HybridIndexProfile(collection_name=collection_name)


def default_retrieval_profile() -> HybridRetrievalProfile:
    return HybridRetrievalProfile()
