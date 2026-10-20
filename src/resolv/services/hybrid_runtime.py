"""Explicit caller-owned hybrid runtime. Nothing is activated by importing main."""

import logging
from contextlib import ExitStack, contextmanager

from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
from resolv.db.knowledge import knowledge_engine
from resolv.db.knowledge_hybrid import consistent_read_factory
from resolv.db.milvus import milvus_client
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.repositories.knowledge_hybrid_milvus import HybridMilvusStore
from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.services.hybrid_authority import HybridAuthorityReader
from resolv.services.hybrid_embedding import HybridEmbeddingAdapter
from resolv.services.hybrid_reranker import LocalHybridReranker
from resolv.services.hybrid_retrieval import HybridKnowledgeRetriever


@contextmanager
def hybrid_retrieval_runtime(
    *, version_id, index_profile, retrieval_profile, cache_dir=None
):
    try:
        with ExitStack() as cleanup:
            for name in ("pymilvus", "pymilvus.milvus_client"):
                logger = logging.getLogger(name)
                cleanup.callback(setattr, logger, "handlers", logger.handlers)
                cleanup.callback(setattr, logger, "propagate", logger.propagate)
                logger.handlers = [logging.NullHandler()]
                logger.propagate = False
            engine = cleanup.enter_context(
                knowledge_engine(KnowledgeDatabaseSettings())
            )
            factory = consistent_read_factory(engine)
            with factory() as session:
                version = HybridIndexRepository(session).get_version(version_id)
                if version.state != "ready" or version.profile != index_profile:
                    raise HybridRetrievalError("hybrid_runtime_version_mismatch")
            client = cleanup.enter_context(milvus_client(MilvusSettings()))
            store = HybridMilvusStore(client, index_profile)
            store.validate_existing()
            embedding = HybridEmbeddingAdapter(index_profile, cache_dir=cache_dir)
            cleanup.callback(embedding.close)
            reranker = LocalHybridReranker(retrieval_profile, cache_dir=cache_dir)
            cleanup.callback(reranker.close)
            authority = HybridAuthorityReader(factory, version_id, index_profile)
            retriever = HybridKnowledgeRetriever(
                embedding, store, authority, reranker, index_profile, retrieval_profile
            )
            cleanup.callback(setattr, retriever, "_active", False)
            yield retriever
    except Exception as exc:
        if isinstance(exc, HybridRetrievalError):
            raise
        raise HybridRetrievalError("hybrid_runtime_failed") from exc
