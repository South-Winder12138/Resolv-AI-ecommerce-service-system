"""Explicitly build one independent hybrid version from selected existing SQL documents."""

import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
from resolv.db.engine import create_session_factory
from resolv.db.knowledge import knowledge_engine
from resolv.db.knowledge_hybrid import hybrid_index_lock
from resolv.db.milvus import milvus_client
from resolv.knowledge import load_knowledge_file_configs
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.repositories.knowledge_hybrid_milvus import HybridMilvusStore
from resolv.schemas.hybrid_knowledge import HybridIndexProfile, HybridRetrievalError
from resolv.services.hybrid_embedding import REVISION, HybridEmbeddingAdapter
from resolv.services.hybrid_indexing import HybridKnowledgeIndexer
from resolv.services.hybrid_manifest import build_corpus_manifest


def build(args) -> dict:
    if args.embedding_revision != REVISION:
        raise HybridRetrievalError("hybrid_dense_revision_mismatch")
    profile = HybridIndexProfile(
        collection_name=args.collection,
        model_revision=args.embedding_revision,
        passage_max_length=args.passage_max_length,
    )
    configs = load_knowledge_file_configs(args.file_config)
    with ExitStack() as cleanup:
        for name in ("pymilvus", "pymilvus.milvus_client"):
            logger = logging.getLogger(name)
            cleanup.callback(setattr, logger, "handlers", logger.handlers)
            cleanup.callback(setattr, logger, "propagate", logger.propagate)
            logger.handlers = [logging.NullHandler()]
            logger.propagate = False
        engine = cleanup.enter_context(knowledge_engine(KnowledgeDatabaseSettings()))
        lock = cleanup.enter_context(hybrid_index_lock(engine, args.version))
        factory = create_session_factory(engine)
        with factory.begin() as session:
            repo = HybridIndexRepository(session)
            sources = repo.read_sources(args.document_id)
            manifest = build_corpus_manifest(
                sources, configs, max_chars=args.max_chars, overlap=args.overlap
            )
            repo.create_version(args.version, profile, manifest)
            lock.assert_owned()
        embedding = HybridEmbeddingAdapter(profile, cache_dir=args.cache_dir)
        cleanup.callback(embedding.close)
        client = cleanup.enter_context(milvus_client(MilvusSettings()))
        store = HybridMilvusStore(client, profile)
        result = HybridKnowledgeIndexer(factory, embedding, store, profile, lock).run(
            args.version, batch_size=args.batch_size
        )
    return {
        "build": result.model_dump(mode="json"),
        "index_profile": profile.model_dump(mode="json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("version", "collection", "embedding-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--document-id", required=True, action="append")
    parser.add_argument("--file-config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--passage-max-length", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        report = build(args)
    except Exception as exc:  # noqa: BLE001 -- command failures must never expose arbitrary credentials
        raise SystemExit(
            "Hybrid build failed: " + getattr(exc, "code", type(exc).__name__)
        ) from None
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
