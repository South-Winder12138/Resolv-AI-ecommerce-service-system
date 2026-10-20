"""Index committed pending knowledge with a pinned local model and explicit lock."""

import argparse
import json
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from pydantic import ValidationError

from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
from resolv.db.engine import create_session_factory
from resolv.db.knowledge import KnowledgeStoreError, knowledge_engine
from resolv.db.knowledge_index import knowledge_index_lock
from resolv.db.milvus import VectorStoreError, milvus_client
from resolv.repositories.knowledge_milvus import KnowledgeMilvusVectorStore
from resolv.schemas.knowledge import KnowledgeIndexProfile
from resolv.services.embedding import EmbeddingError
from resolv.services.knowledge_embedding import pinned_knowledge_embedding
from resolv.services.knowledge_indexing import KnowledgeIndexer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-revision", required=True, help="Pinned lowercase 40-character HF commit")
    parser.add_argument("--collection", default="resolv_knowledge_v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--passage-max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=512)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or not 1 <= args.query_max_length <= 8192:
        parser.error("batch-size must be positive; query-max-length must be between 1 and 8192")
    try:
        profile = KnowledgeIndexProfile(
            model_revision=args.model_revision, collection_name=args.collection,
            passage_max_length=args.passage_max_length,
        )
        database_settings, vector_settings = KnowledgeDatabaseSettings(), MilvusSettings()
        with ExitStack() as cleanup:
            for name in ("pymilvus", "pymilvus.milvus_client"):
                logger = logging.getLogger(name)
                cleanup.callback(setattr, logger, "handlers", logger.handlers)
                cleanup.callback(setattr, logger, "propagate", logger.propagate)
                logger.handlers, logger.propagate = [logging.NullHandler()], False
            engine = cleanup.enter_context(knowledge_engine(database_settings))
            lock = cleanup.enter_context(knowledge_index_lock(engine))
            embedding = cleanup.enter_context(pinned_knowledge_embedding(
                profile, batch_size=args.batch_size, query_max_length=args.query_max_length,
                cache_dir=args.cache_dir,
            ))
            client = cleanup.enter_context(milvus_client(vector_settings))
            store = KnowledgeMilvusVectorStore(client, profile.collection_name)
            store.ensure_collection(embedding.dimension)
            result = KnowledgeIndexer(
                create_session_factory(engine), embedding, store, profile, lock
            ).run(batch_size=args.batch_size)
    except ValidationError as exc:
        raise SystemExit("Knowledge index configuration invalid. Check pinned revision, collection and database settings.") from exc
    except (KnowledgeStoreError, EmbeddingError, VectorStoreError) as exc:
        raise SystemExit("Knowledge indexing failed. Inspect pending state, MySQL lock, local model cache and Milvus before retrying.") from exc
    except ValueError as exc:
        raise SystemExit("Knowledge indexing input/profile mismatch. Check staged hashes and declared index profile.") from exc
    print(json.dumps({"index": result.model_dump(), "index_profile": profile.model_dump()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
