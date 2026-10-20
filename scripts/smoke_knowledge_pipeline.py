"""Stage one configured file, index pending knowledge, and retrieve with SQL authority."""

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
from resolv.knowledge import (
    DEFAULT_SOURCE_ROOT,
    chunk_knowledge_document,
    get_knowledge_file_config,
    load_document,
    load_knowledge_file_configs,
)
from resolv.repositories.knowledge_milvus import KnowledgeMilvusVectorStore
from resolv.schemas.knowledge import KnowledgeIndexProfile
from resolv.services.embedding import EmbeddingError
from resolv.services.knowledge import KnowledgeConflictError, KnowledgeIngestionService
from resolv.services.knowledge_embedding import pinned_knowledge_embedding
from resolv.services.knowledge_indexing import KnowledgeIndexer
from resolv.services.knowledge_retrieval import KnowledgeRetriever


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--query", default="软件已经激活了，还能退款吗？")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-revision", required=True, help="Pinned lowercase 40-character HF commit")
    parser.add_argument("--collection", default="resolv_knowledge_v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--passage-max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=512)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.query.strip() or not 1 <= args.top_k <= 16383:
        parser.error("query must be nonblank; top-k must be between 1 and 16383")
    if args.batch_size <= 0 or not 1 <= args.query_max_length <= 8192:
        parser.error("batch-size must be positive; query-max-length must be between 1 and 8192")
    try:
        profile = KnowledgeIndexProfile(
            model_revision=args.model_revision, collection_name=args.collection,
            passage_max_length=args.passage_max_length,
        )
        configs = load_knowledge_file_configs(args.config, source_root=args.source_root)
        document = load_document(args.path, source_root=args.source_root)
        file_config = get_knowledge_file_config(configs, document.source)
        # Validate the file/chunk budget before opening databases or loading the model.
        chunk_knowledge_document(document, file_config, max_chars=args.max_chars, overlap=args.overlap)
        database_settings, vector_settings = KnowledgeDatabaseSettings(), MilvusSettings()
        with ExitStack() as cleanup:
            for name in ("pymilvus", "pymilvus.milvus_client"):
                logger = logging.getLogger(name)
                cleanup.callback(setattr, logger, "handlers", logger.handlers)
                cleanup.callback(setattr, logger, "propagate", logger.propagate)
                logger.handlers, logger.propagate = [logging.NullHandler()], False
            engine = cleanup.enter_context(knowledge_engine(database_settings))
            lock = cleanup.enter_context(knowledge_index_lock(engine))
            factory = create_session_factory(engine)
            staged = KnowledgeIngestionService(factory).stage_document(
                document, file_config, profile, max_chars=args.max_chars, overlap=args.overlap
            )
            embedding = cleanup.enter_context(pinned_knowledge_embedding(
                profile, batch_size=args.batch_size, query_max_length=args.query_max_length,
                cache_dir=args.cache_dir,
            ))
            client = cleanup.enter_context(milvus_client(vector_settings))
            store = KnowledgeMilvusVectorStore(client, profile.collection_name)
            store.ensure_collection(embedding.dimension)
            indexed = KnowledgeIndexer(factory, embedding, store, profile, lock).run(batch_size=args.batch_size)
            hits = KnowledgeRetriever(factory, embedding, store, profile).retrieve(args.query, top_k=args.top_k)
    except ValidationError as exc:
        raise SystemExit("Knowledge pipeline configuration invalid. Check metadata, pinned revision and database settings.") from exc
    except KnowledgeConflictError as exc:
        raise SystemExit(str(exc)) from exc
    except (KnowledgeStoreError, EmbeddingError, VectorStoreError) as exc:
        raise SystemExit("Knowledge pipeline failed. Inspect pending state, MySQL lock, local model cache and Milvus before retrying.") from exc
    except (ValueError, OSError) as exc:
        raise SystemExit("Knowledge pipeline input/profile invalid. Check UTF-8 files, source-root, TOML mappings and chunk budgets.") from exc
    print(json.dumps({
        "stage": staged.model_dump(), "index": indexed.model_dump(),
        "index_profile": profile.model_dump(), "query": args.query, "top_k": args.top_k,
        "hits": [hit.model_dump() for hit in hits],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
