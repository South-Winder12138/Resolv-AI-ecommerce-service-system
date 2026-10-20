"""Query an existing Milvus knowledge index with real BGE-M3 dense embedding."""

import argparse
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from importlib.metadata import version

from pydantic import ValidationError

from resolv.core.config import EmbeddingSettings, MilvusSettings
from resolv.db.milvus import VectorStoreError, milvus_client
from resolv.repositories.milvus import MilvusVectorStore
from resolv.services.embedding import EmbeddingError, EmbeddingService
from resolv.services.retrieval import Retriever


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="软件已经激活了，还能退款吗？")
    parser.add_argument("--top-k", type=int, default=5, help="maximum number of hits")
    args = parser.parse_args(argv)
    if not args.query.strip():
        parser.error("query must be a nonblank string")
    if args.top_k <= 0:
        parser.error("top-k must be a positive integer")

    try:
        with ExitStack() as cleanup:
            # PyMilvus owns console handlers on both namespaces. The CLI emits
            # safe boundary errors and restores the SDK's logging on every exit.
            for name in ("pymilvus", "pymilvus.milvus_client"):
                logger = logging.getLogger(name)
                cleanup.callback(setattr, logger, "handlers", logger.handlers)
                cleanup.callback(setattr, logger, "propagate", logger.propagate)
                logger.handlers = [logging.NullHandler()]
                logger.propagate = False

            embedding_settings = EmbeddingSettings()
            milvus_settings = MilvusSettings()
            print("Query:", args.query)
            print("Top", args.top_k)
            print("Embedding model:", embedding_settings.model_name)
            print("Device:", embedding_settings.device, "FP16:", embedding_settings.use_fp16)
            print(
                "Normalize: True; batch_size:", embedding_settings.batch_size,
                "query_max_length:", embedding_settings.query_max_length,
            )
            print("Milvus collection:", milvus_settings.milvus_collection)
            for package in ("FlagEmbedding", "torch", "transformers", "numpy", "pymilvus"):
                print(f"{package}: {version(package)}")

            service = None
            body_error: BaseException | None = None
            try:
                service = EmbeddingService(embedding_settings)
                print("Embedding dimension:", service.dimension, flush=True)
                with milvus_client(milvus_settings) as client:
                    store = MilvusVectorStore(client, milvus_settings.milvus_collection)
                    hits = Retriever(service, store).retrieve(args.query, top_k=args.top_k)
                    print("Hits:", len(hits))
                    for rank, hit in enumerate(hits, start=1):
                        print(f"\n[{rank}]")
                        print(f"score={hit.score}")
                        print(f"chunk_id={hit.chunk_id}")
                        print(f"document_id={hit.document_id}")
                        print(f"source={hit.source}")
                        print(f"chunk_index={hit.chunk_index}\n")
                        print(hit.text)
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                if service is not None:
                    try:
                        service.close()
                    except EmbeddingError:
                        if body_error is None:
                            raise
                        body_error.add_note("Embedding model cleanup also failed.")
    except ValidationError as exc:
        raise SystemExit(
            "Retrieval configuration invalid. Check MILVUS_* / BGE_M3_MODEL / EMBEDDING_*."
        ) from exc
    except (EmbeddingError, VectorStoreError) as exc:
        raise SystemExit(f"Retrieval smoke failed: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(
            "Retrieval input invalid. Check query, top_k and EMBEDDING_QUERY_MAX_LENGTH."
        ) from exc
    print("Retrieval completed; resources closed.")


if __name__ == "__main__":
    main()
