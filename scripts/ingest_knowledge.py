"""Explicit local Markdown/TXT indexing, without FastAPI, Agent or LLM calls."""

import argparse
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from pydantic import ValidationError

from resolv.core.config import EmbeddingSettings, MilvusSettings
from resolv.db.milvus import VectorStoreError, milvus_client
from resolv.knowledge import DEFAULT_SOURCE_ROOT, chunk_document, load_document
from resolv.repositories.milvus import MilvusVectorStore
from resolv.services.embedding import EmbeddingError, EmbeddingService


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Markdown/TXT path inside source-root")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        with ExitStack() as cleanup:
            # PyMilvus owns console handlers on both namespaces. This local CLI
            # prints safe boundary errors and restores SDK logging on every exit.
            for name in ("pymilvus", "pymilvus.milvus_client"):
                logger = logging.getLogger(name)
                cleanup.callback(setattr, logger, "handlers", logger.handlers)
                cleanup.callback(setattr, logger, "propagate", logger.propagate)
                logger.handlers = [logging.NullHandler()]
                logger.propagate = False

            document = load_document(args.path, source_root=args.source_root)
            chunks = chunk_document(
                document, max_chars=args.max_chars, overlap=args.overlap
            )
            embedding_settings = EmbeddingSettings()
            milvus_settings = MilvusSettings()
            print("Document:", document.document_id)
            print("Source:", document.source)
            print("Chunk count:", len(chunks))
            print("Embedding model:", embedding_settings.model_name)
            print("Device:", embedding_settings.device, "FP16:", embedding_settings.use_fp16)
            print("Normalize: True; batch_size:", embedding_settings.batch_size)
            print("Milvus collection:", milvus_settings.milvus_collection)
            for chunk in chunks:
                preview = chunk.text[:200] + ("…" if len(chunk.text) > 200 else "")
                print(f"Chunk [{chunk.chunk_index}] {chunk.chunk_id}\n{preview}")

            service = None
            body_error: BaseException | None = None
            try:
                if chunks:
                    service = EmbeddingService(embedding_settings)
                    print("Embedding dimension:", service.dimension, flush=True)
                    with milvus_client(milvus_settings) as client:
                        store = MilvusVectorStore(client, milvus_settings.milvus_collection)
                        store.ensure_collection(service.dimension)
                        vectors = service.embed_documents([chunk.text for chunk in chunks])
                        count = store.upsert(chunks, vectors)
                        print("Upsert count:", count)
                        if count != len(chunks):
                            raise VectorStoreError(
                                "Incomplete Milvus upsert; some chunks may already be written."
                            )
                else:
                    print("Embedding dimension: not loaded (empty document)")
                    print("Upsert count: 0")
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
            "Knowledge ingestion configuration invalid. Check MILVUS_* / BGE_M3_MODEL / EMBEDDING_*."
        ) from exc
    except (EmbeddingError, VectorStoreError) as exc:
        raise SystemExit(f"Knowledge ingestion failed: {exc}") from exc
    except (ValueError, OSError) as exc:
        raise SystemExit(
            "Knowledge ingestion failed. Check file path/encoding, chunk size and token budgets."
        ) from exc
    print("Ingestion completed; resources closed.")


if __name__ == "__main__":
    main()
