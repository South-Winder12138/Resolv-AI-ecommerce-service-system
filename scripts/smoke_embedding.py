"""Explicit real-model CPU/FP32 embedding check, without Milvus or an LLM."""

from importlib.metadata import version
from math import sqrt

from pydantic import ValidationError

from resolv.core.config import EmbeddingSettings
from resolv.knowledge import chunk_document, load_document
from resolv.services.embedding import EmbeddingError, EmbeddingService


def main() -> None:
    service = None
    body_error: BaseException | None = None
    try:
        settings = EmbeddingSettings()
        document = load_document("knowledge/refund.md")
        chunks = chunk_document(document)
        print("Embedding model:", settings.model_name)
        print("Device:", settings.device, "FP16:", settings.use_fp16)
        print("Normalize: True; batch_size:", settings.batch_size)
        print(
            "Token limits: query=",
            settings.query_max_length,
            "passage=",
            settings.passage_max_length,
        )
        for package in ("FlagEmbedding", "torch", "transformers", "numpy"):
            print(f"{package}: {version(package)}")
        print("Document:", document.document_id, "source=", document.source)
        print("Chunk count:", len(chunks), flush=True)

        service = EmbeddingService(settings)
        vectors = service.embed_documents([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            norm = sqrt(sum(value * value for value in vector))
            print(
                f"Chunk [{chunk.chunk_index}] dimension={len(vector)} L2 norm={norm:.6f}"
            )
        query = "软件已经激活了，还能退款吗？"
        print("Query:", query)
        vector = service.embed_query(query)
        norm = sqrt(sum(value * value for value in vector))
        print(f"Query vector dimension={len(vector)} L2 norm={norm:.6f}")
    except ValidationError as exc:
        body_error = exc
        raise SystemExit(
            "Embedding configuration invalid. Check BGE_M3_MODEL / EMBEDDING_*."
        ) from exc
    except EmbeddingError as exc:
        body_error = exc
        raise SystemExit(f"Embedding smoke failed: {exc}") from exc
    except (ValueError, OSError) as exc:
        body_error = exc
        raise SystemExit(
            "Embedding smoke failed. Check the sample file and input token budgets."
        ) from exc
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        if service is not None:
            try:
                service.close()
            except EmbeddingError as exc:
                if body_error is None:
                    raise SystemExit("Embedding smoke cleanup failed.") from exc
                body_error.add_note("Embedding model cleanup also failed.")
    print("Embedding smoke passed; model resources closed.")


if __name__ == "__main__":
    main()
