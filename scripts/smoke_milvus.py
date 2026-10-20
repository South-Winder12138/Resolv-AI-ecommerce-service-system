"""Check a real Milvus connection without creating collections or loading models."""

import logging
from contextlib import ExitStack

from pydantic import ValidationError

from resolv.core.config import MilvusSettings
from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, VectorStoreError, milvus_client


def main() -> None:
    with ExitStack() as cleanup:
        # PyMilvus 3.0.2 has separate console handlers on these two namespaces.
        # This single-threaded CLI owns its output; restore SDK logging on exit.
        for name in ("pymilvus", "pymilvus.milvus_client"):
            logger = logging.getLogger(name)
            cleanup.callback(setattr, logger, "handlers", logger.handlers)
            cleanup.callback(setattr, logger, "propagate", logger.propagate)
            logger.handlers = [logging.NullHandler()]
            logger.propagate = False
        try:
            settings = MilvusSettings()
            with milvus_client(settings) as client:
                try:
                    collections = client.list_collections(timeout=MILVUS_TIMEOUT_SECONDS)
                except Exception as exc:
                    raise VectorStoreError("Milvus list_collections failed.") from exc
                print("Milvus collections:", collections)
        except ValidationError as exc:
            raise SystemExit("Check MILVUS_URI: an HTTP(S) Milvus server URI is required.") from exc
        except VectorStoreError as exc:
            raise SystemExit(
                "Milvus connectivity check failed. Check the Milvus service and MILVUS_URI."
            ) from exc


if __name__ == "__main__":
    main()
