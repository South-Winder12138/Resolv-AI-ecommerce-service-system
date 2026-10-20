"""Explicit Milvus connection ownership; no connections at import time."""

from collections.abc import Iterator
from contextlib import contextmanager

from pymilvus import MilvusClient

from resolv.core.config import MilvusSettings

MILVUS_TIMEOUT_SECONDS = 10.0


class VectorStoreError(RuntimeError):
    """A Milvus operation failed before producing a business result."""


def create_milvus_client(settings: MilvusSettings) -> MilvusClient:
    """Create a caller-owned client with a bounded connection timeout."""
    try:
        return MilvusClient(
            uri=settings.milvus_uri, timeout=MILVUS_TIMEOUT_SECONDS
        )
    except Exception as exc:
        raise VectorStoreError("Milvus connection failed.") from exc


@contextmanager
def milvus_client(settings: MilvusSettings) -> Iterator[MilvusClient]:
    """Close the client on exit, preserving any failure from the body."""
    client = create_milvus_client(settings)
    body_error: BaseException | None = None
    try:
        yield client
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            client.close()
        except Exception as exc:
            if body_error is None:
                raise VectorStoreError("Milvus close failed.") from exc
            body_error.add_note("Milvus client cleanup also failed.")
