"""Explicit, connection-scoped MySQL indexing lock; no resource initialization."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, Engine, text

from resolv.db.knowledge import KnowledgeStoreError

_LOCK_NAME = "resolv_knowledge_index_v2"


class KnowledgeIndexLock:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._active = True

    def assert_owned(self) -> None:
        if not self._active or self._connection.closed:
            raise KnowledgeStoreError("Knowledge indexing lock is no longer active.")
        try:
            owner, connection_id = self._connection.execute(text(
                "SELECT IS_USED_LOCK(:name), CONNECTION_ID()"
            ), {"name": _LOCK_NAME}).one()
            self._connection.commit()
        except Exception as exc:
            raise KnowledgeStoreError("Knowledge indexing lock ownership check failed.") from exc
        if owner is None or owner != connection_id:
            raise KnowledgeStoreError("Knowledge indexing lock ownership was lost.")


@contextmanager
def knowledge_index_lock(engine: Engine) -> Iterator[KnowledgeIndexLock]:
    if engine.dialect.name != "mysql":
        raise ValueError("Knowledge indexing requires a MySQL named lock.")
    connection: Connection | None = None
    lock: KnowledgeIndexLock | None = None
    acquired = False
    uncertain_acquisition = False
    primary_error: BaseException | None = None
    try:
        try:
            connection = engine.connect()
            result = connection.execute(text(
                "SELECT GET_LOCK(:name, 0)"
            ), {"name": _LOCK_NAME}).scalar()
            acquired = type(result) is int and result == 1
            connection.commit()
        except Exception as exc:
            uncertain_acquisition = True
            raise KnowledgeStoreError("Knowledge indexing lock acquisition failed.") from exc
        if not acquired:
            raise KnowledgeStoreError("Knowledge indexing lock is unavailable.")
        lock = KnowledgeIndexLock(connection)
        yield lock
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if lock is not None:
            lock._active = False
        cleanup_error: Exception | None = None
        if connection is not None:
            if acquired:
                try:
                    result = connection.execute(text(
                        "SELECT RELEASE_LOCK(:name)"
                    ), {"name": _LOCK_NAME}).scalar()
                    connection.commit()
                    if type(result) is not int or result != 1:
                        raise KnowledgeStoreError("Knowledge indexing lock release failed.")
                except Exception as exc:  # noqa: BLE001 - cleanup must preserve the primary failure
                    cleanup_error = exc
                    if primary_error is not None:
                        primary_error.add_note("Knowledge indexing lock release cleanup also failed.")
            if cleanup_error is not None or uncertain_acquisition:
                # close() normally returns a DBAPI connection to the pool. If
                # release/acquisition was uncertain it could retain GET_LOCK,
                # so discard that physical connection before returning it.
                try:
                    connection.invalidate()
                except Exception as exc:  # noqa: BLE001 - cleanup must preserve the primary failure
                    cleanup_error = cleanup_error or exc
                    if primary_error is not None:
                        primary_error.add_note("Knowledge indexing lock connection invalidation also failed.")
            try:
                connection.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must preserve the primary failure
                cleanup_error = cleanup_error or exc
                if primary_error is not None:
                    primary_error.add_note("Knowledge indexing lock connection cleanup also failed.")
            if cleanup_error is not None and primary_error is None:
                raise KnowledgeStoreError("Knowledge indexing lock cleanup failed.") from cleanup_error
