"""Explicit, connection-scoped MySQL indexing lock; no resource initialization."""

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256

from sqlalchemy import Connection, Engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from resolv.schemas.hybrid_knowledge import HybridRetrievalError


class HybridIndexLock:
    def __init__(self, connection: Connection, name: str) -> None:
        self._connection = connection
        self._name = name
        self._active = True

    def assert_owned(self) -> None:
        if not self._active or self._connection.closed:
            raise HybridRetrievalError("Knowledge indexing lock is no longer active.")
        try:
            owner, connection_id = self._connection.execute(
                text("SELECT IS_USED_LOCK(:name), CONNECTION_ID()"),
                {"name": self._name},
            ).one()
            self._connection.commit()
        except Exception as exc:
            raise HybridRetrievalError(
                "Knowledge indexing lock ownership check failed."
            ) from exc
        if owner is None or owner != connection_id:
            raise HybridRetrievalError("Knowledge indexing lock ownership was lost.")


@contextmanager
def hybrid_index_lock(engine: Engine, version_id: str) -> Iterator[HybridIndexLock]:
    if engine.dialect.name != "mysql":
        raise ValueError("Knowledge indexing requires a MySQL named lock.")
    name = "mhhybrid_" + sha256(version_id.encode("utf-8")).hexdigest()[:48]
    connection: Connection | None = None
    lock: HybridIndexLock | None = None
    acquired = False
    uncertain_acquisition = False
    primary_error: BaseException | None = None
    try:
        try:
            connection = engine.connect()
            result = connection.execute(
                text("SELECT GET_LOCK(:name, 0)"), {"name": name}
            ).scalar()
            acquired = type(result) is int and result == 1
            connection.commit()
        except Exception as exc:
            uncertain_acquisition = True
            raise HybridRetrievalError(
                "Knowledge indexing lock acquisition failed."
            ) from exc
        if not acquired:
            raise HybridRetrievalError("Knowledge indexing lock is unavailable.")
        lock = HybridIndexLock(connection, name)
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
                    result = connection.execute(
                        text("SELECT RELEASE_LOCK(:name)"), {"name": name}
                    ).scalar()
                    connection.commit()
                    if type(result) is not int or result != 1:
                        raise HybridRetrievalError(
                            "Knowledge indexing lock release failed."
                        )
                except Exception as exc:  # noqa: BLE001 - cleanup must preserve the primary failure
                    cleanup_error = exc
                    if primary_error is not None:
                        primary_error.add_note(
                            "Knowledge indexing lock release cleanup also failed."
                        )
            if cleanup_error is not None or uncertain_acquisition:
                # close() normally returns a DBAPI connection to the pool. If
                # release/acquisition was uncertain it could retain GET_LOCK,
                # so discard that physical connection before returning it.
                try:
                    connection.invalidate()
                except Exception as exc:  # noqa: BLE001 - cleanup must preserve the primary failure
                    cleanup_error = cleanup_error or exc
                    if primary_error is not None:
                        primary_error.add_note(
                            "Knowledge indexing lock connection invalidation also failed."
                        )
            try:
                connection.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must preserve the primary failure
                cleanup_error = cleanup_error or exc
                if primary_error is not None:
                    primary_error.add_note(
                        "Knowledge indexing lock connection cleanup also failed."
                    )
            if cleanup_error is not None and primary_error is None:
                raise HybridRetrievalError(
                    "Knowledge indexing lock cleanup failed."
                ) from cleanup_error


class HybridReadSession(Session):
    """Only new read sessions opt into explicit SQLite read transactions."""


@event.listens_for(HybridReadSession, "after_begin")
def _sqlite_read_begin(session, transaction, connection):
    if (
        connection.dialect.name == "sqlite"
        and not connection.connection.driver_connection.in_transaction
    ):
        connection.exec_driver_sql("BEGIN")


def consistent_read_factory(engine: Engine) -> sessionmaker[Session]:
    if engine.dialect.name not in ("mysql", "sqlite"):
        raise HybridRetrievalError("hybrid_read_dialect_unsupported")
    isolated = engine.execution_options(
        isolation_level="REPEATABLE READ"
        if engine.dialect.name == "mysql"
        else "SERIALIZABLE"
    )
    return sessionmaker(bind=isolated, class_=HybridReadSession, expire_on_commit=False)
