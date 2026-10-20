"""One non-reconnecting MySQL session owns both the named lock and saver.

Importing creates no resources. Setup is an explicit administrative operation.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from threading import RLock

import pymysql
from sqlalchemy.engine import make_url

from resolv.core.config import ConversationDatabaseSettings


class CheckpointError(RuntimeError):
    def __init__(self, code: str = "checkpoint_failed"):
        self.code = code
        super().__init__(code)


def _connect(**kwargs):
    return pymysql.connect(**kwargs)


def _open_connection(settings: ConversationDatabaseSettings):
    url = make_url(settings.conversation_database_url.get_secret_value())
    try:
        return _connect(
            host=url.host, port=url.port or 3306, user=url.username,
            password=url.password, database=url.database, charset="utf8mb4",
            autocommit=True, connect_timeout=10, read_timeout=30, write_timeout=30,
        )
    except Exception:  # noqa: BLE001 — redact driver exceptions at the secret boundary
        raise CheckpointError() from None


def _create_saver(connection, session):
    # Gate installs the candidate into a uv overlay; normal imports remain offline.
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

    class OwnedSaver(PyMySQLSaver):
        @contextmanager
        def _cursor(self, *, pipeline=False):
            with session.serialized():
                session.assert_owned()
                try:
                    with super()._cursor(pipeline=pipeline) as cursor:
                        yield cursor
                except (pymysql.InterfaceError, pymysql.OperationalError):
                    session._lost = True
                    raise CheckpointError() from None

    saver = OwnedSaver(connection)
    saver.lock = session._lock
    return saver


class CheckpointSession:
    def __init__(self, connection, lock_name: str):
        self._connection = connection
        self._lock_name = lock_name
        self._lock = RLock()
        self._active = True
        self._lost = False
        self.saver = None

    @contextmanager
    def serialized(self):
        with self._lock:
            if not self._active or self._lost:
                raise CheckpointError()
            yield

    def assert_owned(self) -> None:
        with self.serialized():
            try:
                self._connection.ping(reconnect=False)
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT IS_USED_LOCK(%s), CONNECTION_ID()", (self._lock_name,)
                    )
                    owner, connection_id = cursor.fetchone()
                if owner is None or owner != connection_id:
                    raise CheckpointError()
            except Exception:  # noqa: BLE001 — ownership is uncertain; fail closed
                self._lost = True
                raise CheckpointError() from None


@contextmanager
def checkpoint_session(
    settings: ConversationDatabaseSettings, *, conversation_id: str
) -> Iterator[CheckpointSession]:
    if not conversation_id:
        raise ValueError("conversation_id is required")
    database = make_url(settings.conversation_database_url.get_secret_value()).database
    lock_name = "resolv:" + sha256(f"{database}\0{conversation_id}".encode()).hexdigest()[:56]
    connection = _open_connection(settings)
    session = CheckpointSession(connection, lock_name)
    acquired = False
    body_error = None
    try:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT GET_LOCK(%s, 0)", (lock_name,))
                result = cursor.fetchone()[0]
            if result == 0:
                raise CheckpointError("conversation_busy")
            if result != 1:
                raise CheckpointError()
            acquired = True
            session.assert_owned()
            session.saver = _create_saver(connection, session)
        except CheckpointError:
            raise
        except Exception:  # noqa: BLE001 — factories can include secret connection inputs
            raise CheckpointError() from None
        yield session
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        # Graph's synchronous durability waits for saver work before exiting here.
        # Taking the same lock also waits for any in-flight cursor to finish.
        with session._lock:
            session._active = False
            cleanup_failed = False
            try:
                if acquired and not session._lost:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                        cleanup_failed = cursor.fetchone()[0] != 1
            except Exception:  # noqa: BLE001 — always close after uncertain release
                cleanup_failed = True
            finally:
                try:
                    connection.close()
                except Exception:  # noqa: BLE001 — report cleanup without driver secrets
                    cleanup_failed = True
            if cleanup_failed:
                if body_error is None:
                    raise CheckpointError("checkpoint_cleanup_failed") from None
                body_error.add_note("Checkpoint connection cleanup also failed.")


def setup_checkpoint_tables(settings: ConversationDatabaseSettings) -> None:
    # setup has its own database-wide lock, separate from application execution.
    with checkpoint_session(settings, conversation_id="__checkpoint_setup__") as session:
        session.saver.setup()
