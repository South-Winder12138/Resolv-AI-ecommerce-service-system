import importlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext as _no_exception
from threading import Event

import pytest

from resolv.core import config

URL = "mysql+pymysql://app:private_pw@127.0.0.1:3307/resolv_conversations?charset=utf8mb4"


class Connection:
    def __init__(self, acquired=1):
        self.acquired = acquired
        self.closed = False
        self.broken = False
        self.calls = []
        self.owned = (17, 17)
        self.released = 1

    def ping(self, *, reconnect):
        self.calls.append(("ping", reconnect))
        assert reconnect is False
        if self.broken:
            raise OSError("private_pw connection lost")

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params):
        self.calls.append((sql, params))
        assert len(params[0]) <= 64
        assert "private_pw" not in params[0]
        if "GET_LOCK" in sql:
            assert ", 0)" in sql
            self.result = (self.acquired,)
        elif "IS_USED_LOCK" in sql:
            self.result = self.owned
        elif "RELEASE_LOCK" in sql:
            self.result = (self.released,)
        else:
            pytest.fail(f"unexpected query: {sql}")

    def fetchone(self):
        return self.result

    def close(self):
        self.closed = True


@pytest.fixture
def boundary(monkeypatch):
    module = importlib.import_module("resolv.db.checkpoints")
    connection = Connection()
    created = []

    def connect(**kwargs):
        assert kwargs["autocommit"] is True
        assert kwargs["charset"] == "utf8mb4"
        assert kwargs["password"] == "private_pw"
        return connection

    def saver(conn, guard):
        assert conn is connection
        created.append(True)
        return guard

    monkeypatch.setattr(module, "_connect", connect)
    monkeypatch.setattr(module, "_create_saver", saver)
    settings = config.ConversationDatabaseSettings(_env_file=None, conversation_database_url=URL)
    return module, connection, created, settings


def test_import_does_not_connect(monkeypatch):
    import pymysql
    monkeypatch.setattr(pymysql, "connect", lambda **kw: pytest.fail("import connected"))
    original = importlib.import_module("resolv.db.checkpoints")
    spec = importlib.util.spec_from_file_location("checkpoint_import_probe", original.__file__)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)


@pytest.mark.parametrize("value", [0, None])
def test_lock_busy_does_not_create_saver(boundary, value):
    module, connection, created, settings = boundary
    connection.acquired = value
    with pytest.raises(module.CheckpointError), module.checkpoint_session(settings, conversation_id="a"):
        pytest.fail("failed acquisition yielded execution")
    assert not created
    assert connection.closed


def test_connection_loss_never_reconnects(boundary):
    module, connection, _, settings = boundary
    with module.checkpoint_session(settings, conversation_id="a") as session:
        session.assert_owned()
        connection.broken = True
        with pytest.raises(module.CheckpointError) as raised:
            session.assert_owned()
        assert "private_pw" not in str(raised.value)
        connection.broken = False
        with pytest.raises(module.CheckpointError):
            session.assert_owned()
    assert connection.closed
    assert all(call[1] is False for call in connection.calls if call[0] == "ping")


@pytest.mark.parametrize("failure", [False, True])
def test_context_closes_on_interrupt_and_exception(boundary, failure):
    module, connection, _, settings = boundary
    with (
        pytest.raises(ValueError) if failure else _no_exception(),
        module.checkpoint_session(settings, conversation_id="a") as session,
    ):
        session.assert_owned()
        if failure:
            raise ValueError("node stopped")
    assert connection.closed
    assert any("RELEASE_LOCK" in call[0] for call in connection.calls)
    with pytest.raises(module.CheckpointError):
        session.assert_owned()


def test_ownership_checks_wait_for_saver_cursor(boundary):
    module, _, _, settings = boundary
    entered, completed = Event(), Event()
    with module.checkpoint_session(settings, conversation_id="a") as session:
        def check():
            entered.set()
            session.assert_owned()
            completed.set()

        with ThreadPoolExecutor() as pool:
            with session.serialized():
                future = pool.submit(check)
                assert entered.wait(1)
                assert not completed.wait(0.05)
            future.result(timeout=1)
    assert completed.is_set()


def test_lost_named_lock_fences_session(boundary):
    module, connection, _, settings = boundary
    with module.checkpoint_session(settings, conversation_id="a") as session:
        connection.owned = (18, 17)
        with pytest.raises(module.CheckpointError):
            session.assert_owned()
