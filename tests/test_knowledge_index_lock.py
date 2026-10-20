"""The dedicated MySQL lock must never silently allow concurrent indexing."""

import importlib
from types import SimpleNamespace

import pytest


class Result:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value

    def one(self):
        return self.value


class Connection:
    def __init__(self, *, acquired=1, owned=(17, 17), released=1):
        self.acquired, self.owned, self.released = acquired, owned, released
        self.calls = []
        self.closed = False
        self.invalidated = False
        self.close_error = None

    def execute(self, statement, parameters):
        sql = str(statement)
        assert parameters == {"name": "resolv_knowledge_index_v2"}
        self.calls.append(sql)
        if "GET_LOCK" in sql:
            assert ", 0)" in sql
            value = self.acquired
        elif "IS_USED_LOCK" in sql:
            assert "CONNECTION_ID()" in sql
            value = self.owned
        elif "RELEASE_LOCK" in sql:
            value = self.released
        else:
            pytest.fail(f"Unexpected lock SQL: {sql}")
        if isinstance(value, Exception):
            raise value
        return Result(value)

    def commit(self):
        self.calls.append("commit")

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error

    def invalidate(self):
        self.invalidated = True


def engine(connection, dialect="mysql"):
    return SimpleNamespace(dialect=SimpleNamespace(name=dialect), connect=lambda: connection)


def lock_context(connection):
    return importlib.import_module("resolv.db.knowledge_index").knowledge_index_lock(engine(connection))


def test_lock_owns_dedicated_connection_and_ends_every_read_transaction():
    connection = Connection()
    with lock_context(connection) as lock:
        lock.assert_owned()
        assert not connection.closed
    assert connection.closed
    assert len(connection.calls) == 6
    assert all(connection.calls[i] == "commit" for i in (1, 3, 5))
    assert "GET_LOCK" in connection.calls[0]
    assert "IS_USED_LOCK" in connection.calls[2]
    assert "RELEASE_LOCK" in connection.calls[4]


@pytest.mark.parametrize("acquired", [0, None])
def test_contention_or_null_never_yields_an_indexing_lock(acquired):
    connection = Connection(acquired=acquired)
    with pytest.raises(RuntimeError), lock_context(connection):
        pytest.fail("A failed lock acquisition allowed indexing")
    assert connection.closed
    assert not any("RELEASE_LOCK" in sql for sql in connection.calls)


@pytest.mark.parametrize("owned", [(None, 17), (18, 17)])
def test_lost_lock_prevents_confirmation(owned):
    connection = Connection(owned=owned)
    with pytest.raises(RuntimeError), lock_context(connection) as lock:
        lock.assert_owned()
    assert connection.closed


@pytest.mark.parametrize("released", [0, None, OSError("private release failure")])
def test_failed_release_is_reported_after_otherwise_successful_work(released):
    connection = Connection(released=released)
    with pytest.raises(RuntimeError, match="cleanup|release"), lock_context(connection):
        pass
    assert connection.closed
    assert connection.invalidated, "Failed release returned a possibly locked connection to the pool"


def test_cleanup_failures_do_not_mask_primary_failure():
    connection = Connection(released=OSError("private release failure"))
    connection.close_error = OSError("private close failure")
    failure = ValueError("primary indexing error")
    with pytest.raises(ValueError) as raised, lock_context(connection):
        raise failure
    assert raised.value is failure
    assert len(raised.value.__notes__) == 2
    assert connection.closed


def test_closed_or_broken_connection_cannot_claim_ownership():
    connection = Connection(owned=OSError("connection lost"))
    with pytest.raises(RuntimeError) as raised, lock_context(connection) as lock:
        lock.assert_owned()
    assert isinstance(raised.value.__cause__, OSError)
    assert connection.closed


def test_production_lock_does_not_bypass_mysql_for_sqlite():
    module = importlib.import_module("resolv.db.knowledge_index")
    connection = Connection()
    with pytest.raises(ValueError), module.knowledge_index_lock(engine(connection, "sqlite")):
        pytest.fail("SQLite has no MySQL named-lock protection")
    assert connection.calls == []


def test_uncertain_acquisition_discards_connection_instead_of_returning_it_to_pool():
    connection = Connection(acquired=OSError("response lost after GET_LOCK"))
    with pytest.raises(RuntimeError), lock_context(connection):
        pytest.fail("Uncertain acquisition yielded a lock")
    assert connection.closed
    assert connection.invalidated


def test_failed_close_is_reported_after_successful_release():
    connection = Connection()
    connection.close_error = OSError("private close error")
    with pytest.raises(RuntimeError, match="cleanup"), lock_context(connection):
        pass
    assert any("RELEASE_LOCK" in sql for sql in connection.calls)


def test_lock_object_cannot_be_reused_after_context_exit():
    connection = Connection()
    with lock_context(connection) as lock:
        lock.assert_owned()
    before = list(connection.calls)
    with pytest.raises(RuntimeError):
        lock.assert_owned()
    assert connection.calls == before
