from hashlib import sha256

import pytest
from sqlalchemy import text

from resolv.db.engine import create_db_engine
from resolv.db.knowledge_hybrid import consistent_read_factory, hybrid_index_lock
from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from tests.test_knowledge_index_lock import Connection, Result, engine


class HybridConnection(Connection):
    def execute(self, statement, parameters):
        assert parameters["name"] == "mhhybrid_" + sha256(b"v3-one").hexdigest()[:48]
        sql = str(statement)
        self.calls.append(sql)
        value = (
            self.acquired
            if "GET_LOCK" in sql
            else self.owned
            if "IS_USED_LOCK" in sql
            else self.released
        )
        if isinstance(value, Exception):
            raise value
        return Result(value)


def test_version_lock_owned_and_released_without_cross_version_lock():
    c = HybridConnection()
    with hybrid_index_lock(engine(c), "v3-one") as lock:
        lock.assert_owned()
    assert c.closed and not c.invalidated
    assert len(c.calls) == 6
    with pytest.raises(HybridRetrievalError):
        lock.assert_owned()


@pytest.mark.parametrize(
    "options",
    [
        {"acquired": 0},
        {"acquired": None},
        {"acquired": OSError("private")},
        {"owned": (18, 17)},
        {"released": 0},
        {"released": OSError("private")},
    ],
)
def test_failed_lock_never_succeeds(options):
    c = HybridConnection(**options)
    with (
        pytest.raises(HybridRetrievalError),
        hybrid_index_lock(engine(c), "v3-one") as lock,
    ):
        lock.assert_owned()
    assert c.closed
    if isinstance(options.get("acquired"), Exception) or "released" in options:
        assert c.invalidated


def test_consistent_factory_explicit_isolation_without_mutating_parent_engine(tmp_path):
    e = create_db_engine(f"sqlite:///{tmp_path / 'read.db'}")
    try:
        factory = consistent_read_factory(e)
        assert (
            factory.kw["bind"].get_execution_options()["isolation_level"]
            == "SERIALIZABLE"
        )
        assert "isolation_level" not in e.get_execution_options()
        with factory() as session:
            assert session.scalar(text("SELECT 1")) == 1
            assert session.connection().connection.driver_connection.in_transaction
    finally:
        e.dispose()
