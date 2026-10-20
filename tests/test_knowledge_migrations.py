"""Run the real knowledge Alembic environment without a MySQL dependency."""

import importlib
import io
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError

import resolv.core.config as config_module
import resolv.db.knowledge as knowledge_db
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from tests.knowledge_fixtures import chunk_record, document_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URI = "mysql+pymysql://app:private%40pw@127.0.0.1:3307/knowledge?charset=utf8mb4"


@pytest.fixture
def engine(tmp_path):
    resource = create_db_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    yield resource
    resource.dispose()


def _command():
    return importlib.import_module("alembic.command")


def _config(connection=None, buffer=None):
    config = importlib.import_module("alembic.config").Config(
        str(PROJECT_ROOT / "alembic.knowledge.ini"), output_buffer=buffer
    )
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def _upgrade(engine, revision="20261012_0002"):
    with engine.begin() as connection:
        _command().upgrade(_config(connection), revision)


def test_upgrade_creates_two_knowledge_tables_and_own_version(engine):
    _upgrade(engine)
    assert set(inspect(engine).get_table_names()) == {
        "knowledge_documents", "knowledge_chunks", "knowledge_alembic_version",
    }
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM knowledge_alembic_version")) == "20261012_0002"
    columns = {column["name"]: column for column in inspect(engine).get_columns("knowledge_chunks")}
    assert set(columns) == {
        "chunk_id", "document_id", "chunk_index", "source", "category", "questions",
        "answer", "section_path", "content_type", "text", "is_key_clause", "index_status",
        "index_hash", "indexed_hash", "indexed_collection", "attempt_count", "last_error",
        "created_at", "updated_at", "indexed_at",
    }
    assert columns["index_hash"]["nullable"] is False
    assert columns["indexed_hash"]["nullable"] is True
    assert "dense_vector" not in columns
    assert "score" not in columns


def test_revision_round_trips_complete_records_and_pending_defaults(engine):
    _upgrade(engine)
    factory = create_session_factory(engine)
    original = "原文😀" * 20000
    with factory.begin() as session:
        document = document_record(raw_text=original)
        document.chunks = [chunk_record()]
        session.add(document)
    with factory() as session:
        assert session.get(KnowledgeDocument, "d" * 64).raw_text == original
        chunk = session.get(KnowledgeChunkRecord, "c" * 64)
        assert chunk.category == "售后政策"
        assert chunk.questions == "软件商品退款"
        assert chunk.index_status == "pending"
        assert chunk.attempt_count == 0
        assert chunk.indexed_hash is None


@pytest.mark.parametrize("overrides", [
    {"index_status": "DONE"}, {"is_key_clause": 2}, {"attempt_count": -1},
    {"chunk_index": -1}, {"index_hash": "short"}, {"answer": ""},
    {"index_status": "done"},
    {"index_status": "done", "indexed_hash": "2" * 64},
])
def test_revision_enforces_chunk_invariants(engine, overrides):
    _upgrade(engine)
    factory = create_session_factory(engine)
    with pytest.raises(IntegrityError), factory.begin() as session:
        document = document_record()
        document.chunks = [chunk_record(**overrides)]
        session.add(document)


def test_revision_enforces_foreign_key_and_unique_document_index(engine):
    _upgrade(engine)
    factory = create_session_factory(engine)
    with pytest.raises(IntegrityError), factory.begin() as session:
        session.add(chunk_record())
    with pytest.raises(IntegrityError), factory.begin() as session:
        document = document_record()
        document.chunks = [chunk_record(), chunk_record(chunk_id="b" * 64)]
        session.add(document)


def test_second_upgrade_preserves_existing_rows(engine):
    _upgrade(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        document = document_record()
        document.chunks = [chunk_record()]
        session.add(document)
    _upgrade(engine)
    with factory() as session:
        assert session.get(KnowledgeChunkRecord, "c" * 64).index_hash == "1" * 64
        assert len(session.get(KnowledgeDocument, "d" * 64).chunks) == 1


def test_schema_matches_metadata_while_autogenerate_ignores_other_tables(engine):
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
        connection.execute(text("INSERT INTO unrelated VALUES (7)"))
        config = _config(connection)
        _command().upgrade(config, "head")
        _command().check(config)
        assert connection.scalar(text("SELECT id FROM unrelated")) == 7
    assert set(inspect(engine).get_table_names()) == {
        "orders", "shipments", "refunds", "tool_execution_receipts", "refund_applications", "refund_submission_receipts", "unrelated",
        "knowledge_documents", "knowledge_chunks", "knowledge_index_versions", "knowledge_chunk_indexes", "knowledge_alembic_version",
    }


def test_downgrade_only_removes_knowledge_tables_in_temporary_sqlite(engine):
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
        connection.execute(text("INSERT INTO unrelated VALUES (7)"))
        config = _config(connection)
        _command().upgrade(config, "head")
        _command().downgrade(config, "base")
        assert connection.scalar(text("SELECT id FROM unrelated")) == 7
    assert set(inspect(engine).get_table_names()) == {
        "orders", "shipments", "refunds", "tool_execution_receipts", "refund_applications", "refund_submission_receipts", "unrelated", "knowledge_alembic_version",
    }


def test_injected_connection_remains_caller_owned_and_needs_no_settings(engine, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Injected connection must not initialize settings/engine")
    monkeypatch.setattr(config_module, "KnowledgeDatabaseSettings", forbidden)
    monkeypatch.setattr(knowledge_db, "create_knowledge_engine", forbidden)
    monkeypatch.chdir(tmp_path)
    with engine.begin() as connection:
        _command().upgrade(_config(connection), "head")
        assert not connection.closed
        assert connection.in_transaction()
        assert connection.scalar(text("SELECT 1")) == 1


def test_offline_mysql_sql_does_not_create_engine_or_leak_percent_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline SQL must not create an engine")
    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", URI)
    monkeypatch.setattr(knowledge_db, "create_knowledge_engine", forbidden)
    buffer = io.StringIO()
    config = _config(buffer=buffer)
    _command().upgrade(config, "head", sql=True)
    sql = buffer.getvalue()
    assert "CREATE TABLE knowledge_documents" in sql
    assert "CREATE TABLE knowledge_chunks" in sql
    assert "LONGTEXT" in sql and "InnoDB" in sql and "utf8mb4_bin" in sql
    assert "private" not in sql and "mysql+pymysql" not in sql
    assert config.get_main_option("sqlalchemy.url") is None


@pytest.mark.parametrize("conflict", [False, True])
def test_online_environment_disposes_owned_engine_on_success_or_ddl_failure(engine, monkeypatch, conflict):
    from sqlalchemy import event

    if conflict:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE knowledge_documents (id INTEGER PRIMARY KEY)"))
    disposed = []
    created = []
    event.listen(engine, "engine_disposed", lambda _engine: disposed.append(True))
    def factory(uri, **options):
        created.append((uri, options))
        return engine
    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", URI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(knowledge_db, "create_db_engine", factory)
    if conflict:
        with pytest.raises(OperationalError):
            _command().upgrade(_config(), "head")
    else:
        _command().upgrade(_config(), "head")
    assert disposed == [True]
    assert created == [(URI, {
        "pool_pre_ping": True, "hide_parameters": True, "connect_args": {"connect_timeout": 10},
    })]


def test_mysql_status_change_uses_one_atomic_alter(monkeypatch):
    import re

    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", URI)
    buffer = io.StringIO()
    _command().upgrade(_config(buffer=buffer), "20261012_0001:20261012_0002", sql=True)
    alterations = re.findall(r"ALTER TABLE knowledge_chunks[^;]+;", buffer.getvalue())
    assert len(alterations) == 1
    assert "DROP CHECK ck_knowledge_chunks_status" in alterations[0]
    assert "ADD CONSTRAINT ck_knowledge_chunks_status CHECK" in alterations[0]


def test_head_adds_only_independent_version_tables_and_preserves_0002_data(engine):
    _upgrade(engine)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO knowledge_documents (document_id,source,content_type,raw_text,ingest_hash,created_at,updated_at) VALUES (:id,'source.md','policy','raw trailing ',:hash,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"),{'id':'a'*64,'hash':'b'*64})
        before=connection.execute(text('SELECT * FROM knowledge_documents')).all()
    old_columns=[c['name'] for c in inspect(engine).get_columns('knowledge_chunks')]
    _upgrade(engine,'head');_upgrade(engine,'head')
    assert set(inspect(engine).get_table_names())=={'knowledge_documents','knowledge_chunks','knowledge_index_versions','knowledge_chunk_indexes','knowledge_alembic_version'}
    assert [c['name'] for c in inspect(engine).get_columns('knowledge_chunks')]==old_columns
    with engine.connect() as connection:
        assert connection.execute(text('SELECT * FROM knowledge_documents')).all()==before
        assert connection.scalar(text('SELECT version_num FROM knowledge_alembic_version'))=='20261014_0003'
