"""Knowledge schema invariants on a real isolated SQLite database."""

import importlib
import re
import subprocess
import sys
from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from resolv.db.base import Base
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db
from tests.knowledge_fixtures import chunk_record, document_record

INDEXED_AT = datetime(2026, 10, 12, tzinfo=UTC).replace(tzinfo=None)


@pytest.fixture
def knowledge_factory(tmp_path):
    base = importlib.import_module("resolv.db.knowledge_base").KnowledgeBase
    importlib.import_module("resolv.models.knowledge")
    importlib.import_module("resolv.models.knowledge_hybrid")
    engine = create_db_engine(f"sqlite:///{tmp_path / 'knowledge.db'}")
    base.metadata.create_all(engine)
    yield create_session_factory(engine)
    engine.dispose()


def test_knowledge_metadata_and_business_init_remain_separate(knowledge_factory):
    base = importlib.import_module("resolv.db.knowledge_base").KnowledgeBase
    assert base.metadata is not Base.metadata
    assert set(base.metadata.tables) == {"knowledge_documents", "knowledge_chunks", "knowledge_index_versions", "knowledge_chunk_indexes"}
    assert set(inspect(knowledge_factory.kw["bind"]).get_table_names()) == {
        "knowledge_documents", "knowledge_chunks", "knowledge_index_versions", "knowledge_chunk_indexes",
    }
    business = create_db_engine("sqlite:///:memory:")
    try:
        init_db(business)
        assert set(inspect(business).get_table_names()) == {"orders", "shipments", "refunds", "tool_execution_receipts", "refund_applications", "refund_submission_receipts"}
    finally:
        business.dispose()


def test_complete_metadata_and_pending_defaults_round_trip(knowledge_factory):
    models = importlib.import_module("resolv.models.knowledge")
    before = datetime.now(UTC).replace(tzinfo=None)
    document = document_record()
    document.chunks = [chunk_record()]
    with knowledge_factory.begin() as session:
        session.add(document)
    with knowledge_factory() as session:
        stored = session.get(models.KnowledgeDocument, "d" * 64)
        assert stored.raw_text == "# 售后政策\n## 软件商品退款\n软件商品一旦激活，不支持无理由退款。"
        chunk = stored.chunks[0]
        assert (chunk.category, chunk.questions, chunk.answer, chunk.section_path) == (
            "售后政策", "软件商品退款", "软件商品一旦激活，不支持无理由退款。",
            "售后政策 > 软件商品退款",
        )
        assert chunk.text == "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。"
        assert chunk.source == stored.source == "knowledge/refund.md"
        assert chunk.content_type == stored.content_type == "refund_policy"
        assert chunk.is_key_clause == 1
        assert (chunk.index_status, chunk.attempt_count) == ("pending", 0)
        assert (chunk.indexed_hash, chunk.indexed_collection, chunk.indexed_at, chunk.last_error) == (None, None, None, None)
        assert before <= stored.created_at <= datetime.now(UTC).replace(tzinfo=None)
        assert stored.created_at.tzinfo is None
        assert chunk.created_at.tzinfo is None


def test_empty_document_and_unheaded_chunk_are_valid(knowledge_factory):
    models = importlib.import_module("resolv.models.knowledge")
    with knowledge_factory.begin() as session:
        document = document_record(raw_text="")
        session.add(document)
    with knowledge_factory.begin() as session:
        session.add(chunk_record(category="", questions="", section_path="", answer="正文", text="正文"))
    with knowledge_factory() as session:
        assert session.get(models.KnowledgeDocument, "d" * 64).raw_text == ""
        assert session.get(models.KnowledgeChunkRecord, "c" * 64).questions == ""


def test_utc_updated_at_changes_on_record_update(knowledge_factory):
    models = importlib.import_module("resolv.models.knowledge")
    old_time = datetime(2000, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    with knowledge_factory.begin() as session:
        document = document_record(created_at=old_time, updated_at=old_time)
        document.chunks = [chunk_record(created_at=old_time, updated_at=old_time)]
        session.add(document)
    with knowledge_factory.begin() as session:
        chunk = session.get(models.KnowledgeChunkRecord, "c" * 64)
        chunk.attempt_count = 1
    with knowledge_factory() as session:
        chunk = session.get(models.KnowledgeChunkRecord, "c" * 64)
        assert chunk.created_at == old_time
        assert chunk.updated_at > old_time
        assert chunk.updated_at.tzinfo is None


@pytest.mark.parametrize("overrides", [
    {"index_status": "processing"}, {"index_status": "DONE"},
    {"is_key_clause": -1}, {"is_key_clause": 2}, {"chunk_index": -1},
    {"attempt_count": -1}, {"chunk_id": "short"}, {"index_hash": ""},
    {"indexed_hash": "short"}, {"source": ""}, {"content_type": "  "},
    {"answer": ""}, {"text": ""}, {"index_status": "done"},
    {"index_status": "done", "indexed_hash": "2" * 64,
     "indexed_collection": "resolv_knowledge_v2", "indexed_at": INDEXED_AT},
    {"index_status": "done", "indexed_hash": "1" * 64,
     "indexed_collection": "", "indexed_at": INDEXED_AT},
    {"index_status": "done", "indexed_hash": "1" * 64,
     "indexed_collection": "resolv_knowledge_v2"},
    {"index_status": "done", "indexed_collection": "resolv_knowledge_v2",
     "indexed_at": INDEXED_AT},
])
def test_database_rejects_invalid_chunk_state(knowledge_factory, overrides):
    models = importlib.import_module("resolv.models.knowledge")
    with pytest.raises(IntegrityError), knowledge_factory.begin() as session:
        document = document_record()
        document.chunks = [chunk_record(**overrides)]
        session.add(document)
    with knowledge_factory() as session:
        assert session.scalar(select(models.KnowledgeDocument)) is None
        assert session.scalar(select(models.KnowledgeChunkRecord)) is None


@pytest.mark.parametrize("overrides", [
    {"document_id": "short"}, {"ingest_hash": ""}, {"source": ""}, {"content_type": " "},
])
def test_database_rejects_invalid_document_identity_or_type(knowledge_factory, overrides):
    with pytest.raises(IntegrityError), knowledge_factory.begin() as session:
        session.add(document_record(**overrides))


def test_done_requires_matching_confirmation_and_round_trips(knowledge_factory):
    models = importlib.import_module("resolv.models.knowledge")
    with knowledge_factory.begin() as session:
        document = document_record()
        document.chunks = [chunk_record(
            index_status="done", indexed_hash="1" * 64,
            indexed_collection="resolv_knowledge_v2", indexed_at=INDEXED_AT,
            attempt_count=1,
        )]
        session.add(document)
    with knowledge_factory() as session:
        chunk = session.get(models.KnowledgeChunkRecord, "c" * 64)
        assert chunk.index_status == "done"
        assert chunk.index_hash == chunk.indexed_hash == "1" * 64


def test_duplicate_document_chunk_index_is_rejected(knowledge_factory):
    with pytest.raises(IntegrityError), knowledge_factory.begin() as session:
        document = document_record()
        document.chunks = [chunk_record(), chunk_record(chunk_id="b" * 64)]
        session.add(document)


def test_chunk_requires_existing_document(knowledge_factory):
    with pytest.raises(IntegrityError), knowledge_factory.begin() as session:
        session.add(chunk_record())


def test_deleting_document_with_chunks_is_restricted(knowledge_factory):
    models = importlib.import_module("resolv.models.knowledge")
    with knowledge_factory.begin() as session:
        document = document_record()
        document.chunks = [chunk_record()]
        session.add(document)
    with pytest.raises(IntegrityError), knowledge_factory.begin() as session:
        document = session.get(models.KnowledgeDocument, "d" * 64)
        assert len(document.chunks) == 1
        session.delete(document)


def test_mysql_ddl_stores_long_originals_with_exact_comparison():
    models = importlib.import_module("resolv.models.knowledge")
    ddl = str(CreateTable(models.KnowledgeDocument.__table__).compile(dialect=mysql.dialect()))
    assert "LONGTEXT" in ddl
    assert "InnoDB" in ddl
    assert "utf8mb4_bin" in ddl
    assert "utf8mb4" in ddl


def test_mysql_text_defaults_are_expressions_not_literals():
    models = importlib.import_module("resolv.models.knowledge")
    ddl = str(CreateTable(models.KnowledgeChunkRecord.__table__).compile(dialect=mysql.dialect()))
    assert not re.search(r"\bTEXT\b[^\n]*DEFAULT\s+''", ddl, re.IGNORECASE)


def test_importing_models_creates_no_engine_session_or_external_resources(tmp_path):
    result = subprocess.run([sys.executable, "-c", """
import sqlalchemy, sqlalchemy.orm, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Model import must not initialize resources")
sqlalchemy.create_engine = forbidden
sqlalchemy.orm.Session = forbidden
from resolv.models.knowledge import KnowledgeDocument, KnowledgeChunkRecord
from resolv.db.knowledge_base import KnowledgeBase
assert set(KnowledgeBase.metadata.tables) == {"knowledge_documents", "knowledge_chunks"}
assert not {"torch", "FlagEmbedding", "pymilvus", "langchain_openai"} & sys.modules.keys()
"""], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_status_is_exact_even_with_mysql_pad_space_comparison(tmp_path):
    from sqlalchemy import MetaData, String, event

    importlib.import_module("resolv.models.knowledge")
    base = importlib.import_module("resolv.db.knowledge_base").KnowledgeBase
    # SQLite's normal binary comparison does not expose MySQL PAD SPACE rules.
    # Keep the production constraints and emulate only that comparison boundary.
    schema = MetaData()
    for table in base.metadata.sorted_tables:
        table.to_metadata(schema)
    schema.tables["knowledge_chunks"].c.index_status.type = String(
        16, collation="mysql_pad_binary"
    )
    engine = create_db_engine(f"sqlite:///{tmp_path / 'pad-space.db'}")
    def install_comparison(connection, _record):
        def compare(left, right):
            left, right = left.rstrip(" "), right.rstrip(" ")
            return (left > right) - (left < right)
        connection.create_collation("mysql_pad_binary", compare)
    event.listen(engine, "connect", install_comparison)
    try:
        schema.create_all(engine)
        factory = create_session_factory(engine)
        for status in ("pending ", "done ", "done   "):
            with pytest.raises(IntegrityError), factory.begin() as session:
                document = document_record()
                document.chunks = [chunk_record(
                    index_status=status, indexed_hash="1" * 64,
                    indexed_collection="resolv_knowledge_v2", indexed_at=INDEXED_AT,
                )]
                session.add(document)
    finally:
        engine.dispose()
