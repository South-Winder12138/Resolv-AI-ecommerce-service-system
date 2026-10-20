"""Stage real prepared knowledge into isolated SQL, including fault paths."""
import importlib
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.knowledge import KnowledgeStoreError
from resolv.db.knowledge_base import KnowledgeBase
from resolv.knowledge import load_document
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.schemas.knowledge import KnowledgeFileConfig, KnowledgeIndexProfile

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RAW = "# 售后政策\n\n## 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。\n"


@pytest.fixture
def factory(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'staging.db'}")
    KnowledgeBase.metadata.create_all(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "refund.md"
    path.write_text(RAW, encoding="utf-8")
    return load_document(path, source_root=tmp_path)


def _module():
    return importlib.import_module("resolv.services.knowledge")


def _stage(factory, document, config=None, profile=None, **options):
    return _module().KnowledgeIngestionService(factory).stage_document(
        document, config or KnowledgeFileConfig(
            content_type="refund_policy", key_clause_keywords=("一旦激活", "不支持")
        ), profile or KnowledgeIndexProfile(model_revision=REVISION), **options,
    )


def _state(factory):
    with factory() as session:
        documents = session.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.document_id)).all()
        chunks = session.scalars(select(KnowledgeChunkRecord).order_by(KnowledgeChunkRecord.chunk_index)).all()
        return (
            [{column.name: getattr(row, column.name) for column in KnowledgeDocument.__table__.columns}
             for row in documents],
            [{column.name: getattr(row, column.name) for column in KnowledgeChunkRecord.__table__.columns}
             for row in chunks],
        )


def test_staging_commits_complete_authoritative_document_and_pending_chunk(factory, source):
    result = _stage(factory, source)
    assert (result.outcome, result.chunk_count, result.pending_count, result.done_count) == ("created", 1, 1, 0)
    docs, chunks = _state(factory)
    assert len(docs) == len(chunks) == 1
    assert docs[0]["document_id"] == source.document_id == result.document_id
    assert docs[0]["raw_text"] == RAW
    assert docs[0]["source"] == "refund.md" and docs[0]["content_type"] == "refund_policy"
    assert len(docs[0]["ingest_hash"]) == 64 and docs[0]["ingest_hash"] == result.ingest_hash
    row = chunks[0]
    assert [row[name] for name in (
        "category", "questions", "answer", "section_path", "content_type", "is_key_clause", "text",
    )] == [
        "售后政策", "软件商品退款", "软件商品一旦激活，不支持无理由退款。",
        "售后政策 > 软件商品退款", "refund_policy", 1,
        "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
    ]
    assert row["index_status"] == "pending" and row["attempt_count"] == 0
    assert row["indexed_hash"] is row["indexed_collection"] is row["indexed_at"] is row["last_error"] is None
    assert len(row["index_hash"]) == 64 and row["index_hash"] != result.ingest_hash
    assert row["created_at"].tzinfo is None


def test_staging_txt_keeps_empty_heading_fields(factory, tmp_path):
    path = tmp_path / "terms.txt"
    path.write_text("正文条款", encoding="utf-8")
    result = _stage(factory, load_document(path, source_root=tmp_path))
    assert result.chunk_count == 1
    assert [_state(factory)[1][0][key] for key in ("category", "questions", "section_path")] == ["", "", ""]


@pytest.mark.parametrize("body", ["", "# Empty\n"])
def test_empty_document_commits_document_with_zero_chunks(factory, source, body):
    empty = source.model_copy(update={"text": body})
    result = _stage(factory, empty)
    assert (result.chunk_count, result.pending_count, result.done_count) == (0, 0, 0)
    assert _state(factory)[0][0]["raw_text"] == body
    assert _state(factory)[1] == []
    assert _stage(factory, empty).outcome == "unchanged"


def test_unchanged_pending_preserves_attempt_error_and_timestamps(factory, source):
    _stage(factory, source)
    with factory.begin() as session:
        chunk = session.scalar(select(KnowledgeChunkRecord))
        chunk.attempt_count, chunk.last_error = 3, "safe previous error"
    previous = _state(factory)
    statements = []
    event.listen(factory.kw["bind"], "before_cursor_execute",
                 lambda conn, cursor, sql, parameters, context, many: statements.append(sql))
    result = _stage(factory, source)
    assert result.outcome == "unchanged" and result.pending_count == 1
    assert _state(factory) == previous
    assert not any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for sql in statements)


def test_unchanged_preserves_mixed_done_and_pending(factory, source):
    document = source.model_copy(update={"text": "# 政策\n## Q1\n正文一\n## Q2\n正文二"})
    _stage(factory, document)
    with factory.begin() as session:
        first = session.scalar(select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.chunk_index == 0))
        first.index_status = "done"
        first.indexed_hash = first.index_hash
        first.indexed_collection = "resolv_knowledge_v2"
        first.indexed_at = datetime(2026, 10, 12, tzinfo=UTC).replace(tzinfo=None)
        first.attempt_count = 2
    previous = _state(factory)
    result = _stage(factory, document)
    assert (result.outcome, result.pending_count, result.done_count) == ("unchanged", 1, 1)
    assert _state(factory) == previous


@pytest.mark.parametrize("change", [
    "raw", "type", "keywords", "max_chars", "overlap", "revision", "collection", "passage",
])
def test_changed_document_configuration_or_target_is_rejected_without_writes(factory, source, change):
    _stage(factory, source)
    previous = _state(factory)
    document, config, profile, options = source, None, None, {}
    if change == "raw":
        document = source.model_copy(update={"text": RAW + "补充条款"})
    elif change == "type":
        config = KnowledgeFileConfig(content_type="support")
    elif change == "keywords":
        config = KnowledgeFileConfig(content_type="refund_policy", key_clause_keywords=("其他词",))
    elif change in {"max_chars", "overlap"}:
        options[change] = 900 if change == "max_chars" else 50
    else:
        profile = KnowledgeIndexProfile(**{
            "model_revision": REVISION,
            **{"revision": {"model_revision": "a" * 40},
               "collection": {"collection_name": "another_v2"},
               "passage": {"passage_max_length": 1024}}[change],
        })
    with pytest.raises(_module().KnowledgeConflictError):
        _stage(factory, document, config, profile, **options)
    assert _state(factory) == previous


@pytest.mark.parametrize("damage", ["missing", "extra", "answer", "text", "hash", "raw"])
def test_existing_inconsistent_payload_is_not_silently_repaired(factory, source, damage):
    _stage(factory, source)
    with factory.begin() as session:
        row = session.scalar(select(KnowledgeChunkRecord))
        if damage == "missing":
            session.delete(row)
        elif damage == "extra":
            values = {column.name: getattr(row, column.name) for column in KnowledgeChunkRecord.__table__.columns}
            values.update(chunk_id="e" * 64, chunk_index=1)
            session.add(KnowledgeChunkRecord(**values))
        elif damage == "hash":
            row.index_hash = "e" * 64
        elif damage == "raw":
            session.get(KnowledgeDocument, source.document_id).raw_text = "篡改原文"
        else:
            setattr(row, damage, "篡改正文")
    damaged = _state(factory)
    with pytest.raises(_module().KnowledgeConflictError):
        _stage(factory, source)
    assert _state(factory) == damaged


@pytest.mark.parametrize("case", ["identity", "budget"])
def test_invalid_source_or_chunk_rules_fail_before_session_creation(source, case):
    def forbidden():
        raise AssertionError("Invalid input must not open Session")
    options = {"max_chars": 0} if case == "budget" else {}
    document = source if case == "budget" else source.model_copy(update={"document_id": "a" * 64})
    with pytest.raises(ValueError):
        _stage(forbidden, document, **options)


def test_real_chunk_constraint_failure_rolls_back_entire_document(factory, source):
    def corrupt(session, context, instances):
        for row in session.new:
            if isinstance(row, KnowledgeChunkRecord):
                row.is_key_clause = 2
    event.listen(Session, "before_flush", corrupt)
    try:
        with pytest.raises(KnowledgeStoreError):
            _stage(factory, source)
    finally:
        event.remove(Session, "before_flush", corrupt)
    assert _state(factory) == ([], [])

@pytest.mark.parametrize("backend,code", [
    ("sqlite", 275), ("sqlite", 1299), ("sqlite", 787),
    ("mysql", 1048), ("mysql", 1452), ("unknown", 1062),
])
def test_nonunique_or_unknown_integrity_error_is_not_hidden_by_same_winner(factory, source, backend, code):
    import sqlite3

    from pymysql.err import IntegrityError as MySQLIntegrityError

    if backend == "sqlite":
        original = sqlite3.IntegrityError("constraint failure")
        original.sqlite_errorcode = code
    elif backend == "mysql":
        original = MySQLIntegrityError(code, "constraint failure")
    else:
        original = RuntimeError(code, "unknown backend error")

    class FailedWithWinner(Session):
        def get(self, entity, identity, **options):
            found = super().get(entity, identity, **options)
            if entity is KnowledgeDocument and found is None:
                _stage(factory, source)
            return found

    def fail(session, context, instances):
        raise IntegrityError("insert", {}, original)

    event.listen(FailedWithWinner, "before_flush", fail)
    failing = sessionmaker(bind=factory.kw["bind"], class_=FailedWithWinner)
    with pytest.raises(KnowledgeStoreError) as caught:
        _stage(failing, source)
    assert isinstance(caught.value.__cause__, IntegrityError)
    assert caught.value.__cause__.orig is original
    assert len(_state(factory)[0]) == len(_state(factory)[1]) == 1


def test_unique_race_with_close_failure_cannot_report_unchanged(factory, source):
    class RaceCloseSession(Session):
        def get(self, entity, identity, **options):
            found = super().get(entity, identity, **options)
            if entity is KnowledgeDocument and found is None:
                _stage(factory, source)
            return found

        def close(self):
            super().close()
            raise OperationalError("close", {}, RuntimeError("private close failure"))

    sessions = iter([RaceCloseSession(bind=factory.kw["bind"]), factory()])
    with pytest.raises(KnowledgeStoreError) as caught:
        _stage(lambda: next(sessions), source)
    assert isinstance(caught.value.__cause__, IntegrityError)
    assert "private" not in str(caught.value)
    assert len(_state(factory)[0]) == len(_state(factory)[1]) == 1


def test_commit_failure_after_flush_rolls_back_document_and_chunks(factory, source):
    class FailedCommitSession(Session):
        pass
    def fail(session):
        session.flush()
        raise OperationalError("commit", {}, RuntimeError("private commit data"))
    event.listen(FailedCommitSession, "before_commit", fail)
    failing = sessionmaker(bind=factory.kw["bind"], class_=FailedCommitSession)
    with pytest.raises(KnowledgeStoreError) as caught:
        _stage(failing, source)
    assert isinstance(caught.value.__cause__, OperationalError)
    assert "private" not in str(caught.value)
    assert _state(factory) == ([], [])


def test_close_failure_reports_error_even_after_committed_stage(factory, source):
    class FailedCloseSession(Session):
        def close(self):
            super().close()
            raise OperationalError("close", {}, RuntimeError("private close data"))
    failing = sessionmaker(bind=factory.kw["bind"], class_=FailedCloseSession)
    with pytest.raises(KnowledgeStoreError) as caught:
        _stage(failing, source)
    assert "private" not in str(caught.value)
    assert len(_state(factory)[0]) == 1
    assert _stage(factory, source).outcome == "unchanged"


def test_close_failure_does_not_mask_original_database_failure(factory, source):
    class BothFailedSession(Session):
        def close(self):
            super().close()
            raise OperationalError("close", {}, RuntimeError("close fault"))
    def fail(session):
        session.flush()
        raise OperationalError("commit", {}, RuntimeError("original commit fault"))
    event.listen(BothFailedSession, "before_commit", fail)
    failing = sessionmaker(bind=factory.kw["bind"], class_=BothFailedSession)
    with pytest.raises(KnowledgeStoreError) as caught:
        _stage(failing, source)
    assert str(caught.value.__cause__.orig) == "original commit fault"
    assert _state(factory) == ([], [])


@pytest.mark.parametrize("winner", ["same", "different"])
def test_raced_unique_insert_checks_complete_winner_in_fresh_session(factory, source, winner):
    class RaceSession(Session):
        def get(self, entity, identity, **options):
            found = super().get(entity, identity, **options)
            if entity is KnowledgeDocument and found is None:
                other = source if winner == "same" else source.model_copy(update={"text": RAW + "变更"})
                _stage(factory, other)
            return found
    sessions = iter([RaceSession(bind=factory.kw["bind"]), factory()])
    if winner == "same":
        result = _stage(lambda: next(sessions), source)
        assert result.outcome == "unchanged"
        assert (result.chunk_count, result.pending_count) == (1, 1)
    else:
        with pytest.raises(_module().KnowledgeConflictError):
            _stage(lambda: next(sessions), source)
    assert len(_state(factory)[0]) == len(_state(factory)[1]) == 1


def test_integrity_failure_without_winner_is_storage_error(factory, source):
    class FailedSession(Session):
        pass
    def fail(session, context, instances):
        raise IntegrityError("insert", {}, RuntimeError("unrelated integrity"))
    event.listen(FailedSession, "before_flush", fail)
    sessions = iter([FailedSession(bind=factory.kw["bind"]), factory()])
    with pytest.raises(KnowledgeStoreError):
        _stage(lambda: next(sessions), source)
    assert _state(factory) == ([], [])


def test_no_pending_rows_remain_after_db_statement_failure(factory, source):
    def fail(connection, cursor, sql, params, context, many):
        if sql.startswith("INSERT INTO knowledge_chunks"):
            raise OperationalError(sql, params, RuntimeError("private SQL params"))
    event.listen(factory.kw["bind"], "before_cursor_execute", fail)
    try:
        with pytest.raises(KnowledgeStoreError):
            _stage(factory, source)
    finally:
        event.remove(factory.kw["bind"], "before_cursor_execute", fail)
    assert _state(factory) == ([], [])
