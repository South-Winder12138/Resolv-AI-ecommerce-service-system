"""Repository behavior uses caller-owned transactions and real SQL."""

import importlib

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.knowledge_base import KnowledgeBase
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from tests.knowledge_fixtures import chunk_record, document_record


@pytest.fixture
def factory(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'repository.db'}")
    KnowledgeBase.metadata.create_all(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _repository(session):
    return importlib.import_module("resolv.repositories.knowledge").KnowledgeRepository(session)


def _seed(factory):
    with factory.begin() as session:
        repository = _repository(session)
        repository.add_document(document_record())
        repository.add_chunks([
            chunk_record(chunk_id="a" * 64, chunk_index=2),
            chunk_record(chunk_id="b" * 64, chunk_index=1),
            chunk_record(),
        ])


def test_repository_add_and_get_preserves_authoritative_metadata(factory):
    _seed(factory)
    with factory() as session:
        repository = _repository(session)
        document = repository.get_document("d" * 64)
        assert document.source == "knowledge/refund.md"
        assert document.content_type == "refund_policy"
        assert document.ingest_hash == "a" * 64
        chunks = repository.list_document_chunks(document.document_id)
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
        assert chunks[0].answer == "软件商品一旦激活，不支持无理由退款。"
        assert chunks[0].text == "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。"
        assert chunks[0].is_key_clause == 1


def test_document_chunk_listing_filters_other_documents(factory):
    _seed(factory)
    with factory.begin() as session:
        repository = _repository(session)
        repository.add_document(document_record(document_id="e" * 64, source="other.txt"))
        repository.add_chunks([chunk_record(chunk_id="f" * 64, document_id="e" * 64)])
    with factory() as session:
        assert [chunk.chunk_id for chunk in _repository(session).list_document_chunks("e" * 64)] == ["f" * 64]


def test_missing_document_and_chunks_are_legal_empty_results(factory):
    with factory() as session:
        repository = _repository(session)
        assert repository.get_document("missing") is None
        assert repository.list_document_chunks("missing") == []
        assert repository.get_chunks_by_ids(["missing"]) == {}


def test_batch_lookup_handles_missing_and_duplicate_ids_in_one_select(factory):
    _seed(factory)
    statements = []
    event.listen(factory.kw["bind"], "before_cursor_execute",
                 lambda _conn, _cursor, statement, _params, _ctx, _many: statements.append(statement))
    with factory() as session:
        results = _repository(session).get_chunks_by_ids(["a" * 64, "missing", "c" * 64, "a" * 64])
        assert set(results) == {"a" * 64, "c" * 64}
        assert results["a" * 64].chunk_index == 2
        assert results["c" * 64].index_status == "pending"
    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("SELECT")


def test_empty_batch_lookup_runs_no_sql(factory):
    statements = []
    event.listen(factory.kw["bind"], "before_cursor_execute",
                 lambda _conn, _cursor, statement, _params, _ctx, _many: statements.append(statement))
    with factory() as session:
        assert _repository(session).get_chunks_by_ids([]) == {}
    assert statements == []


def test_repository_does_not_commit_caller_transaction(factory):
    with factory() as session:
        repository = _repository(session)
        repository.add_document(document_record())
        repository.add_chunks([chunk_record()])
        session.flush()
        assert repository.get_document("d" * 64) is not None
        with factory() as reader:
            assert reader.get(KnowledgeDocument, "d" * 64) is None
            assert reader.get(KnowledgeChunkRecord, "c" * 64) is None
        session.rollback()
    with factory() as reader:
        assert reader.get(KnowledgeDocument, "d" * 64) is None


def test_caller_explicit_commit_persists_records_and_keeps_session_usable(factory):
    with factory() as session:
        repository = _repository(session)
        repository.add_document(document_record())
        repository.add_chunks([chunk_record()])
        session.commit()
        assert repository.list_document_chunks("d" * 64)[0].chunk_id == "c" * 64
    with factory() as reader:
        assert reader.get(KnowledgeChunkRecord, "c" * 64) is not None


def test_caller_rollback_on_unique_failure_removes_document_and_all_chunks(factory):
    with pytest.raises(IntegrityError), factory.begin() as session:
        repository = _repository(session)
        repository.add_document(document_record())
        repository.add_chunks([chunk_record(), chunk_record(chunk_id="b" * 64)])
    with factory() as reader:
        assert reader.scalar(select(KnowledgeDocument)) is None
        assert reader.scalar(select(KnowledgeChunkRecord)) is None
