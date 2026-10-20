"""Real SQL confirmations stay pending after uncertain inference/vector writes."""

import importlib
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from sqlalchemy import MetaData, String, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from resolv.db.engine import create_db_engine
from resolv.db.knowledge import KnowledgeStoreError
from resolv.db.knowledge_base import KnowledgeBase
from resolv.db.milvus import VectorStoreError
from resolv.knowledge import knowledge_chunk_from_record, knowledge_index_hash
from resolv.models.knowledge import KnowledgeChunkRecord
from resolv.repositories.knowledge import KnowledgeRepository
from resolv.schemas.knowledge import KnowledgeIndexProfile, KnowledgeIndexSnapshot
from tests.knowledge_fixtures import chunk_record, document_record

PROFILE = KnowledgeIndexProfile(model_revision="a" * 40)
INDEXED_AT = datetime(2026, 10, 12, tzinfo=UTC).replace(tzinfo=None)


class TrackedSession(Session):
    active: ClassVar[set] = set()
    fail_confirmation = False
    fail_error = False
    fail_close = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active.add(self)

    def commit(self):
        if self.fail_confirmation and any(
            "index_status" in str(statement) and "indexed_hash" in str(statement)
            for statement in self.info.get("updates", [])
        ):
            type(self).fail_confirmation = False
            raise OperationalError("COMMIT", {}, OSError("private db password"))
        if self.fail_error and self.info.get("error_write"):
            raise OperationalError("COMMIT", {}, OSError("private error recording"))
        return super().commit()

    def close(self):
        self.active.discard(self)
        super().close()
        if self.fail_close:
            raise OSError("private close error")


@pytest.fixture
def factory(tmp_path):
    TrackedSession.active.clear()
    TrackedSession.fail_confirmation = False
    TrackedSession.fail_error = False
    TrackedSession.fail_close = False
    engine = create_db_engine(f"sqlite:///{tmp_path / 'indexing.db'}")
    KnowledgeBase.metadata.create_all(engine)
    factory = sessionmaker(engine, class_=TrackedSession)
    yield factory
    engine.dispose()


@event.listens_for(TrackedSession, "do_orm_execute")
def track_updates(state):
    if state.is_update:
        state.session.info.setdefault("updates", []).append(state.statement)
        if "last_error" in str(state.statement):
            state.session.info["error_write"] = True


def seed(factory, count=1, **overrides):
    with factory.begin() as session:
        session.add(document_record())
        for index in range(count):
            record = chunk_record(chunk_id=f"{index + 1:064x}", chunk_index=index)
            record.index_hash = knowledge_index_hash(knowledge_chunk_from_record(record), PROFILE)
            for name, value in overrides.items():
                setattr(record, name, value)
            session.add(record)


def read(factory):
    with factory() as session:
        return [
            (row.chunk_id, row.index_status, row.attempt_count, row.last_error,
             row.index_hash, row.indexed_hash, row.indexed_collection, row.indexed_at)
            for row in session.query(KnowledgeChunkRecord).order_by(KnowledgeChunkRecord.chunk_id)
        ]


class Lock:
    lost = False

    def assert_owned(self):
        if self.lost:
            raise RuntimeError("index lock lost")


class Embedding:
    profile = PROFILE
    dimension = 1024

    def __init__(self, callback=None):
        self.callback = callback
        self.texts = []

    def embed_documents(self, texts):
        assert not TrackedSession.active, "Inference retained a SQL Session"
        self.texts.extend(texts)
        if self.callback:
            self.callback()
        return [[1.0] + [0.0] * 1023 for _ in texts]


class Store:
    collection_name = PROFILE.collection_name

    def __init__(self, callback=None, count=None):
        self.callback, self.count = callback, count
        self.entries = []

    def upsert(self, entries, vectors):
        assert not TrackedSession.active, "Vector RPC retained a SQL Session"
        self.entries.extend(entries)
        assert len(entries) == len(vectors)
        if self.callback:
            self.callback()
        return len(entries) if self.count is None else self.count


def indexer(factory, embedding=None, store=None, lock=None):
    module = importlib.import_module("resolv.services.knowledge_indexing")
    return module.KnowledgeIndexer(factory, embedding or Embedding(), store or Store(), PROFILE, lock or Lock())


def test_pending_batches_close_sessions_before_model_and_rpc_then_confirm(factory):
    seed(factory, 5)
    embedding, store = Embedding(), Store()
    result = indexer(factory, embedding, store).run(batch_size=2)
    assert result.model_dump() == {"attempted_count": 5, "indexed_count": 5, "batch_count": 3}
    assert embedding.texts == ["售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。"] * 5
    assert [entry.chunk_id for entry in store.entries] == [f"{i:064x}" for i in range(1, 6)]
    for row in read(factory):
        assert row[1:4] == ("done", 1, None)
        assert row[4] == row[5]
        assert row[6] == "resolv_knowledge_v2"
        assert isinstance(row[7], datetime)
    assert indexer(factory).run().model_dump() == {"attempted_count": 0, "indexed_count": 0, "batch_count": 0}


def test_empty_database_never_loads_or_calls_inference(factory):
    def unexpected():
        pytest.fail("Empty run called external resource")
    result = indexer(factory, Embedding(unexpected), Store(unexpected)).run()
    assert result.indexed_count == result.attempted_count == result.batch_count == 0


@pytest.fixture
def pad_space_factory(tmp_path):
    """Exercise MySQL's documented PAD SPACE equality without external services."""
    engine = create_db_engine(f"sqlite:///{tmp_path / 'pad-space.db'}")
    def register(connection, _record):
        def compare(left, right):
            left, right = left.rstrip(" "), right.rstrip(" ")
            return (left > right) - (left < right)
        connection.create_collation("PAD_SPACE", compare)
    event.listen(engine, "connect", register)
    metadata = MetaData()
    for table in KnowledgeBase.metadata.sorted_tables:
        copy = table.to_metadata(metadata)
        for column in copy.columns:
            if isinstance(column.type, String):
                column.type = column.type.copy()
                column.type.collation = "PAD_SPACE"
    metadata.create_all(engine)
    yield sessionmaker(engine, class_=TrackedSession)
    engine.dispose()


@pytest.mark.parametrize("phase", ["before_attempt", "before_confirmation"])
@pytest.mark.parametrize("field", ["source", "category", "questions", "answer", "section_path", "content_type", "text"])
def test_pad_space_payload_change_never_updates_snapshot(pad_space_factory, phase, field):
    factory = pad_space_factory
    seed(factory)
    def mutate():
        with factory.begin() as session:
            record = session.get(KnowledgeChunkRecord, f"{1:064x}")
            setattr(record, field, getattr(record, field) + " ")
    class MutationLock(Lock):
        calls = 0
        def assert_owned(self):
            self.calls += 1
            if phase == "before_attempt" and self.calls == 2:
                mutate()
    embedding, store = Embedding(), Store(mutate if phase == "before_confirmation" else None)
    with pytest.raises(KnowledgeStoreError):
        indexer(factory, embedding, store, MutationLock()).run()
    row = read(factory)[0]
    assert row[1:4] == ("pending", 0 if phase == "before_attempt" else 1, None)
    assert row[5:] == (None, None, None)
    if phase == "before_attempt":
        assert not embedding.texts and not store.entries


def test_pad_space_normalized_field_rejected_before_attempt(pad_space_factory):
    seed(pad_space_factory, content_type="refund_policy ")
    embedding, store = Embedding(), Store()
    with pytest.raises(KnowledgeStoreError):
        indexer(pad_space_factory, embedding, store).run()
    assert read(pad_space_factory)[0][1:4] == ("pending", 0, None)
    assert not embedding.texts and not store.entries


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size_fails_before_any_sql_or_inference(factory, size):
    with pytest.raises(ValueError):
        indexer(factory).run(batch_size=size)
    assert not TrackedSession.active


@pytest.mark.parametrize("bad", ["profile", "dimension", "collection"])
def test_wrong_resource_profile_fails_before_attempt(factory, bad):
    seed(factory)
    embedding, store = Embedding(), Store()
    if bad == "profile":
        embedding.profile = PROFILE.model_copy(update={"model_revision": "b" * 40})
    elif bad == "dimension":
        embedding.dimension = 3
    else:
        store.collection_name = "wrong_collection"
    with pytest.raises(ValueError):
        indexer(factory, embedding, store).run()
    assert read(factory)[0][1:4] == ("pending", 0, None)


@pytest.mark.parametrize("overrides", [{"index_hash": "b" * 64}, {"text": "stale text"}, {"answer": "edited answer"}])
def test_invalid_source_hash_or_text_is_rejected_before_model(factory, overrides):
    seed(factory, **overrides)
    embedding, store = Embedding(), Store()
    with pytest.raises(ValueError):
        indexer(factory, embedding, store).run()
    assert embedding.texts == []
    assert store.entries == []
    assert read(factory)[0][1] == "pending"


@pytest.mark.parametrize("phase", ["embedding", "rpc"])
def test_external_failure_keeps_pending_and_records_only_safe_error(factory, phase):
    seed(factory, 2)
    failure = TimeoutError("private password and customer text")
    def fail():
        assert [row[2] for row in read(factory)] == [1, 1]
        raise failure
    embedding = Embedding(fail if phase == "embedding" else None)
    store = Store(fail if phase == "rpc" else None)
    with pytest.raises(TimeoutError) as raised:
        indexer(factory, embedding, store).run()
    assert raised.value is failure
    for row in read(factory):
        assert row[1:3] == ("pending", 1)
        assert row[3] and "private" not in row[3] and "password" not in row[3]
        assert row[5:] == (None, None, None)


@pytest.mark.parametrize("count", [0, 1, True, None])
def test_partial_or_invalid_upsert_count_cannot_confirm_any_row(factory, count):
    seed(factory, 2)
    store = Store(count=count)
    if count is None:
        store.upsert = lambda _entries, _vectors: None
    with pytest.raises(VectorStoreError):
        indexer(factory, store=store).run()
    assert [row[1:3] for row in read(factory)] == [("pending", 1)] * 2


@pytest.mark.parametrize("field,value", [("index_hash", "b" * 64), ("text", "late edited text"), ("answer", "late edited answer")])
def test_source_mutation_after_upsert_rolls_back_entire_confirmation(factory, field, value):
    seed(factory, 2)
    def mutate():
        with factory.begin() as session:
            setattr(session.get(KnowledgeChunkRecord, f"{2:064x}"), field, value)
    with pytest.raises(RuntimeError):
        indexer(factory, store=Store(mutate)).run()
    rows = read(factory)
    assert [row[1] for row in rows] == ["pending", "pending"]
    assert [row[5] for row in rows] == [None, None]


def test_lost_lock_after_upsert_cannot_confirm_done(factory):
    seed(factory)
    lock = Lock()
    def lose():
        lock.lost = True
    with pytest.raises(RuntimeError, match="lock lost"):
        indexer(factory, store=Store(lose), lock=lock).run()
    assert read(factory)[0][1:3] == ("pending", 1)


def test_done_commit_failure_is_safe_and_idempotent_retry_completes(factory):
    seed(factory)
    TrackedSession.fail_confirmation = True
    with pytest.raises(RuntimeError) as raised:
        indexer(factory).run()
    assert isinstance(raised.value.__cause__, OperationalError)
    assert "password" not in str(raised.value)
    assert read(factory)[0][1:3] == ("pending", 1)
    result = indexer(factory).run()
    assert result.indexed_count == 1
    assert read(factory)[0][1:3] == ("done", 2)


def test_error_recording_and_session_close_cannot_mask_external_failure(factory):
    seed(factory)
    failure = TimeoutError("original model failure")
    def fail():
        TrackedSession.fail_error = True
        TrackedSession.fail_close = True
        raise failure
    with pytest.raises(TimeoutError) as raised:
        indexer(factory, Embedding(fail)).run()
    assert raised.value is failure
    assert raised.value.__notes__
    TrackedSession.fail_close = False
    assert read(factory)[0][1:3] == ("pending", 1)


def test_repository_keyset_and_guarded_updates_are_caller_owned(factory):
    seed(factory, 3)
    with factory() as session:
        repository = KnowledgeRepository(session)
        records = repository.list_pending_chunks(after_id=f"{1:064x}", limit=1)
        assert [record.chunk_id for record in records] == [f"{2:064x}"]
        snapshots = [KnowledgeIndexSnapshot(**knowledge_chunk_from_record(record).model_dump(), index_hash=record.index_hash) for record in records]
        assert repository.increment_attempts(snapshots) == 1
        assert read(factory)[1][2] == 0
        session.rollback()
        assert repository.record_indexing_error(snapshots, "private secret") == 1
        session.rollback()
        assert repository.confirm_indexed(snapshots, PROFILE.collection_name, INDEXED_AT) == 1
        assert read(factory)[1][1] == "pending"
        session.rollback()
    assert read(factory)[1][1:4] == ("pending", 0, None)


def test_guarded_repository_updates_skip_done_or_changed_snapshot(factory):
    seed(factory, 2)
    with factory() as session:
        records = KnowledgeRepository(session).list_pending_chunks()
        snapshots = [KnowledgeIndexSnapshot(**knowledge_chunk_from_record(record).model_dump(), index_hash=record.index_hash) for record in records]
    with factory.begin() as session:
        row = session.get(KnowledgeChunkRecord, f"{1:064x}")
        row.index_status = "done"
        row.indexed_hash = row.index_hash
        row.indexed_collection = PROFILE.collection_name
        row.indexed_at = INDEXED_AT
        session.get(KnowledgeChunkRecord, f"{2:064x}").index_hash = "b" * 64
    with factory.begin() as session:
        repository = KnowledgeRepository(session)
        assert repository.increment_attempts(snapshots) == 0
        assert repository.confirm_indexed(snapshots, PROFILE.collection_name, INDEXED_AT) == 0
        assert repository.record_indexing_error(snapshots, "private secret") == 0
    assert [row[2:4] for row in read(factory)] == [(0, None), (0, None)]


def test_run_does_not_repeat_a_failed_external_attempt(factory):
    seed(factory, 3)
    calls = 0
    def fail():
        nonlocal calls
        calls += 1
        raise TimeoutError("inference failure")
    with pytest.raises(TimeoutError):
        indexer(factory, Embedding(fail)).run(batch_size=2)
    assert calls == 1
    assert [row[2] for row in read(factory)] == [1, 1, 0]


def test_attempt_increment_rolls_back_if_any_snapshot_changed(factory):
    seed(factory, 2)
    lock = Lock()
    assertions = 0
    def assert_owned():
        nonlocal assertions
        assertions += 1
        if assertions == 2:
            with factory.begin() as session:
                session.get(KnowledgeChunkRecord, f"{2:064x}").index_hash = "b" * 64
    lock.assert_owned = assert_owned
    embedding = Embedding()
    with pytest.raises(RuntimeError):
        indexer(factory, embedding, lock=lock).run()
    assert [row[2] for row in read(factory)] == [0, 0]
    assert embedding.texts == []


def test_session_creation_failure_is_safe_and_preserves_sql_cause():
    failure = OperationalError("CONNECT", {}, OSError("private password"))
    def broken_factory():
        raise failure
    with pytest.raises(RuntimeError) as raised:
        indexer(broken_factory).run()
    assert raised.value.__cause__ is failure
    assert "password" not in str(raised.value)


def test_failed_snapshot_session_close_stops_before_attempt_or_inference(factory):
    seed(factory)
    TrackedSession.fail_close = True
    embedding, store = Embedding(), Store()
    with pytest.raises(RuntimeError, match="cleanup"):
        indexer(factory, embedding, store).run()
    TrackedSession.fail_close = False
    assert embedding.texts == []
    assert store.entries == []
    assert read(factory)[0][1:3] == ("pending", 0)
