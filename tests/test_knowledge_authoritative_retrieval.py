"""Retrieve from real SQL authority; only embedding and vector RPCs are fake."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from resolv.db.knowledge import KnowledgeStoreError
from resolv.db.knowledge_base import KnowledgeBase
from resolv.knowledge import knowledge_index_hash
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.schemas.knowledge import (
    KnowledgeCandidate,
    KnowledgeChunk,
    KnowledgeIndexProfile,
    KnowledgeSearchHit,
)
from resolv.services.knowledge_retrieval import KnowledgeRetriever

PROFILE = KnowledgeIndexProfile(model_revision="a" * 40)
DOC_ID = "d" * 64
FIRST_ID = "1" * 64
SECOND_ID = "2" * 64
MISSING_ID = "f" * 64
CONFIRMED_AT = datetime(2026, 10, 12, 1, 2, 3, tzinfo=UTC).replace(tzinfo=None)


@dataclass
class Database:
    engine: object
    factory: object
    active: list[Session] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


@pytest.fixture
def database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'retrieval.sqlite'}")
    KnowledgeBase.metadata.create_all(engine)
    db = Database(engine, None)

    class TrackingSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            db.active.append(self)
            db.events.append("sql-open")

        def close(self):
            # Expire before detach: using ORM state after close must fail.
            self.expire_all()
            super().close()
            db.active.remove(self)
            db.events.append("sql-close")

    db.factory = sessionmaker(bind=engine, class_=TrackingSession)

    @event.listens_for(engine, "before_cursor_execute")
    def capture_reads(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            db.queries.append(statement)

    yield db
    assert not db.active
    engine.dispose()


def _chunk(chunk_id=FIRST_ID, chunk_index=0):
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=DOC_ID,
        chunk_index=chunk_index,
        source="policies/refund.md",
        category="退款政策",
        questions="已激活软件",
        answer="激活后不支持退款。",
        section_path="退款政策 > 已激活软件",
        content_type="refund_policy",
        is_key_clause=1,
        text="退款政策 > 已激活软件\n\n激活后不支持退款。",
    )


def _seed(database, *chunks, profile=PROFILE):
    with Session(database.engine) as session, session.begin():
        session.add(KnowledgeDocument(
            document_id=DOC_ID, source="policies/refund.md",
            content_type="refund_policy", raw_text="authoritative original",
            ingest_hash="e" * 64,
        ))
        session.flush()
        for chunk in chunks:
            index_hash = knowledge_index_hash(chunk, profile)
            session.add(KnowledgeChunkRecord(
                **chunk.model_dump(), index_hash=index_hash, index_status="done",
                indexed_hash=index_hash, indexed_collection=profile.collection_name,
                indexed_at=CONFIRMED_AT,
            ))
    database.queries.clear()


def _candidate(chunk=None, score=0.8, **changes):
    chunk = chunk or _chunk()
    payload = {
        "chunk_id": chunk.chunk_id, "document_id": chunk.document_id,
        "index_hash": knowledge_index_hash(chunk, PROFILE), "score": score,
    }
    payload.update(changes)
    return KnowledgeCandidate(**payload)


class FakeEmbedding:
    def __init__(self, database, *, profile=PROFILE, dimension=1024, error=None):
        self.database = database
        self.profile = profile
        self.dimension = dimension
        self.error = error
        self.queries = []
        self.vector = [1.0] + [0.0] * 1023

    def embed_query(self, query):
        assert not self.database.active
        self.database.events.append("embed")
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.vector


class FakeVectorStore:
    def __init__(self, database, candidates, *, collection=None, error=None):
        self.database = database
        self.collection_name = collection or PROFILE.collection_name
        self.candidates = candidates
        self.error = error
        self.requests = []

    def search(self, vector, top_k):
        assert not self.database.active
        self.database.events.append("search")
        self.requests.append((vector, top_k))
        if self.error:
            raise self.error
        return self.candidates


def _retriever(database, candidates=(), *, embedding=None, store=None,
               factory=None, profile=PROFILE):
    return KnowledgeRetriever(
        database.factory if factory is None else factory,
        embedding or FakeEmbedding(database),
        store or FakeVectorStore(database, list(candidates)),
        profile,
    )


def test_authoritative_payload_is_detached_with_score_after_sql_close(database):
    # Returning index payload, dropping a knowledge field, or reading detached ORM fails.
    _seed(database, _chunk())
    hits = _retriever(database, [_candidate(score=-0.25)]).retrieve("退款", 1)

    assert len(hits) == 1
    assert isinstance(hits[0], KnowledgeSearchHit)
    assert hits[0].model_dump() == {
        "chunk_id": FIRST_ID, "document_id": DOC_ID, "chunk_index": 0,
        "source": "policies/refund.md",
        "category": "退款政策", "questions": "已激活软件",
        "answer": "激活后不支持退款。", "section_path": "退款政策 > 已激活软件",
        "content_type": "refund_policy", "is_key_clause": 1,
        "text": "退款政策 > 已激活软件\n\n激活后不支持退款。", "score": -0.25,
    }
    assert database.events == ["embed", "search", "sql-open", "sql-close"]
    assert len(database.queries) == 1


def test_provider_order_score_and_first_id_are_preserved_in_one_batch(database):
    # Sorting by SQL order/score, applying a threshold, or retaining duplicates fails.
    first, second = _chunk(), _chunk(SECOND_ID, 1)
    _seed(database, first, second)
    hits = _retriever(database, [
        _candidate(second, -0.8), _candidate(first, 0.2),
        _candidate(second, 0.99), _candidate(chunk_id=MISSING_ID),
    ]).retrieve("退款", 4)
    assert [(hit.chunk_id, hit.score) for hit in hits] == [
        (SECOND_ID, -0.8), (FIRST_ID, 0.2),
    ]
    assert len(database.queries) == 1
    assert " IN " in database.queries[0]
    assert database.events == ["embed", "search", "sql-open", "sql-close"]


@pytest.mark.parametrize("changes", [
    {"index_status": "pending"},
    {"indexed_hash": None},
    {"indexed_hash": "b" * 64},
    {"indexed_collection": None},
    {"indexed_collection": "another_collection"},
    {"indexed_at": None},
    {"index_hash": "c" * 64, "indexed_hash": "c" * 64},
    {"category": "修改后的分类"},
    {"questions": "修改后的问题"},
    {"answer": "修改后的正文"},
    {"text": "被修改的索引正文"},
    {"source": "other.md"},
    {"section_path": "other heading"},
    {"content_type": "other_type"},
    {"content_type": " refund_policy "},
    {"content_type": " "},
    {"is_key_clause": 0},
    {"chunk_index": 1},
    {"answer": "   "},
])
def test_unconfirmed_stale_or_inconsistent_sql_payload_is_filtered(database, changes):
    # Omitting the corresponding confirmation, text, or profile-hash check fails.
    _seed(database, _chunk())
    with Session(database.engine) as session, session.begin():
        # Model legacy/corrupted rows that stricter modern SQL constraints prevent.
        session.execute(text("PRAGMA ignore_check_constraints = ON"))
        record = session.get(KnowledgeChunkRecord, FIRST_ID)
        for name, value in changes.items():
            setattr(record, name, value)
    database.queries.clear()
    assert _retriever(database, [_candidate()]).retrieve("退款") == []
    assert len(database.queries) == 1
    assert database.events == ["embed", "search", "sql-open", "sql-close"]


@pytest.mark.parametrize("changes", [
    {"document_id": "9" * 64}, {"index_hash": "b" * 64},
])
def test_candidate_identity_or_hash_must_match_sql(database, changes):
    # Trusting only SQL done while ignoring candidate confirmation fails.
    _seed(database, _chunk())
    assert _retriever(database, [_candidate(**changes)]).retrieve("退款") == []


def test_sql_hash_must_be_recomputed_for_declared_profile(database):
    # Accepting confirmed data from a different embedding revision fails.
    other_profile = PROFILE.model_copy(update={"model_revision": "b" * 40})
    _seed(database, _chunk(), profile=other_profile)
    candidate = _candidate(index_hash=knowledge_index_hash(_chunk(), other_profile))
    assert _retriever(database, [candidate]).retrieve("退款") == []


@pytest.mark.parametrize("query,top_k", [
    ("", 5), (" \n\t", 5), (None, 5), (1, 5), (True, 5),
    ("refund", 0), ("refund", -1), ("refund", 16384),
    ("refund", True), ("refund", 1.5), ("refund", "5"), ("refund", None),
])
def test_invalid_inputs_fail_before_embedding_search_or_sql(database, query, top_k):
    # Calling any resource or accepting bool/noninteger/out-of-range top_k fails.
    retriever = _retriever(database)
    with pytest.raises(ValueError):
        retriever.retrieve(query, top_k)
    assert database.events == []
    assert database.queries == []


@pytest.mark.parametrize("top_k", [1, 16383])
def test_supported_limits_and_original_query_reach_search(database, top_k):
    # Trimming the query or forwarding the wrong search limit/vector fails.
    embedding = FakeEmbedding(database)
    store = FakeVectorStore(database, [])
    assert _retriever(database, embedding=embedding, store=store).retrieve("  退款  ", top_k) == []
    assert embedding.queries == ["  退款  "]
    assert store.requests == [(embedding.vector, top_k)]
    assert database.events == ["embed", "search"]


@pytest.mark.parametrize("mismatch", ["profile", "dimension", "collection"])
def test_declared_target_mismatch_fails_before_any_resource(database, mismatch):
    # Embedding with a different profile/dimension or searching another collection fails.
    embedding = FakeEmbedding(database)
    store = FakeVectorStore(database, [])
    if mismatch == "profile":
        embedding.profile = PROFILE.model_copy(update={"model_revision": "b" * 40})
    elif mismatch == "dimension":
        embedding.dimension = 768
    else:
        store.collection_name = "wrong_collection"
    with pytest.raises(ValueError):
        _retriever(database, embedding=embedding, store=store).retrieve("退款")
    assert database.events == []


def test_empty_candidates_are_success_without_opening_sql(database):
    # An unnecessary database read or treating legal no hits as failure fails.
    assert _retriever(database).retrieve("退款") == []
    assert database.events == ["embed", "search"]
    assert database.queries == []


@pytest.mark.parametrize("boundary", ["embedding", "search"])
def test_external_failures_propagate_unchanged_without_opening_sql(database, boundary):
    # Returning [] or wrapping embedding/vector errors as a store error fails.
    error = RuntimeError("external failure")
    embedding = FakeEmbedding(database, error=error if boundary == "embedding" else None)
    store = FakeVectorStore(database, [], error=error if boundary == "search" else None)
    with pytest.raises(RuntimeError) as caught:
        _retriever(database, embedding=embedding, store=store).retrieve("退款")
    assert caught.value is error
    assert "sql-open" not in database.events


def test_sql_factory_failure_is_safe_with_original_cause(database):
    # Leaking credentials or returning [] after an unavailable SQL factory fails.
    error = RuntimeError("mysql://user:secret@private-host/db")

    def broken_factory():
        raise error

    with pytest.raises(KnowledgeStoreError) as caught:
        _retriever(database, [_candidate()], factory=broken_factory).retrieve("退款")
    assert caught.value.__cause__ is error
    assert "secret" not in str(caught.value)
    assert database.events == ["embed", "search"]


def test_sql_read_failure_is_safe_and_session_closes(database):
    # Returning [] after a SQL failure, leaking parameters, or leaving Session open fails.
    error = OperationalError("select secret", {}, RuntimeError("secret"))

    @event.listens_for(database.engine, "before_cursor_execute")
    def fail_select(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            raise error

    with pytest.raises(KnowledgeStoreError) as caught:
        _retriever(database, [_candidate()]).retrieve("退款")
    assert caught.value.__cause__ is error
    assert "secret" not in str(caught.value)
    assert database.events == ["embed", "search", "sql-open", "sql-close"]


@pytest.mark.parametrize("read_fails", [False, True])
def test_sql_close_failure_does_not_hide_primary_read_error(database, read_fails):
    # Reporting success despite failed cleanup or replacing the primary SQL error fails.
    read_error = OperationalError("select secret", {}, RuntimeError("read secret"))
    close_error = RuntimeError("close secret")

    def factory():
        session = database.factory()
        original_close = session.close

        def broken_close():
            original_close()
            raise close_error

        session.close = broken_close
        return session

    if read_fails:
        @event.listens_for(database.engine, "before_cursor_execute")
        def fail_select(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                raise read_error

    with pytest.raises(KnowledgeStoreError) as caught:
        _retriever(database, [_candidate()], factory=factory).retrieve("退款")
    assert caught.value.__cause__ is (read_error if read_fails else close_error)
    assert "secret" not in str(caught.value)
    if read_fails:
        assert caught.value.__notes__
        assert all("secret" not in note for note in caught.value.__notes__)
    assert database.events == ["embed", "search", "sql-open", "sql-close"]
