import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from resolv.models.knowledge import KnowledgeDocument
from resolv.models.knowledge_hybrid import KnowledgeChunkIndex
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.services.hybrid_indexing import HybridKnowledgeIndexer
from tests.hybrid_fixtures import profile
from tests.test_knowledge_hybrid_repository import hybrid_factory, legacy_rows

__all__ = ["hybrid_factory"]


class Lock:
    lost = False

    def assert_owned(self):
        if self.lost:
            raise HybridRetrievalError("lock_lost")


class CountingSession(Session):
    active = 0
    lose_confirmation = False
    lost_once = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        CountingSession.active += 1
        self._counted = True

    def commit(self):
        done = self.scalar(
            select(func.count())
            .select_from(KnowledgeChunkIndex)
            .where(KnowledgeChunkIndex.index_status == "done")
        )
        super().commit()
        if done and CountingSession.lose_confirmation and not CountingSession.lost_once:
            CountingSession.lost_once = True
            raise OSError("confirmation commit succeeded; response lost")

    def close(self):
        try:
            super().close()
        finally:
            if self._counted:
                CountingSession.active -= 1
                self._counted = False


def components(factory, p=None):
    p = p or profile()
    lock = Lock()
    state = {"upserts": 0, "count": None, "hook": None}

    class Embed:
        profile = p
        dimension = 1024

        def embed_documents(self, texts):
            assert CountingSession.active == 0
            return [[1.0] + [0.0] * 1023 for _ in texts]

    class Store:
        profile = p
        collection_name = p.collection_name

        def create_or_validate(self):
            assert CountingSession.active == 0

        def validate_existing(self):
            assert CountingSession.active == 0

        def upsert(self, entries, vectors):
            assert CountingSession.active == 0
            state["upserts"] += 1
            if state["hook"]:
                state["hook"]()
            return len(entries) if state["count"] is None else state["count"]

    cls = sessionmaker(
        bind=factory.kw["bind"], class_=CountingSession, expire_on_commit=False
    )
    CountingSession.lose_confirmation = CountingSession.lost_once = False
    indexer = HybridKnowledgeIndexer(cls, Embed(), Store(), p, lock)
    return indexer, state, lock


def test_unknown_commit_is_reconciled_before_resume(hybrid_factory):
    factory, _, manifest = hybrid_factory
    indexer, state, _ = components(factory)
    indexer.prepare("v3", manifest)
    CountingSession.lose_confirmation = True
    with pytest.raises(HybridRetrievalError):
        indexer.run("v3")
    before = state["upserts"]
    result = indexer.run("v3")
    assert state["upserts"] == before
    assert result.summary.state == "ready" and result.summary.done_count == 2
    assert indexer.run("v3").attempted_count == 0
    assert CountingSession.active == 0


def test_two_generations_leave_legacy_pending_and_unselected_source_unchanged(
    hybrid_factory,
):
    factory, _, m = hybrid_factory
    with factory() as session:
        before = legacy_rows(session)
    for version, collection in [
        ("one", "mhhybrid_gate_one"),
        ("two", "mhhybrid_gate_two"),
    ]:
        indexer, _state, _ = components(
            factory, profile().model_copy(update={"collection_name": collection})
        )
        indexer.prepare(version, m)
        result = indexer.run(version, batch_size=1)
        assert result.summary.state == "ready" and result.batch_count == 2
    with factory() as session:
        assert legacy_rows(session) == before


@pytest.mark.parametrize("failure", ["partial", "lost", "source", "before_confirm"])
def test_failed_batch_never_ready(hybrid_factory, failure):
    factory, s, m = hybrid_factory
    indexer, state, lock = components(factory)
    indexer.prepare("v3", m)
    if failure == "partial":
        state["count"] = 1
    elif failure == "lost":
        state["hook"] = lambda: setattr(lock, "lost", True)
    elif failure == "source":

        def mutate():
            with factory.begin() as session:
                session.get(KnowledgeDocument, s.document.document_id).raw_text += " "

        state["hook"] = mutate
    else:

        def hook(stage):
            if stage == "before_confirm_commit":
                raise RuntimeError("injected")

        indexer.fault_hook = hook
    with pytest.raises(HybridRetrievalError):
        indexer.run("v3")
    with factory() as session:
        assert HybridIndexRepository(session).get_version("v3").state == "building"
        assert all(
            x.index_status == "pending"
            for x in session.scalars(select(KnowledgeChunkIndex))
        )
    assert CountingSession.active == 0


def test_profile_conflict_precedes_external_writes(hybrid_factory):
    factory, _, m = hybrid_factory
    indexer, _state, _ = components(factory)
    indexer.prepare("v3", m)
    other, new_state, _ = components(factory, profile(passage_max_length=1024))
    with pytest.raises(HybridRetrievalError):
        other.prepare("v3", m)
    assert new_state["upserts"] == 0
