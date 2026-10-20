import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import sessionmaker

from resolv.db.knowledge_hybrid import HybridReadSession, consistent_read_factory
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.schemas.hybrid_knowledge import HybridCandidate, HybridRetrievalError
from resolv.services.hybrid_authority import HybridAuthorityReader
from resolv.services.hybrid_manifest import build_corpus_manifest
from resolv.services.hybrid_ranking import fuse_rrf
from tests.hybrid_fixtures import profile
from tests.test_hybrid_manifest import CONFIG, source
from tests.test_knowledge_hybrid_repository import hybrid_factory, seed_source

__all__ = ["hybrid_factory"]


def ready(factory, m, version="v3"):
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        repo.create_version(version, profile(), m)
        snaps = repo.list_pending(version, None, 1000)
        repo.confirm_done(version, snaps)
        repo.mark_ready(version)
    return tuple(
        HybridCandidate(
            index_version=version,
            chunk_id=x.entry.manifest_chunk.chunk.chunk_id,
            document_id=x.entry.manifest_chunk.chunk.document_id,
            index_hash=x.entry.index_hash,
            route="dense",
            rank=i + 1,
            raw_score=0.5,
        )
        for i, x in enumerate(snaps[:50])
    )


def reader(factory):
    return HybridAuthorityReader(
        consistent_read_factory(factory.kw["bind"]), "v3", profile()
    )


def test_authority_batches_reads_and_closes_snapshot(hybrid_factory):
    factory, _, m = hybrid_factory
    candidates = ready(factory, m)
    engine = factory.kw["bind"]
    select_count = 0
    active_sessions = 0

    class Tracking(HybridReadSession):
        def __init__(self, *a, **kw):
            nonlocal active_sessions
            super().__init__(*a, **kw)
            active_sessions += 1
            self.closed_once = False

        def close(self):
            nonlocal active_sessions
            super().close()
            if not self.closed_once:
                active_sessions -= 1
                self.closed_once = True

    def count(conn, cursor, statement, parameters, context, many):
        nonlocal select_count
        if statement.lstrip().upper().startswith("SELECT"):
            select_count += 1

    event.listen(engine, "before_cursor_execute", count)
    try:
        cls = sessionmaker(
            bind=consistent_read_factory(engine).kw["bind"],
            class_=Tracking,
            expire_on_commit=False,
        )
        centers, chunks = HybridAuthorityReader(cls, "v3", profile()).prepare(
            fuse_rrf(candidates, ())
        )
        assert select_count <= 4 and active_sessions == 0
        assert len(centers) == 2 and len(chunks) == 2
        assert all(not hasattr(x, "_sa_instance_state") for x in chunks.values())
    finally:
        event.remove(engine, "before_cursor_execute", count)


def test_old_hash_in_first_position_does_not_hide_current_identity(hybrid_factory):
    factory, _, m = hybrid_factory
    c = ready(factory, m)[0]
    old = c.model_copy(update={"index_hash": "f" * 64, "rank": 1})
    valid = c.model_copy(update={"rank": 2})
    centers, _ = reader(factory).prepare(fuse_rrf((old, valid), ()))
    assert len(centers) == 1 and centers[0].fused_candidate.dense.rank == 2
    assert centers[0].verified_chunk.index_hash == c.index_hash


@pytest.mark.parametrize("column", ["raw_text", "answer", "text"])
def test_unhashed_sql_content_changes_are_invalid_not_trimmed(hybrid_factory, column):
    factory, s, m = hybrid_factory
    candidates = ready(factory, m)
    with factory.begin() as session:
        if column == "raw_text":
            session.get(KnowledgeDocument, s.document.document_id).raw_text += " "
        else:
            for row in session.scalars(select(KnowledgeChunkRecord)):
                setattr(row, column, getattr(row, column) + " ")
    centers, chunks = reader(factory).prepare(fuse_rrf(candidates, ()))
    assert centers == () and chunks == {}


def test_version_profile_not_ready_and_sql_failure_are_faults(hybrid_factory):
    factory, _, m = hybrid_factory
    with factory.begin() as session:
        HybridIndexRepository(session).create_version("v3", profile(), m)
    candidate = HybridCandidate(
        index_version="v3",
        chunk_id="a" * 64,
        document_id="d" * 64,
        index_hash="c" * 64,
        route="dense",
        rank=1,
        raw_score=1.0,
    )
    with pytest.raises(HybridRetrievalError):
        reader(factory).prepare(fuse_rrf((candidate,), ()))

    class Fault:
        def __call__(self):
            raise OSError("private SQL")

    with pytest.raises(HybridRetrievalError):
        HybridAuthorityReader(Fault(), "v3", profile()).prepare(
            fuse_rrf((candidate,), ())
        )


def test_neighbor_query_failure_cannot_become_missing_neighbor(
    hybrid_factory, monkeypatch
):
    factory, _, m = hybrid_factory
    candidates = ready(factory, m)

    def fail(*a, **kw):
        raise OSError("private neighbor SQL")

    monkeypatch.setattr(HybridIndexRepository, "read_neighbor_rows", fail)
    with pytest.raises(HybridRetrievalError):
        reader(factory).prepare(fuse_rrf(candidates, ()))


def test_filter_all_hundred_then_choose_fifty(hybrid_factory):
    factory, _, _ = hybrid_factory
    s = source(
        "# 长政策\n" + "政策内容未拆封可以退货。" * 500,
        "many.md",
        max_chars=100,
        overlap=10,
    )
    seed_source(factory, s)
    m = build_corpus_manifest((s,), {"many.md": CONFIG}, max_chars=100, overlap=10)
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        repo.create_version("v3", profile(), m)
        snaps = repo.list_pending("v3", None, 1000)
        repo.confirm_done("v3", snaps)
        repo.mark_ready("v3")
    assert len(snaps) >= 70
    values = []
    for i in range(100):
        if i < 30:
            cid = f"{i + 1:064x}"
            ih = "f" * 64
        else:
            cid = snaps[i - 30].entry.manifest_chunk.chunk.chunk_id
            ih = snaps[i - 30].entry.index_hash
        values.append(
            HybridCandidate(
                index_version="v3",
                chunk_id=cid,
                document_id=s.document.document_id,
                index_hash=ih,
                route="dense" if i < 50 else "bm25",
                rank=i % 50 + 1,
                raw_score=0.5,
            )
        )
    fused = fuse_rrf(values[:50], values[50:])
    centers, _ = reader(factory).prepare(fused)
    assert len(centers) == 50
    expected = [
        c
        for c in fused
        if c.chunk_id in {x.entry.manifest_chunk.chunk.chunk_id for x in snaps}
    ][:50]
    assert [c.fused_candidate for c in centers] == expected
