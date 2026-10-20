from copy import deepcopy

import pytest
from sqlalchemy import select

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.knowledge_base import KnowledgeBase
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.models.knowledge_hybrid import KnowledgeChunkIndex, KnowledgeIndexVersion
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.services.hybrid_manifest import build_corpus_manifest
from tests.hybrid_fixtures import profile
from tests.test_hybrid_manifest import CONFIG, source


def seed_source(factory, s):
    with factory.begin() as session:
        d = KnowledgeDocument(
            document_id=s.document.document_id,
            source=s.document.source,
            raw_text=s.document.text,
            content_type=CONFIG.content_type,
            ingest_hash=s.ingest_hash,
        )
        d.chunks = [
            KnowledgeChunkRecord(**c.model_dump(), index_hash="1" * 64)
            for c in s.chunks
        ]
        session.add(d)


@pytest.fixture
def hybrid_factory(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'hybrid.db'}")
    KnowledgeBase.metadata.create_all(engine)
    factory = create_session_factory(engine)
    s = source("# 退货\n一段不能拆封的政策。\n# 运费\n运费二十元。")
    seed_source(factory, s)
    yield factory, s, build_corpus_manifest((s,), {s.document.source: CONFIG})
    engine.dispose()


def legacy_rows(session):
    return [
        tuple(getattr(row, c.name) for c in row.__table__.columns)
        for cls in (KnowledgeDocument, KnowledgeChunkRecord)
        for row in session.scalars(select(cls))
    ]


def test_two_generations_preserve_legacy_rows(hybrid_factory):
    factory, _, manifest = hybrid_factory
    with factory() as session:
        before = deepcopy(legacy_rows(session))
    for version, collection in [
        ("one", "mhhybrid_gate_one"),
        ("two", "mhhybrid_gate_two"),
    ]:
        with factory.begin() as session:
            repo = HybridIndexRepository(session)
            repo.create_version(
                version,
                profile().model_copy(update={"collection_name": collection}),
                manifest,
            )
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        snapshots = repo.list_pending("one", None, 8)
        assert repo.begin_attempt("one", snapshots) == 2
        assert repo.confirm_done("one", snapshots) == 2
        repo.mark_ready("one")
    with factory() as session:
        repo = HybridIndexRepository(session)
        assert legacy_rows(session) == before
        assert (
            repo.get_version("one").state == "ready"
            and repo.get_version("two").state == "building"
        )
        assert len(repo.list_pending("two", None, 8)) == 2


def test_version_profile_and_manifest_conflicts_refused(hybrid_factory):
    factory, _, m = hybrid_factory
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        original = repo.create_version("one", profile(), m)
        assert repo.create_version("one", profile(), m) == original
        with pytest.raises(HybridRetrievalError):
            repo.create_version("two", profile(), m)
        with pytest.raises(HybridRetrievalError):
            repo.create_version(
                "one", profile().model_copy(update={"passage_max_length": 1024}), m
            )
        changed = m.model_copy(
            update={
                "documents": (
                    m.documents[0].model_copy(update={"document_hash": "f" * 64}),
                )
            }
        )
        with pytest.raises(HybridRetrievalError):
            repo.create_version("one", profile(), changed)


@pytest.mark.parametrize(
    "column,value",
    [
        ("answer", "tail"),
        ("text", "tail"),
        ("raw_text", "tail"),
        ("content_type", "other"),
    ],
)
def test_confirmation_guards_current_source_even_without_hash_change(
    hybrid_factory, column, value
):
    factory, s, m = hybrid_factory
    with factory.begin() as session:
        HybridIndexRepository(session).create_version("one", profile(), m)
    with factory() as session:
        snaps = HybridIndexRepository(session).list_pending("one", None, 8)
    with factory.begin() as session:
        cls = (
            KnowledgeDocument
            if column in ("raw_text", "content_type")
            else KnowledgeChunkRecord
        )
        key = (
            s.document.document_id if cls is KnowledgeDocument else s.chunks[0].chunk_id
        )
        row = session.get(cls, key)
        setattr(row, column, getattr(row, column) + " " if value == "tail" else value)
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        with pytest.raises(HybridRetrievalError):
            repo.confirm_done("one", snaps)
    with factory() as session:
        assert HybridIndexRepository(session).get_version("one").state == "building"
        assert all(
            x.index_status == "pending"
            for x in session.scalars(select(KnowledgeChunkIndex))
        )


def test_ready_requires_all_confirmations_and_source_still_matching(hybrid_factory):
    factory, s, m = hybrid_factory
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        repo.create_version("one", profile(), m)
        with pytest.raises(HybridRetrievalError):
            repo.mark_ready("one")
        snaps = repo.list_pending("one", None, 1)
        assert len(snaps) == 1 and repo.confirm_done("one", snaps) == 1
        with pytest.raises(HybridRetrievalError):
            repo.mark_ready("one")
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        assert repo.confirm_done("one", repo.list_pending("one", None, 8)) == 1
    with factory.begin() as session:
        session.get(KnowledgeDocument, s.document.document_id).raw_text += " "
    with factory.begin() as session, pytest.raises(HybridRetrievalError):
        HybridIndexRepository(session).mark_ready("one")


def test_tampered_persisted_profile_or_manifest_hash_rejected(hybrid_factory):
    factory, _, m = hybrid_factory
    with factory.begin() as session:
        HybridIndexRepository(session).create_version("one", profile(), m)
    with factory.begin() as session:
        session.get(KnowledgeIndexVersion, "one").profile_hash = "f" * 64
    with factory() as session, pytest.raises(HybridRetrievalError):
        HybridIndexRepository(session).get_version("one")


def test_safe_error_and_no_reconfirmation_of_done(hybrid_factory):
    factory, _, m = hybrid_factory
    with factory.begin() as session:
        repo = HybridIndexRepository(session)
        repo.create_version("one", profile(), m)
        snaps = repo.list_pending("one", None, 8)
        assert repo.record_error("one", snaps, "private URI secret") == 2
        assert all(
            "private" not in x.last_error
            for x in session.scalars(select(KnowledgeChunkIndex))
        )
        assert repo.confirm_done("one", snaps) == 2
        assert repo.confirm_done("one", snaps) == 0
