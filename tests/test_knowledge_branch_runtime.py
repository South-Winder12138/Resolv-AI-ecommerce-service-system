"""Offline pool lifetime/version/concurrency behavior; not real MPS evidence."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from resolv.schemas.hybrid_knowledge import (
    HybridIndexProfile,
    HybridRetrievalError,
    HybridRetrievalProfile,
    HybridVersionSnapshot,
)
from resolv.schemas.knowledge_branch import KnowledgeBranchError
from resolv.schemas.knowledge_release import ReleaseSelection
from resolv.services import knowledge_branch_runtime as module
from resolv.services.hybrid_fingerprints import index_hash, manifest_hash, profile_hash
from tests.hybrid_fixtures import manifest
from tests.test_hybrid_retrieval import Embedding, Reranker, Store


def selection(version="old"):
    p = HybridIndexProfile(collection_name="knowledge_" + version)
    return ReleaseSelection(
        status="configured",
        release_id="release_" + version,
        index_version=version,
        collection=p.collection_name,
        profile_hash=profile_hash(p),
        manifest_hash=manifest_hash(manifest()),
        profile=p,
    )


@pytest.fixture
def resources(monkeypatch):
    events, loads = [], {"dense": 0, "reranker": 0}
    fail = {"reranker": False}
    active, release = Event(), Event()
    block = {"enabled": False}

    @contextmanager
    def engine(*args):
        events.append("engine-open")
        try:
            yield object()
        finally:
            events.append("engine-close")

    @contextmanager
    def session():
        yield object()

    class Client:
        def query(self, **kwargs):
            return [{"count(*)": 1}]

        def get(self, **kwargs):
            version = kwargs["collection_name"].removeprefix("knowledge_")
            s = selection(version)
            c = manifest().documents[0].chunks[0]
            return [
                {
                    "chunk_id": c.chunk.chunk_id,
                    "document_id": c.chunk.document_id,
                    "index_version": version,
                    "index_hash": index_hash(c, s.profile, version),
                    "text": c.chunk.text,
                }
            ]

    @contextmanager
    def milvus(*args):
        events.append("client-open")
        try:
            yield Client()
        finally:
            events.append("client-close")

    class Repo:
        def __init__(self, session):
            pass

        def get_version(self, version):
            return HybridVersionSnapshot(
                version_id=version,
                profile=selection(version).profile,
                manifest=manifest(),
                state="ready",
            )

        def summary(self, version):
            return SimpleNamespace(
                expected_count=1, done_count=1, pending_count=0, state="ready"
            )

    class Dense(Embedding):
        def __init__(self, profile, **kwargs):
            super().__init__()
            self.profile = profile
            loads["dense"] += 1

        def embed_query(self, query):
            if block["enabled"]:
                active.set()
                assert release.wait(5)
            return super().embed_query(query)

        def close(self):
            events.append("dense-close")

    class Ranker(Reranker):
        def __init__(self, profile, **kwargs):
            if fail["reranker"]:
                raise HybridRetrievalError("reranker_mps_unavailable")
            super().__init__()
            self.profile = profile
            loads["reranker"] += 1

        def count_query_tokens(self, text):
            return len(text)

        def close(self):
            events.append("reranker-close")

    class Vector(Store):
        def __init__(self, client, p):
            super().__init__()
            self.client, self.profile, self.collection_name = (
                client,
                p,
                p.collection_name,
            )

        def validate_existing(self):
            events.append("validate:" + self.collection_name)

    monkeypatch.setattr(module, "knowledge_engine", engine)
    monkeypatch.setattr(module, "milvus_client", milvus)
    monkeypatch.setattr(module, "consistent_read_factory", lambda _: session)
    monkeypatch.setattr(module, "HybridIndexRepository", Repo)
    monkeypatch.setattr(module, "HybridEmbeddingAdapter", Dense)
    monkeypatch.setattr(module, "LocalHybridReranker", Ranker)
    monkeypatch.setattr(module, "HybridMilvusStore", Vector)
    return SimpleNamespace(
        events=events,
        loads=loads,
        fail=fail,
        block=block,
        active=active,
        release=release,
    )


def pool():
    return module.KnowledgeRetrieverPool(
        knowledge_settings=object(),
        milvus_settings=object(),
        retrieval_profile=HybridRetrievalProfile(),
    )


def test_selector_states_never_initialize_resources(tmp_path, resources):
    path = tmp_path / "active.json"
    assert module.read_release_selection(path).status == "disabled"
    path.write_text('{"enabled":false,"config_version":1}')
    assert module.read_release_selection(path).status == "disabled"
    for text in [
        "{}",
        '{"enabled":true}',
        '{"enabled":false,"unknown":1}',
        '{"enabled":false,"enabled":true}',
    ]:
        path.write_text(text)
        assert module.read_release_selection(path).status == "invalid"
    owner = pool()
    for status in ("disabled", "invalid"):
        s = ReleaseSelection(
            status=status, error_code="invalid" if status == "invalid" else None
        )
        with pytest.raises(KnowledgeBranchError):
            owner.validate_selection(s)
    owner.close()
    assert resources.loads == {"dense": 0, "reranker": 0} and resources.events == []


def test_valid_selector_freezes_explicit_default_profile(tmp_path, resources):
    s = selection()
    path = tmp_path / "active.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "config_version": 1,
                **s.model_dump(
                    mode="json", exclude={"status", "profile", "error_code"}
                ),
            }
        )
    )
    assert module.read_release_selection(path) == s


def test_versions_use_distinct_bindings_and_load_models_once(resources):
    owner = pool()
    assert resources.events == []
    assert owner.validate_selection(selection()).version_id == "old"
    assert resources.loads == {"dense": 0, "reranker": 0}
    results = [
        owner.retrieve(selection(v), "query", bm25_text="query\nextra")
        for v in ("old", "new", "old")
    ]
    assert [r.index_version for r in results] == ["old", "new", "old"]
    assert resources.loads == {"dense": 1, "reranker": 1}
    owner.close()
    assert (
        resources.events.count("dense-close")
        == resources.events.count("reranker-close")
        == 1
    )
    with pytest.raises(KnowledgeBranchError, match="knowledge_runtime_closed"):
        owner.retrieve(selection(), "query", bm25_text="query")


def test_wait_timeout_never_enters_inference(resources):
    owner = pool()
    with module.INFERENCE_LOCK, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            owner.retrieve, selection(), "query", bm25_text="q", timeout_seconds=0.01
        )
        with pytest.raises(KnowledgeBranchError, match="knowledge_inference_busy"):
            future.result(timeout=2)
    assert resources.loads == {"dense": 0, "reranker": 0}
    owner.close()


def test_mps_failure_closes_dense_without_cpu_fallback(resources):
    owner = pool()
    resources.fail["reranker"] = True
    with pytest.raises(HybridRetrievalError, match="reranker_mps_unavailable"):
        owner.retrieve(selection(), "query", bm25_text="q")
    assert resources.loads == {"dense": 1, "reranker": 0}
    assert "dense-close" in resources.events
    owner.close()


def test_count_query_reuses_ranker_without_dense_inference(resources):
    owner = pool()
    assert owner.count_query_tokens("查询") == 2
    assert resources.loads == {"dense": 0, "reranker": 1}
    owner.retrieve(selection(), "query", bm25_text="query")
    assert resources.loads == {"dense": 1, "reranker": 1}
    owner.close()


def test_ready_sql_flag_cannot_hide_missing_collection_rows(resources):
    owner = pool()
    owner._connect()
    owner._client.query = lambda **kwargs: [{"count(*)": 0}]
    with pytest.raises(
        KnowledgeBranchError, match="knowledge_release_collection_incomplete"
    ):
        owner.validate_selection(selection())
    assert resources.loads == {"dense": 0, "reranker": 0}
    owner.close()


def test_retrieval_error_releases_lock_for_subsequent_call(resources):
    owner = pool()
    owner.retrieve(selection(), "query", bm25_text="query")
    original = owner._dense.embed_query

    def fail(query):
        raise RuntimeError("synthetic inference failure")

    owner._dense.embed_query = fail
    with pytest.raises(HybridRetrievalError):
        owner.retrieve(selection(), "query", bm25_text="query")
    owner._dense.embed_query = original
    assert (
        owner.retrieve(selection(), "query", bm25_text="query").index_version == "old"
    )
    owner.close()


def test_close_waits_for_active_inference(resources):
    owner = pool()
    resources.block["enabled"] = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(owner.retrieve, selection(), "query", bm25_text="q")
        assert resources.active.wait(3)
        closed = Event()
        thread = Thread(target=lambda: (owner.close(), closed.set()))
        thread.start()
        assert not closed.wait(0.03) and "dense-close" not in resources.events
        resources.release.set()
        assert future.result(timeout=3).index_version == "old"
        thread.join(3)
        assert closed.is_set()
