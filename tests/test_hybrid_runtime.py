from contextlib import contextmanager

import pytest

from resolv.schemas.hybrid_knowledge import (
    HybridRetrievalError,
    HybridRetrievalProfile,
)
from resolv.services import hybrid_runtime as module
from tests.hybrid_fixtures import profile
from tests.test_hybrid_authority import ready
from tests.test_hybrid_retrieval import Embedding, Reranker, Store
from tests.test_knowledge_hybrid_repository import hybrid_factory

__all__ = ["hybrid_factory"]


@pytest.fixture
def resources(hybrid_factory, monkeypatch):
    monkeypatch.setenv(
        "KNOWLEDGE_DATABASE_URL",
        "mysql+pymysql://offline:offline@127.0.0.1:3307/unused?charset=utf8mb4",
    )
    factory, _, m = hybrid_factory
    ready(factory, m)
    events = []
    counts = {"dense": 0, "reranker": 0}
    fail = {"init": False, "close": False}

    @contextmanager
    def engine(*a):
        try:
            yield factory.kw["bind"]
        finally:
            events.append("engine-close")

    @contextmanager
    def client(*a):
        try:
            yield object()
        finally:
            events.append("client-close")

    class Dense(Embedding):
        def __init__(self, p, **kw):
            super().__init__()
            self.profile = p
            counts["dense"] += 1

        def close(self):
            events.append("dense-close")

    class Ranker(Reranker):
        def __init__(self, p, **kw):
            if fail["init"]:
                raise RuntimeError("private init")
            super().__init__()
            self.profile = p
            counts["reranker"] += 1

        def close(self):
            events.append("reranker-close")
            if fail["close"]:
                raise RuntimeError("private close")

    class Vector(Store):
        def __init__(self, client, p):
            super().__init__()
            self.profile = p
            self.collection_name = p.collection_name

        def validate_existing(self):
            pass

    monkeypatch.setattr(module, "knowledge_engine", engine)
    monkeypatch.setattr(module, "milvus_client", client)
    monkeypatch.setattr(module, "HybridEmbeddingAdapter", Dense)
    monkeypatch.setattr(module, "LocalHybridReranker", Ranker)
    monkeypatch.setattr(module, "HybridMilvusStore", Vector)
    return events, counts, fail


def context():
    return module.hybrid_retrieval_runtime(
        version_id="v3",
        index_profile=profile(),
        retrieval_profile=HybridRetrievalProfile(),
    )


def test_runtime_loads_each_model_once_and_closes_all_owners(resources):
    events, counts, _ = resources
    with context() as service:
        assert service.retrieve("query1").windows == ()
        assert service.retrieve("query2").windows == ()
    assert counts == {"dense": 1, "reranker": 1}
    assert events == ["reranker-close", "dense-close", "client-close", "engine-close"]
    with pytest.raises(HybridRetrievalError):
        service.retrieve("after-close")


def test_initialization_failure_closes_prior_owners(resources):
    events, _, fail = resources
    fail["init"] = True
    with pytest.raises(HybridRetrievalError), context():
        pass
    assert events == ["dense-close", "client-close", "engine-close"]


def test_close_failure_does_not_retract_completed_persistent_result(resources):
    events, _, fail = resources
    fail["close"] = True
    result = None
    with pytest.raises(HybridRetrievalError), context() as service:
        result = service.retrieve("query1")
    assert result.index_version == "v3" and result.windows == ()
    assert events == ["reranker-close", "dense-close", "client-close", "engine-close"]
