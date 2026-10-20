import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from resolv.schemas.hybrid_knowledge import (
    HybridRetrievalError,
    HybridRetrievalProfile,
)
from resolv.services.hybrid_retrieval import HybridKnowledgeRetriever
from tests.hybrid_fixtures import profile
from tests.test_hybrid_authority import reader, ready
from tests.test_knowledge_hybrid_repository import hybrid_factory

__all__ = ["hybrid_factory"]


class Embedding:
    profile = profile()
    dimension = 1024

    def __init__(self):
        self.queries = []
        self.active = 0
        self.maximum = 0

    def embed_query(self, q):
        self.queries.append(q)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        time.sleep(0.01)
        self.active -= 1
        if len(q) > 512:
            raise ValueError("query too long")
        return [1.0] + [0.0] * 1023


class Store:
    profile = profile()
    collection_name = profile.collection_name

    def __init__(self, dense=(), bm25=()):
        self.dense = dense
        self.bm25 = bm25
        self.queries = []
        self.error = None

    def search_dense(self, v, **kw):
        return self.dense

    def search_bm25(self, q, **kw):
        self.queries.append(q)
        if self.error:
            raise self.error
        return self.bm25


class Reranker:
    profile = HybridRetrievalProfile()
    query_tokens = 3
    pair_tokens = ()

    def __init__(self, engine=None):
        self.calls = 0
        self.engine = engine
        self.queries = []
        self.error = None

    def score(self, q, windows):
        if self.engine is not None:
            assert self.engine.pool.checkedout() == 0
        self.calls += 1
        self.queries.append(q)
        self.pair_tokens = (10,) * len(windows)
        if self.error:
            raise self.error
        return tuple(float(i) for i in range(len(windows)))


def retriever(factory, m, *, bm25_empty=True):
    candidates = ready(factory, m)
    embedding = Embedding()
    store = Store(
        candidates,
        ()
        if bm25_empty
        else tuple(c.model_copy(update={"route": "bm25"}) for c in candidates),
    )
    r = Reranker(factory.kw["bind"])
    service = HybridKnowledgeRetriever(
        embedding, store, reader(factory), r, profile(), r.profile
    )
    return service, embedding, store, r


def test_failed_bm25_never_calls_reranker(hybrid_factory):
    factory, _, m = hybrid_factory
    service, _, store, r = retriever(factory, m)
    store.error = RuntimeError("private storage failure")
    with pytest.raises(HybridRetrievalError):
        service.retrieve("退货")
    assert r.calls == 0


def test_original_query_and_single_route_empty_are_normal(hybrid_factory):
    factory, _, m = hybrid_factory
    service, e, s, r = retriever(factory, m)
    query = "  型号MX100可以退货吗？  "
    result = service.retrieve(query)
    assert e.queries == s.queries == r.queries == [query]
    assert len(result.windows) == 2 and len(result.chunks) == 2
    assert result.diagnostics.dense_count == 2 and result.diagnostics.bm25_count == 0
    assert (
        result.diagnostics.scored_count == 2 and result.diagnostics.selected_count == 2
    )
    assert result.diagnostics.query_tokens == 3
    assert all(t >= 0 for t in result.diagnostics.stage_seconds.values())


def test_separate_bm25_input_never_reaches_dense_or_reranker(hybrid_factory):
    factory, _, m = hybrid_factory
    service, embedding, store, reranker = retriever(factory, m)
    result = service.retrieve(
        "退货运费由谁承担？", bm25_text="退货运费由谁承担？\n邮费"
    )
    assert embedding.queries == reranker.queries == ["退货运费由谁承担？"]
    assert store.queries == ["退货运费由谁承担？\n邮费"]
    assert result.index_version == service.authority.version_id
    assert result.windows and result.diagnostics.bm25_count == 0


@pytest.mark.parametrize("invalid", ["", " \n", 42, [], {}, False])
def test_invalid_explicit_bm25_is_rejected_before_any_external_call(
    hybrid_factory, invalid
):
    factory, _, m = hybrid_factory
    service, embedding, store, reranker = retriever(factory, m)
    with pytest.raises(HybridRetrievalError, match="invalid_bm25_text"):
        service.retrieve("退货", bm25_text=invalid)
    assert embedding.queries == store.queries == reranker.queries == []


def test_explicit_none_preserves_single_input_contract(hybrid_factory):
    factory, _, m = hybrid_factory
    service, embedding, store, reranker = retriever(factory, m)
    result = service.retrieve("  退货  ", bm25_text=None)
    assert embedding.queries == store.queries == reranker.queries == ["  退货  "]
    assert result.windows


def test_separate_bm25_failure_never_returns_dense_only(hybrid_factory):
    factory, _, m = hybrid_factory
    service, _, store, reranker = retriever(factory, m)
    store.error = RuntimeError("BM25 unavailable")
    with pytest.raises(HybridRetrievalError):
        service.retrieve("退货", bm25_text="退货\n运费")
    assert store.queries == ["退货\n运费"] and reranker.calls == 0


def test_both_empty_skip_authority_and_reranker():
    e, s, r = Embedding(), Store(), Reranker()

    class Authority:
        version_id = "v3"
        profile = profile()

        def prepare(self, *a):
            pytest.fail("SQL for empty routes")

    result = HybridKnowledgeRetriever(
        e, s, Authority(), r, profile(), r.profile
    ).retrieve("独立query")
    assert result.windows == () and result.chunks == {} and r.calls == 0


@pytest.mark.parametrize("failure", ["long", "reranker", "invalid_score"])
def test_faults_never_return_partial_evidence(hybrid_factory, failure):
    factory, _, m = hybrid_factory
    service, _, _, r = retriever(factory, m)
    if failure == "reranker":
        r.error = RuntimeError("private model error")
    if failure == "invalid_score":
        r.score = lambda *a: (float("nan"),) * 2
    with pytest.raises(HybridRetrievalError):
        service.retrieve("q" * 513 if failure == "long" else "独立query")


def test_repeated_retrieval_serializes_inference(hybrid_factory):
    factory, _, m = hybrid_factory
    service, e, _, r = retriever(factory, m)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(service.retrieve, ("query1", "query2")))
    assert len(results) == 2 and e.maximum == 1 and r.calls == 2


def test_full_counts_fifty_windows_then_ten_evidence_and_unique_bodies():
    from resolv.schemas.hybrid_knowledge import HybridCandidate, VerifiedHybridCenter
    from tests.test_hybrid_windows import body, mapping

    chunks = mapping([body(i) for i in range(150)])
    dense = tuple(
        HybridCandidate(
            index_version="v3",
            chunk_id=f"{3 * i + 1:064x}",
            document_id="d" * 64,
            index_hash=chunks[f"{3 * i + 1:064x}"].index_hash,
            route="dense",
            rank=i + 1,
            raw_score=0.5,
        )
        for i in range(50)
    )
    bm25 = tuple(
        HybridCandidate(
            index_version="v3",
            chunk_id=f"{i + 1000:064x}",
            document_id="d" * 64,
            index_hash="f" * 64,
            route="bm25",
            rank=i + 1,
            raw_score=0.5,
        )
        for i in range(50)
    )

    class Authority:
        version_id = "v3"
        profile = profile()
        invalid_count = 50

        def prepare(self, fused):
            return tuple(
                VerifiedHybridCenter(
                    verified_chunk=chunks[c.chunk_id], fused_candidate=c
                )
                for c in fused
                if c.chunk_id in chunks
            ), chunks

    result = HybridKnowledgeRetriever(
        Embedding(),
        Store(dense, bm25),
        Authority(),
        Reranker(),
        profile(),
        HybridRetrievalProfile(),
    ).retrieve("query")
    d = result.diagnostics
    assert (
        d.dense_count,
        d.bm25_count,
        d.fused_count,
        d.invalid_count,
        d.center_count,
    ) == (50, 50, 100, 50, 50)
    assert (
        d.windows_before_merge,
        d.windows_after_merge,
        d.scored_count,
        d.selected_count,
    ) == (50, 50, 50, 10)
    assert len(result.chunks) == 30 and len(result.windows) == 10
