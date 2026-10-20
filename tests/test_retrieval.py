import importlib
import subprocess
import sys

import pytest

from resolv.db.milvus import VectorStoreError
from resolv.schemas.knowledge import SearchHit
from resolv.services.embedding import EmbeddingError


@pytest.fixture
def retrieval_boundary():
    module = importlib.import_module("resolv.services.retrieval")
    events = []
    state = {
        "embedding_error": None,
        "search_error": None,
        "hits": [
            SearchHit(
                chunk_id="c0", document_id="doc0", source="knowledge/refund.md",
                text="售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
                chunk_index=0, score=0.7,
            ),
            SearchHit(
                chunk_id="c1", document_id="doc1", source="knowledge/shipment.md",
                text="物流政策", chunk_index=1, score=-0.25,
            ),
        ],
    }

    class FakeEmbedding:
        def embed_query(self, text):
            events.append(("embed_query", text))
            if state["embedding_error"] is not None:
                raise state["embedding_error"]
            return [0.6, 0.8, 0.0]

        def embed_documents(self, texts):
            pytest.fail("Retriever must use query embedding")

        def close(self):
            pytest.fail("Caller owns the model lifecycle")

    class FakeVectorStore:
        def search(self, vector, top_k):
            events.append(("search", vector, top_k))
            if state["search_error"] is not None:
                raise state["search_error"]
            return state["hits"][:top_k]

        def ensure_collection(self, dimension):
            pytest.fail("Retriever must not prepare or create collections")

        def upsert(self, chunks, vectors):
            pytest.fail("Retriever must not ingest knowledge")

        def close(self):
            pytest.fail("Caller owns the vector-store client lifecycle")

    retriever = module.Retriever(
        embedding_service=FakeEmbedding(), vector_store=FakeVectorStore()
    )
    assert events == []
    return retriever, events, state


def test_query_embedding_then_search_with_default_k_preserves_search_hits(retrieval_boundary):
    retriever, events, _ = retrieval_boundary
    hits = retriever.retrieve("软件已经激活了，还能退款吗？")
    assert events == [
        ("embed_query", "软件已经激活了，还能退款吗？"),
        ("search", [0.6, 0.8, 0.0], 5),
    ]
    assert all(isinstance(hit, SearchHit) for hit in hits)
    assert [hit.model_dump() for hit in hits] == [
        {
            "chunk_id": "c0", "document_id": "doc0", "source": "knowledge/refund.md",
            "text": "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
            "chunk_index": 0, "score": 0.7,
        },
        {
            "chunk_id": "c1", "document_id": "doc1", "source": "knowledge/shipment.md",
            "text": "物流政策", "chunk_index": 1, "score": -0.25,
        },
    ]


@pytest.mark.parametrize("top_k,expected_ids", [(1, ["c0"]), (2, ["c0", "c1"]), (20, ["c0", "c1"])])
def test_explicit_top_k_reaches_vector_store(retrieval_boundary, top_k, expected_ids):
    retriever, events, _ = retrieval_boundary
    hits = retriever.retrieve("退款政策", top_k=top_k)
    assert events[-1] == ("search", [0.6, 0.8, 0.0], top_k)
    assert [hit.chunk_id for hit in hits] == expected_ids


def test_valid_query_is_passed_without_rewriting_or_stripping(retrieval_boundary):
    retriever, events, _ = retrieval_boundary
    retriever.retrieve(" \t软件激活了还能退款吗？\n ")
    assert events[0] == ("embed_query", " \t软件激活了还能退款吗？\n ")


@pytest.mark.parametrize("query", ["", " \t\n", "　", None, 42, False, ["query"]])
def test_invalid_query_fails_before_any_embedding_or_search(retrieval_boundary, query):
    retriever, events, _ = retrieval_boundary
    with pytest.raises(ValueError):
        retriever.retrieve(query)
    assert events == []


@pytest.mark.parametrize("top_k", [0, -1, True, False, 1.0, "5", None])
def test_invalid_top_k_fails_before_any_embedding_or_search(retrieval_boundary, top_k):
    retriever, events, _ = retrieval_boundary
    with pytest.raises(ValueError):
        retriever.retrieve("退款政策", top_k=top_k)
    assert events == []


def test_no_hits_is_an_empty_business_result(retrieval_boundary):
    retriever, events, state = retrieval_boundary
    state["hits"] = []
    assert retriever.retrieve("没有相关知识") == []
    assert [event[0] for event in events] == ["embed_query", "search"]


@pytest.mark.parametrize("error_type", [EmbeddingError, ValueError])
def test_embedding_failure_keeps_exception_and_cause_and_never_searches(retrieval_boundary, error_type):
    retriever, events, state = retrieval_boundary
    failure = error_type("query inference or token validation failed")
    cause = RuntimeError("private model detail")
    failure.__cause__ = cause
    state["embedding_error"] = failure
    with pytest.raises(error_type) as error:
        retriever.retrieve("退款政策")
    assert error.value is failure
    assert error.value.__cause__ is cause
    assert events == [("embed_query", "退款政策")]


@pytest.mark.parametrize("error_type", [VectorStoreError, ValueError])
def test_search_failure_is_not_no_hits_and_keeps_exception_chain(retrieval_boundary, error_type):
    retriever, events, state = retrieval_boundary
    failure = error_type("vector search or input validation failed")
    cause = RuntimeError("private RPC detail")
    failure.__cause__ = cause
    state["search_error"] = failure
    with pytest.raises(error_type) as error:
        retriever.retrieve("退款政策")
    assert error.value is failure
    assert error.value.__cause__ is cause
    assert events == [("embed_query", "退款政策"), ("search", [0.6, 0.8, 0.0], 5)]


def test_repeated_queries_reuse_resources_without_caching_results(retrieval_boundary):
    retriever, events, state = retrieval_boundary
    assert [hit.chunk_id for hit in retriever.retrieve("第一次问题")] == ["c0", "c1"]
    state["hits"] = [state["hits"][1]]
    assert [hit.chunk_id for hit in retriever.retrieve("第二次问题", top_k=1)] == ["c1"]
    assert events == [
        ("embed_query", "第一次问题"), ("search", [0.6, 0.8, 0.0], 5),
        ("embed_query", "第二次问题"), ("search", [0.6, 0.8, 0.0], 1),
    ]


def test_fresh_import_neither_loads_model_nor_connects_milvus():
    result = subprocess.run(
        [
            sys.executable, "-c",
            """
import sys
import pymilvus
import resolv.core.config as config
import resolv.db.milvus as db
import resolv.services.embedding as embedding

def unexpected(*args, **kwargs):
    raise AssertionError('import must not construct external resources')

embedding._load_model = unexpected
embedding.EmbeddingService = unexpected
db.create_milvus_client = unexpected
db.MilvusClient = unexpected
pymilvus.MilvusClient = unexpected
pymilvus.connections.connect = unexpected
config.MilvusSettings = unexpected
config.EmbeddingSettings = unexpected
import resolv.services.retrieval
assert 'FlagEmbedding' not in sys.modules
assert 'torch' not in sys.modules
""",
        ],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
