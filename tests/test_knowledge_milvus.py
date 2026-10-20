"""v2 contracts reuse the v1 storage foundation with isolated RPC doubles."""
import importlib
from copy import deepcopy

import pytest
from pymilvus import DataType

from resolv.db.milvus import VectorStoreError
from resolv.schemas.knowledge import KnowledgeChunk, KnowledgeIndexProfile
from tests.test_milvus import FakeKnowledgeClient, knowledge_schema

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def _schema(dimension=1024):
    schema = knowledge_schema(dimension)
    schema["fields"].extend([
        {"name": "section_path", "type": DataType.VARCHAR, "params": {"max_length": 4096}},
        {"name": "content_type", "type": DataType.VARCHAR, "params": {"max_length": 255}},
        {"name": "index_hash", "type": DataType.VARCHAR, "params": {"max_length": 64}},
        {"name": "is_key_clause", "type": DataType.INT8, "params": {}},
    ])
    return schema


def _chunk(**changes):
    return KnowledgeChunk(**{
        "chunk_id": "c"*64, "document_id": "d"*64, "source": "refund.md",
        "chunk_index": 0, "category": "售后", "questions": "退款", "answer": "退款条款",
        "section_path": "售后 > 退款", "text": "售后 > 退款\n\n退款条款",
        "content_type": "refund", "is_key_clause": 1, **changes,
    })


def _entry(**changes):
    return importlib.import_module("resolv.knowledge").knowledge_index_entry(
        _chunk(**changes), KnowledgeIndexProfile(model_revision=REVISION)
    )


def _store(client):
    return importlib.import_module("resolv.repositories.knowledge_milvus").KnowledgeMilvusVectorStore(
        client, "resolv_knowledge_v2"
    )


def _vector():
    return [1.0] + [0.0]*1023


@pytest.fixture
def boundary():
    client = FakeKnowledgeClient()
    client.description = _schema()
    return client, _store(client)


def test_new_collection_exact_projection_schema_and_flat_cosine():
    client = FakeKnowledgeClient(exists=False)
    client.index = None
    store = _store(client)
    assert client.events == []
    assert store.collection_name == "resolv_knowledge_v2"
    store.ensure_collection()
    fields = {row["name"]: row for row in client.description["fields"]}
    assert set(fields) == {
        "chunk_id", "document_id", "chunk_index", "source", "text", "section_path",
        "content_type", "is_key_clause", "index_hash", "dense_vector",
    }
    assert fields["dense_vector"]["type"] == DataType.FLOAT_VECTOR
    assert fields["dense_vector"]["params"]["dim"] == 1024
    assert fields["is_key_clause"]["type"] == DataType.INT8
    assert fields["index_hash"]["params"]["max_length"] == 64
    assert client.index["index_type"] == "FLAT" and client.index["metric_type"] == "COSINE"


def test_v1_collection_is_rejected_without_recreation():
    client = FakeKnowledgeClient()
    with pytest.raises(VectorStoreError):
        _store(client).ensure_collection()
    assert not any(name in {"create", "create_index", "upsert"} for name, _ in client.events)


def test_v2_projection_is_explicit_and_same_pk_upsert_is_idempotent(boundary):
    client, store = boundary
    entry = _entry()
    assert store.upsert([entry], [_vector()]) == 1
    assert store.upsert([entry], [_vector()]) == 1
    assert len(client.rows) == 1
    row = client.rows["c"*64]
    assert set(row) == {
        "chunk_id", "document_id", "chunk_index", "source", "text", "section_path",
        "content_type", "is_key_clause", "index_hash", "dense_vector",
    }
    assert row["is_key_clause"] == 1 and row["section_path"] == "售后 > 退款"
    assert len(row["index_hash"]) == 64
    assert len(row["dense_vector"]) == 1024


def test_txt_empty_section_is_valid_and_not_added_to_dynamic_fields(boundary):
    client, store = boundary
    store.upsert([_entry(category="", questions="", section_path="", text="退款条款")], [_vector()])
    assert client.rows["c"*64]["section_path"] == ""


@pytest.mark.parametrize("changes", [
    {"source": "汉"*1366}, {"section_path": "汉"*1366},
    {"content_type": "汉"*86}, {"text": "汉"*2731},
    {"source": " "}, {"text": " "},
])
def test_utf8_capacity_and_blank_payload_fail_before_upsert(boundary, changes):
    client, store = boundary
    with pytest.raises(ValueError):
        store.upsert([_entry(**changes)], [_vector()])
    assert not client.rows
    assert not any(name == "upsert" for name, _ in client.events)


@pytest.mark.parametrize("dimension", [3, 0, True])
def test_v2_dimension_is_fixed(boundary, dimension):
    client, store = boundary
    with pytest.raises(ValueError):
        store.ensure_collection(dimension)
    assert client.events == []


@pytest.mark.parametrize("vector", [[1.0]*3, [float("nan")]*1024, [0.0]*1024, [True]*1024])
def test_bad_vectors_never_write(boundary, vector):
    client, store = boundary
    with pytest.raises(ValueError):
        store.upsert([_entry()], [vector])
    assert client.rows == {}


def test_duplicate_ids_count_mismatch_and_empty_batch(boundary):
    client, store = boundary
    with pytest.raises(ValueError):
        store.upsert([_entry(), _entry()], [_vector(), _vector()])
    with pytest.raises(ValueError):
        store.upsert([_entry()], [])
    assert store.upsert([], []) == 0
    assert client.rows == {}


@pytest.mark.parametrize("count", [True, -1, 2, "1", None])
def test_invalid_upsert_count_is_storage_error(boundary, count):
    client, store = boundary
    client.upsert_response = {"upsert_count": count}
    with pytest.raises(VectorStoreError):
        store.upsert([_entry()], [_vector()])


def test_partial_count_exposed_for_indexer_confirmation(boundary):
    client, store = boundary
    client.upsert_response = {"upsert_count": 0}
    assert store.upsert([_entry()], [_vector()]) == 0


def _hit(key="c", score=0.5):
    return {"id": key*64, "distance": score, "entity": {
        "chunk_id": key*64, "document_id": "d"*64, "index_hash": "e"*64,
        "text": "Untrusted index text is not authority",
    }}


def test_search_returns_only_candidates_and_preserves_provider_order(boundary):
    client, store = boundary
    client.search_response = [[_hit("a", 0.2), _hit("b", 0.9)]]
    hits = store.search(_vector(), 5)
    assert [(hit.chunk_id, hit.score) for hit in hits] == [("a"*64, 0.2), ("b"*64, 0.9)]
    assert set(hits[0].model_dump()) == {"chunk_id", "document_id", "index_hash", "score"}
    request = next(options for name, options in client.events if name == "search")
    assert request["limit"] == 5 and request["consistency_level"] == "Strong"
    assert set(request["output_fields"]) == {"chunk_id", "document_id", "index_hash"}
    assert not any(name in {"create", "create_index", "upsert"} for name, _ in client.events)


@pytest.mark.parametrize("bad", [
    [], {}, [None], [[{"id":"short","distance":0.5,"entity":{}}]],
    [[_hit(score=float("nan"))]], [[_hit(score=True)]],
    [[{"id":"a"*64, "chunk_id":"b"*64, "distance":0.5,"entity":{"document_id":"d"*64,"index_hash":"e"*64}}]],
    [[{"id":"c"*64,"distance":0.5,"entity":{"chunk_id":"f"*64,"document_id":"d"*64,"index_hash":"e"*64}}]],
    [[{"id":"c"*64,"distance":0.5,"entity":{"document_id":"d"*64,"index_hash":"short"}}]],
])
def test_invalid_result_cannot_escape_storage_boundary(boundary, bad):
    client, store = boundary
    client.search_response = deepcopy(bad)
    with pytest.raises(VectorStoreError):
        store.search(_vector(), 5)


def test_no_hits_is_legal(boundary):
    assert boundary[1].search(_vector(), 5) == []


@pytest.mark.parametrize("operation", ["upsert", "search"])
def test_rpc_failure_retains_cause_and_safe_message(boundary, operation):
    client, store = boundary
    client.error_at = operation
    with pytest.raises(VectorStoreError) as caught:
        if operation == "upsert":
            store.upsert([_entry()], [_vector()])
        else:
            store.search(_vector(), 5)
    assert caught.value.__cause__ is client.failure
    assert "private" not in str(caught.value)
