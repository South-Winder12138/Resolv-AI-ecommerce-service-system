import importlib
import logging
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
from pymilvus import DataType, MilvusClient

from resolv.core.config import MilvusSettings
from resolv.schemas.knowledge import Chunk, SearchHit


def knowledge_schema(dimension=3):
    return {
        "auto_id": False,
        "enable_dynamic_field": False,
        "fields": [
            {
                "name": "chunk_id",
                "type": DataType.VARCHAR,
                "is_primary": True,
                "params": {"max_length": 64},
            },
            {
                "name": "document_id",
                "type": DataType.VARCHAR,
                "params": {"max_length": 64},
            },
            {"name": "text", "type": DataType.VARCHAR, "params": {"max_length": 8192}},
            {
                "name": "source",
                "type": DataType.VARCHAR,
                "params": {"max_length": 4096},
            },
            {"name": "chunk_index", "type": DataType.INT64, "params": {}},
            {
                "name": "dense_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": dimension},
            },
        ],
    }


class FakeKnowledgeClient:
    create_schema = staticmethod(MilvusClient.create_schema)
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self, exists=True):
        self.exists = exists
        self.description = knowledge_schema()
        self.index = {
            "field_name": "dense_vector",
            "index_name": "dense_index",
            "index_type": "FLAT",
            "metric_type": "COSINE",
            "state": "Finished",
        }
        self.events = []
        self.error_at = None
        self.failure = RuntimeError("private RPC details")
        self.rows = {}
        self.upsert_response = None
        self.search_response = [[]]

    def record(self, method, kwargs):
        self.events.append((method, kwargs))
        if self.error_at == method:
            raise self.failure

    def has_collection(self, **kwargs):
        self.record("has", kwargs)
        return self.exists

    def create_collection(self, **kwargs):
        self.record("create", kwargs)
        self.description = kwargs["schema"].to_dict()
        self.exists = True

    def describe_collection(self, **kwargs):
        self.record("describe", kwargs)
        if not self.exists:
            raise self.failure
        return deepcopy(self.description)

    def list_indexes(self, **kwargs):
        self.record("indexes", kwargs)
        return [self.index["index_name"]] if self.index else []

    def describe_index(self, **kwargs):
        self.record("describe_index", kwargs)
        return deepcopy(self.index)

    def create_index(self, **kwargs):
        self.record("create_index", kwargs)
        self.index = kwargs["index_params"][0].to_dict()

    def load_collection(self, **kwargs):
        self.record("load", kwargs)

    def upsert(self, **kwargs):
        self.record("upsert", kwargs)
        self.rows.update({row["chunk_id"]: deepcopy(row) for row in kwargs["data"]})
        return (
            self.upsert_response
            if self.upsert_response is not None
            else {"upsert_count": len(kwargs["data"])}
        )

    def search(self, **kwargs):
        self.record("search", kwargs)
        return deepcopy(self.search_response)

    def drop_collection(self, **kwargs):
        pytest.fail("repository must never drop a collection")

    def drop_index(self, **kwargs):
        pytest.fail("repository must never replace an existing index")


@pytest.fixture
def vector_boundary():
    module = importlib.import_module("resolv.repositories.milvus")
    client = FakeKnowledgeClient()
    return module, client, module.MilvusVectorStore(client, "knowledge")


def test_vector_store_constructor_does_not_call_client(vector_boundary):
    _, client, _ = vector_boundary
    assert client.events == []


def test_ensure_creates_exact_six_fields_and_cosine_flat_index(vector_boundary):
    _, client, store = vector_boundary
    client.exists = False
    client.index = None
    store.ensure_collection(3)
    fields = client.description["fields"]
    assert [(f["name"], f["type"]) for f in fields] == [
        ("chunk_id", DataType.VARCHAR),
        ("document_id", DataType.VARCHAR),
        ("text", DataType.VARCHAR),
        ("source", DataType.VARCHAR),
        ("chunk_index", DataType.INT64),
        ("dense_vector", DataType.FLOAT_VECTOR),
    ]
    assert fields[0]["is_primary"] is True
    assert client.description["auto_id"] is False
    assert client.description["enable_dynamic_field"] is False
    assert [f.get("params", {}).get("max_length") for f in fields[:4]] == [
        64,
        64,
        8192,
        4096,
    ]
    assert fields[5]["params"]["dim"] == 3
    assert client.index["index_type"] == "FLAT"
    assert client.index["metric_type"] == "COSINE"
    assert client.events[-1][0] == "load"
    assert all(event[1].get("timeout") for event in client.events)


def test_ensure_existing_compatible_collection_is_not_recreated(vector_boundary):
    _, client, store = vector_boundary
    store.ensure_collection(3)
    store.ensure_collection(3)
    assert not any(name in {"create", "create_index"} for name, _ in client.events)


def test_ensure_retries_compatible_collection_missing_index(vector_boundary):
    _, client, store = vector_boundary
    client.index = None
    store.ensure_collection(3)
    assert [name for name, _ in client.events].count("create_index") == 1
    assert not any(name == "create" for name, _ in client.events)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "auto_id",
        "dynamic",
        "primary",
        "type",
        "dimension",
        "text_capacity",
        "nullable",
        "metric",
        "index_type",
        "index_field",
    ],
)
def test_ensure_rejects_incompatible_existing_without_writes(vector_boundary, mutation):
    module, client, store = vector_boundary
    if mutation == "missing":
        client.description["fields"].pop()
    elif mutation == "extra":
        client.description["fields"].append({"name": "extra", "type": DataType.INT64})
    elif mutation == "auto_id":
        client.description["auto_id"] = True
    elif mutation == "dynamic":
        client.description["enable_dynamic_field"] = True
    elif mutation == "primary":
        client.description["fields"][0]["is_primary"] = False
    elif mutation == "type":
        client.description["fields"][5]["type"] = DataType.BINARY_VECTOR
    elif mutation == "dimension":
        client.description["fields"][5]["params"]["dim"] = 4
    elif mutation == "text_capacity":
        client.description["fields"][2]["params"]["max_length"] = 10
    elif mutation == "nullable":
        client.description["fields"][2]["nullable"] = True
    else:
        field, value = {
            "metric": ("metric_type", "L2"),
            "index_type": ("index_type", "HNSW"),
            "index_field": ("field_name", "other"),
        }[mutation]
        client.index[field] = value
    with pytest.raises(module.VectorStoreError):
        store.ensure_collection(3)
    assert not any(
        name in {"create", "create_index", "load"} for name, _ in client.events
    )


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_invalid_collection_dimension_does_not_rpc(vector_boundary, dimension):
    _, client, store = vector_boundary
    with pytest.raises(ValueError):
        store.ensure_collection(dimension)
    assert client.events == []


@pytest.mark.parametrize(
    "stage",
    ["has", "create", "describe", "indexes", "describe_index", "create_index", "load"],
)
def test_collection_rpc_failures_preserve_cause(vector_boundary, stage):
    module, client, store = vector_boundary
    client.error_at = stage
    if stage == "create":
        client.exists = False
    if stage == "create_index":
        client.index = None
    with pytest.raises(module.VectorStoreError) as error:
        store.ensure_collection(3)
    assert error.value.__cause__ is client.failure
    assert "private" not in str(error.value)


def knowledge_chunk(index=0, **overrides):
    return Chunk.model_construct(
        **{
            "chunk_id": f"c{index}",
            "document_id": "doc",
            "text": "退款政策",
            "source": "knowledge/refund.md",
            "chunk_index": index,
            **overrides,
        }
    )


@pytest.fixture
def ready_vector_boundary(vector_boundary):
    module, client, store = vector_boundary
    store.ensure_collection(3)
    client.events.clear()
    return module, client, store


def test_upsert_pairs_rows_in_order_and_reuses_primary_keys(ready_vector_boundary):
    _, client, store = ready_vector_boundary
    chunks = [knowledge_chunk(0), knowledge_chunk(1)]
    vectors = [[1, 0, 0], [0, 2, 0]]
    assert store.upsert(chunks, vectors) == 2
    assert store.upsert(chunks, vectors) == 2
    assert list(client.rows) == ["c0", "c1"]
    assert client.rows["c1"] == {
        "chunk_id": "c1",
        "document_id": "doc",
        "text": "退款政策",
        "source": "knowledge/refund.md",
        "chunk_index": 1,
        "dense_vector": [0.0, 2.0, 0.0],
    }
    assert client.events[0][1]["data"][0]["dense_vector"] == [1.0, 0.0, 0.0]


def test_empty_upsert_does_not_contact_client(vector_boundary):
    _, client, store = vector_boundary
    assert store.upsert([], []) == 0
    assert client.events == []


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0]],
        [[0, 0, 0]],
        [[np.nan, 0, 0]],
        [[np.inf, 0, 0]],
        [[1e300, 0, 0]],
        [["1", "0", "0"]],
        [[True, False, False]],
        [[1 + 2j, 0, 0]],
    ],
)
def test_invalid_upsert_vectors_do_not_write(ready_vector_boundary, vectors):
    _, client, store = ready_vector_boundary
    with pytest.raises(ValueError):
        store.upsert([knowledge_chunk()], vectors)
    assert client.events == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_id", ""),
        ("chunk_id", "a" * 65),
        ("document_id", "a" * 65),
        ("text", " "),
        ("text", "中" * 2731),
        ("source", "中" * 1366),
        ("chunk_index", -1),
        ("chunk_index", 2**63),
        ("chunk_index", True),
    ],
)
def test_invalid_tail_chunk_prevents_entire_write(ready_vector_boundary, field, value):
    _, client, store = ready_vector_boundary
    chunks = [knowledge_chunk(0), knowledge_chunk(1, **{field: value})]
    with pytest.raises(ValueError):
        store.upsert(chunks, [[1, 0, 0], [0, 1, 0]])
    assert client.events == []


def test_varchar_exact_utf8_budget_accepted(ready_vector_boundary):
    _, client, store = ready_vector_boundary
    chunk = knowledge_chunk(
        chunk_id="a" * 64,
        document_id="b" * 64,
        text="中" * 2730 + "ab",
        source="中" * 1365 + "a",
    )
    assert store.upsert([chunk], [[1, 0, 0]]) == 1
    assert len(client.rows["a" * 64]["text"].encode()) == 8192


def test_duplicate_primary_keys_in_batch_rejected(ready_vector_boundary):
    _, client, store = ready_vector_boundary
    with pytest.raises(ValueError):
        store.upsert([knowledge_chunk(), knowledge_chunk()], [[1, 0, 0], [0, 1, 0]])
    assert client.events == []


def test_upsert_reports_actual_count(ready_vector_boundary):
    _, client, store = ready_vector_boundary
    client.upsert_response = {"upsert_count": 1}
    assert (
        store.upsert([knowledge_chunk(0), knowledge_chunk(1)], [[1, 0, 0], [0, 1, 0]])
        == 1
    )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"upsert_count": -1},
        {"upsert_count": 2},
        {"upsert_count": True},
        {"upsert_count": "1"},
    ],
)
def test_bad_upsert_response_is_storage_error(ready_vector_boundary, response):
    module, client, store = ready_vector_boundary
    client.upsert_response = response
    with pytest.raises(module.VectorStoreError) as error:
        store.upsert([knowledge_chunk()], [[1, 0, 0]])
    assert error.value.__cause__ is not None


def raw_hit(index=0, score=0.8):
    return {
        "id": f"c{index}",
        "distance": score,
        "entity": knowledge_chunk(index).model_dump(),
    }


def test_search_maps_cosine_hits_sorts_limits_and_uses_caller_k(ready_vector_boundary):
    _, client, store = ready_vector_boundary
    client.search_response = [[raw_hit(0, -0.2), raw_hit(1, 0.9), raw_hit(2, 0.2)]]
    hits = store.search([1, 0, 0], top_k=2)
    assert all(isinstance(hit, SearchHit) for hit in hits)
    assert [(hit.chunk_id, hit.score) for hit in hits] == [("c1", 0.9), ("c2", 0.2)]
    assert hits[0].source == "knowledge/refund.md"
    assert client.events == [
        (
            "search",
            {
                "collection_name": "knowledge",
                "data": [[1.0, 0.0, 0.0]],
                "anns_field": "dense_vector",
                "limit": 2,
                "output_fields": [
                    "chunk_id",
                    "document_id",
                    "text",
                    "source",
                    "chunk_index",
                ],
                "search_params": {"metric_type": "COSINE", "params": {}},
                "consistency_level": "Strong",
                "timeout": 10.0,
            },
        )
    ]


def test_empty_search_is_success(ready_vector_boundary):
    _, _, store = ready_vector_boundary
    assert store.search([1, 0, 0], 5) == []


@pytest.mark.parametrize("sdk_hit", [False, True])
def test_search_maps_named_primary_key_returned_by_pymilvus(
    ready_vector_boundary, sdk_hit
):
    from pymilvus.client.search_result import Hit

    _, client, store = ready_vector_boundary
    raw = raw_hit()
    raw["chunk_id"] = raw.pop("id")
    client.search_response = [[Hit(raw, pk_name="chunk_id") if sdk_hit else raw]]

    hits = store.search([1, 0, 0], 5)

    assert len(hits) == 1
    assert hits[0].model_dump() == {**knowledge_chunk().model_dump(), "score": 0.8}


@pytest.mark.parametrize(
    "raw",
    [
        {**raw_hit(), "chunk_id": "different"},
        {**raw_hit(), "chunk_id": "c0", "id": "different"},
        {
            "chunk_id": "different",
            "distance": 0.8,
            "entity": knowledge_chunk().model_dump(),
        },
    ],
)
def test_search_rejects_conflicting_primary_key_fields(ready_vector_boundary, raw):
    module, client, store = ready_vector_boundary
    client.search_response = [[raw]]

    with pytest.raises(module.VectorStoreError) as error:
        store.search([1, 0, 0], 5)

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "5"])
def test_invalid_top_k_does_not_rpc(ready_vector_boundary, k):
    _, client, store = ready_vector_boundary
    with pytest.raises(ValueError):
        store.search([1, 0, 0], k)
    assert client.events == []


@pytest.mark.parametrize(
    "vector", [[], [1, 0], [0, 0, 0], [np.nan, 0, 0], ["1", "0", "0"]]
)
def test_invalid_search_vector_does_not_rpc(ready_vector_boundary, vector):
    _, client, store = ready_vector_boundary
    with pytest.raises(ValueError):
        store.search(vector, 5)
    assert client.events == []


@pytest.mark.parametrize(
    "response",
    [
        [],
        [[], []],
        [[{"id": "c0"}]],
        [[raw_hit(score=np.inf)]],
        [
            [
                {
                    "id": "different",
                    "distance": 0.8,
                    "entity": knowledge_chunk().model_dump(),
                }
            ]
        ],
    ],
)
def test_malformed_search_output_is_error(ready_vector_boundary, response):
    module, client, store = ready_vector_boundary
    client.search_response = response
    with pytest.raises(module.VectorStoreError) as error:
        store.search([1, 0, 0], 5)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize("response", [[{}], [""], [b""], [()], {0: []}, ([],)])
def test_malformed_empty_search_containers_are_not_no_hits(ready_vector_boundary, response):
    module, client, store = ready_vector_boundary
    client.search_response = response
    with pytest.raises(module.VectorStoreError) as error:
        store.search([1, 0, 0], 5)
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("operation", ["upsert", "search"])
def test_data_rpc_failure_is_not_empty_success(ready_vector_boundary, operation):
    module, client, store = ready_vector_boundary
    client.error_at = operation
    with pytest.raises(module.VectorStoreError) as error:
        if operation == "upsert":
            store.upsert([knowledge_chunk()], [[1, 0, 0]])
        else:
            store.search([1, 0, 0], 5)
    assert error.value.__cause__ is client.failure
    assert "private" not in str(error.value)


def test_search_existing_collection_verifies_without_creation(vector_boundary):
    _, client, store = vector_boundary
    store.search([1, 0, 0], 5)
    assert client.events[-1][0] == "search"
    assert not any(name in {"create", "create_index"} for name, _ in client.events)


def test_search_missing_collection_never_creates_or_returns_empty(vector_boundary):
    module, client, store = vector_boundary
    client.exists = False
    with pytest.raises(module.VectorStoreError):
        store.search([1, 0, 0], 5)
    assert [name for name, _ in client.events] == ["describe"]


def test_factory_passes_uri_and_timeout(monkeypatch) -> None:
    milvus = importlib.import_module("resolv.db.milvus")
    recorded = {}
    sentinel = object()

    def constructor(**kwargs):
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr(milvus, "MilvusClient", constructor)
    settings = MilvusSettings(milvus_uri="http://localhost:19531", _env_file=None)
    assert milvus.create_milvus_client(settings) is sentinel
    assert recorded == {"uri": "http://localhost:19531", "timeout": 10.0}


def test_factory_wraps_connection_failure(monkeypatch) -> None:
    milvus = importlib.import_module("resolv.db.milvus")
    backend_error = RuntimeError("secret credentials")

    def broken_constructor(**kwargs):
        raise backend_error

    monkeypatch.setattr(milvus, "MilvusClient", broken_constructor)
    with pytest.raises(milvus.VectorStoreError) as captured:
        milvus.create_milvus_client(MilvusSettings(_env_file=None))
    assert captured.value.__cause__ is backend_error
    assert "secret" not in str(captured.value)


def test_import_does_not_connect(monkeypatch) -> None:
    import pymilvus

    def forbidden_constructor(**kwargs):
        pytest.fail("Import must not connect to Milvus")

    with monkeypatch.context() as patch:
        patch.setattr(pymilvus, "MilvusClient", forbidden_constructor)
        milvus = importlib.import_module("resolv.db.milvus")
        importlib.reload(milvus)
    importlib.reload(milvus)


@pytest.fixture
def client_boundary(monkeypatch):
    milvus = importlib.import_module("resolv.db.milvus")

    class FakeClient:
        close_calls = 0
        close_error = None

        def close(self):
            self.close_calls += 1
            if self.close_error:
                raise self.close_error

    client = FakeClient()
    monkeypatch.setattr(milvus, "MilvusClient", lambda **kwargs: client)
    return milvus, client


def test_context_closes_after_success(client_boundary) -> None:
    milvus, client = client_boundary
    with milvus.milvus_client(MilvusSettings(_env_file=None)) as actual:
        assert actual is client
        assert client.close_calls == 0
    assert client.close_calls == 1


def test_context_closes_after_body_failure(client_boundary) -> None:
    milvus, client = client_boundary
    original = RuntimeError("RPC failed")
    with (
        pytest.raises(RuntimeError) as captured,
        milvus.milvus_client(MilvusSettings(_env_file=None)),
    ):
        raise original
    assert captured.value is original
    assert client.close_calls == 1


def test_context_reports_close_failure(client_boundary) -> None:
    milvus, client = client_boundary
    original = RuntimeError("secret cleanup detail")
    client.close_error = original
    with (
        pytest.raises(milvus.VectorStoreError) as captured,
        milvus.milvus_client(MilvusSettings(_env_file=None)),
    ):
        pass
    assert captured.value.__cause__ is original
    assert "secret" not in str(captured.value)
    assert client.close_calls == 1


def test_context_preserves_body_error_when_close_also_fails(client_boundary) -> None:
    milvus, client = client_boundary
    original = RuntimeError("RPC failed")
    client.close_error = RuntimeError("secret cleanup detail")
    with (
        pytest.raises(RuntimeError) as captured,
        milvus.milvus_client(MilvusSettings(_env_file=None)),
    ):
        raise original
    assert captured.value is original
    assert client.close_calls == 1
    assert any("cleanup" in note for note in original.__notes__)
    assert all("secret" not in note for note in original.__notes__)


def test_smoke_empty_collections_succeeds(client_boundary, capsys) -> None:
    _, client = client_boundary
    rpc_calls = []

    def list_collections(**kwargs):
        rpc_calls.append(kwargs)
        return []

    client.list_collections = list_collections
    smoke = importlib.import_module("scripts.smoke_milvus")
    smoke.main()
    assert "Milvus collections: []" in capsys.readouterr().out
    assert rpc_calls == [{"timeout": 10.0}]
    assert client.close_calls == 1


def test_smoke_displays_existing_collections(client_boundary, capsys) -> None:
    _, client = client_boundary
    client.list_collections = lambda **kwargs: ["existing_knowledge"]
    smoke = importlib.import_module("scripts.smoke_milvus")
    smoke.main()
    assert "existing_knowledge" in capsys.readouterr().out
    assert client.close_calls == 1


def test_smoke_rpc_failure_exits_and_closes(client_boundary, capsys) -> None:
    milvus, client = client_boundary
    original = RuntimeError("secret RPC details")

    def list_collections(**kwargs):
        raise original

    client.list_collections = list_collections
    smoke = importlib.import_module("scripts.smoke_milvus")
    with pytest.raises(SystemExit) as captured:
        smoke.main()
    assert captured.value.code
    assert "secret" not in str(captured.value)
    assert isinstance(captured.value.__cause__, milvus.VectorStoreError)
    assert captured.value.__cause__.__cause__ is original
    assert client.close_calls == 1
    assert "secret" not in str(capsys.readouterr())


def test_smoke_connection_failure_exits(monkeypatch, capsys) -> None:
    milvus = importlib.import_module("resolv.db.milvus")
    original = RuntimeError("secret connection details")

    def broken_constructor(**kwargs):
        raise original

    monkeypatch.setattr(milvus, "MilvusClient", broken_constructor)
    smoke = importlib.import_module("scripts.smoke_milvus")
    with pytest.raises(SystemExit) as captured:
        smoke.main()
    assert captured.value.code
    assert "secret" not in str(captured.value)
    assert isinstance(captured.value.__cause__, milvus.VectorStoreError)
    assert captured.value.__cause__.__cause__ is original
    assert "secret" not in str(capsys.readouterr())


def test_smoke_invalid_settings_exits_before_connect(monkeypatch, capsys) -> None:
    milvus = importlib.import_module("resolv.db.milvus")

    def forbidden_constructor(**kwargs):
        pytest.fail("Invalid settings must not create a client")

    monkeypatch.setattr(milvus, "MilvusClient", forbidden_constructor)
    monkeypatch.setenv("MILVUS_URI", " ")
    smoke = importlib.import_module("scripts.smoke_milvus")
    with pytest.raises(SystemExit) as captured:
        smoke.main()
    assert captured.value.code
    assert "MILVUS_URI" in str(captured.value)
    assert capsys.readouterr().out == ""


@pytest.fixture
def sdk_log_boundary(monkeypatch):
    output = StringIO()
    states = []
    for name in ("pymilvus", "pymilvus.milvus_client"):
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [logging.StreamHandler(output)])
        monkeypatch.setattr(logger, "propagate", False)
        states.append((logger, logger.handlers, logger.propagate))
    return output, states


def test_smoke_contains_real_sdk_rpc_diagnostics(
    monkeypatch, sdk_log_boundary, capsys
) -> None:
    from pymilvus import MilvusClient
    from pymilvus.client.grpc_handler import GrpcHandler
    from pymilvus.exceptions import MilvusException

    milvus = importlib.import_module("resolv.db.milvus")
    original = MilvusException(
        message="backend URI=https://user:secret@example.invalid"
    )
    releases = []

    class FailingTransport:
        def ShowCollections(self, *args, **kwargs):
            raise original

    handler = object.__new__(GrpcHandler)
    handler._stub = FailingTransport()
    client = object.__new__(MilvusClient)
    client._handler = handler
    client._config = SimpleNamespace(db_name="")
    client._manager = SimpleNamespace(
        release=lambda *args, **kwargs: releases.append(args)
    )
    monkeypatch.setattr(milvus, "MilvusClient", lambda **kwargs: client)
    smoke = importlib.import_module("scripts.smoke_milvus")
    with pytest.raises(SystemExit) as captured:
        smoke.main()

    output, states = sdk_log_boundary
    assert "secret" not in output.getvalue()
    assert "secret" not in str(capsys.readouterr())
    assert "secret" not in str(captured.value)
    assert captured.value.__cause__.__cause__ is original
    assert len(releases) == 1
    assert client._handler is None
    for logger, handlers, propagate in states:
        assert logger.handlers is handlers
        assert logger.propagate == propagate


@pytest.mark.parametrize("stage", ["connect", "close"])
def test_smoke_contains_lifecycle_logs_and_preserves_other_logging(
    stage, client_boundary, monkeypatch, sdk_log_boundary
) -> None:
    milvus, client = client_boundary
    application_output = StringIO()
    application_logger = logging.getLogger("resolv.smoke_test")
    monkeypatch.setattr(
        application_logger, "handlers", [logging.StreamHandler(application_output)]
    )
    monkeypatch.setattr(application_logger, "propagate", False)

    def fail_with_log(**kwargs):
        logging.getLogger("pymilvus.milvus_client.milvus_client").error(
            "secret SDK detail"
        )
        application_logger.error("application diagnostic")
        raise RuntimeError("secret exception detail")

    if stage == "connect":
        monkeypatch.setattr(milvus, "MilvusClient", fail_with_log)
    else:
        client.list_collections = lambda **kwargs: []
        monkeypatch.setattr(client, "close", fail_with_log)
    smoke = importlib.import_module("scripts.smoke_milvus")
    with pytest.raises(SystemExit) as captured:
        smoke.main()

    output, states = sdk_log_boundary
    assert "secret" not in output.getvalue()
    assert "secret" not in str(captured.value)
    assert "application diagnostic" in application_output.getvalue()
    for logger, handlers, propagate in states:
        assert logger.handlers is handlers
        assert logger.propagate == propagate
