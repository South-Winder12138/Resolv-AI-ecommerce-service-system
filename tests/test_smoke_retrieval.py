import importlib
import logging
import subprocess
import sys
from io import StringIO

import pytest

from resolv.core.config import EmbeddingSettings, MilvusSettings
from resolv.schemas.knowledge import SearchHit
from resolv.services.embedding import EmbeddingError


@pytest.fixture
def smoke_boundary(monkeypatch):
    smoke = importlib.import_module("scripts.smoke_retrieval")
    db = importlib.import_module("resolv.db.milvus")
    for name in (
        "MILVUS_URI", "MILVUS_COLLECTION", "BGE_M3_MODEL", "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE", "EMBEDDING_USE_FP16", "EMBEDDING_QUERY_MAX_LENGTH",
        "EMBEDDING_PASSAGE_MAX_LENGTH", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(smoke, "EmbeddingSettings", lambda: EmbeddingSettings(_env_file=None))
    monkeypatch.setattr(smoke, "MilvusSettings", lambda: MilvusSettings(_env_file=None))
    events = []
    state = {
        "failure": None, "log_at": None, "model_close_error": False,
        "client_close_error": False, "primary": None,
        "hits": [
            SearchHit(
                chunk_id="c0", document_id="doc0", source="knowledge/refund.md",
                text="售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
                chunk_index=0, score=0.731234567,
            ),
            SearchHit(
                chunk_id="c1", document_id="doc1", source="knowledge/shipment.md",
                text="物流政策", chunk_index=1, score=-0.25,
            ),
        ],
    }

    def fail(error_type, message):
        error = error_type(message)
        state["primary"] = error
        raise error from RuntimeError("private backend details")

    def log_sdk(stage):
        if state["log_at"] == stage:
            for name in ("pymilvus", "pymilvus.milvus_client"):
                logging.getLogger(name).error("uri=https://user:sensitive-marker@example.invalid")
            logging.getLogger("resolv.smoke_retrieval_test").error("application diagnostic")

    class FakeEmbedding:
        dimension = 3

        def __init__(self, settings):
            events.append(("model", settings.model_name, settings.device, settings.use_fp16))
            if state["failure"] == "model":
                fail(EmbeddingError, "Embedding model loading failed.")

        def embed_query(self, query):
            events.append(("query", query))
            if state["failure"] == "query":
                fail(EmbeddingError, "Query embedding failed.")
            if state["failure"] == "token":
                fail(ValueError, "Query exceeds the token budget.")
            if state["failure"] == "interrupt":
                raise KeyboardInterrupt
            return [1.0, 0.0, 0.0]

        def embed_documents(self, texts):
            pytest.fail("Retrieval smoke must not embed documents")

        def close(self):
            events.append(("model_close",))
            if state["model_close_error"]:
                raise EmbeddingError("Embedding model cleanup failed.")

    class FakeClient:
        def close(self):
            events.append(("client_close",))
            log_sdk("close")
            if state["client_close_error"]:
                raise RuntimeError("private client cleanup details")

    client = FakeClient()

    def create_client(settings):
        events.append(("client", settings.milvus_uri))
        log_sdk("client")
        if state["failure"] == "client":
            fail(db.VectorStoreError, "Milvus connection failed.")
        return client

    class FakeStore:
        def __init__(self, passed_client, collection):
            assert passed_client is client
            events.append(("store", collection))

        def search(self, vector, top_k):
            events.append(("search", vector, top_k))
            log_sdk("search")
            if state["failure"] in {"search", "missing", "incompatible"}:
                fail(db.VectorStoreError, "Milvus knowledge search or collection validation failed.")
            return state["hits"][:top_k]

        def ensure_collection(self, dimension):
            pytest.fail("Retrieval smoke must not create collections")

        def upsert(self, chunks, vectors):
            pytest.fail("Retrieval smoke must not write vectors")

    monkeypatch.setattr(smoke, "EmbeddingService", FakeEmbedding)
    monkeypatch.setattr(db, "create_milvus_client", create_client)
    monkeypatch.setattr(smoke, "MilvusVectorStore", FakeStore)
    return smoke, events, state


def test_default_smoke_composes_real_retriever_and_prints_full_hits(smoke_boundary, capsys):
    smoke, events, _ = smoke_boundary
    smoke.main([])
    assert events == [
        ("model", "BAAI/bge-m3", "cpu", False), ("client", "http://127.0.0.1:19530"),
        ("store", "resolv_knowledge_v1"), ("query", "软件已经激活了，还能退款吗？"),
        ("search", [1.0, 0.0, 0.0], 5), ("client_close",), ("model_close",),
    ]
    output = capsys.readouterr().out
    for expected in (
        "Query: 软件已经激活了，还能退款吗？", "Top 5", "Hits: 2", "[1]", "[2]",
        "Embedding model: BAAI/bge-m3", "Device: cpu FP16: False", "Normalize: True",
        "batch_size: 8", "query_max_length: 512", "Embedding dimension: 3",
        "Milvus collection: resolv_knowledge_v1", "score=0.731234567", "score=-0.25",
        "chunk_id=c0", "document_id=doc0", "source=knowledge/refund.md", "chunk_index=0",
        "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
        "FlagEmbedding:", "torch:", "transformers:", "numpy:", "pymilvus:",
        "Retrieval completed; resources closed.",
    ):
        assert expected in output


def test_custom_query_and_top_k_preserve_input_and_limit(smoke_boundary, capsys):
    smoke, events, _ = smoke_boundary
    smoke.main([" \t激活后还能退款吗？\n", "--top-k", "1"])
    assert events[3:5] == [("query", " \t激活后还能退款吗？\n"), ("search", [1.0, 0.0, 0.0], 1)]
    output = capsys.readouterr().out
    assert "Top 1" in output and "Hits: 1" in output
    assert "[2]" not in output and "物流政策" not in output


def test_zero_hits_is_successful_data_not_a_customer_answer(smoke_boundary, capsys):
    smoke, events, state = smoke_boundary
    state["hits"] = []
    smoke.main([])
    output = capsys.readouterr().out
    assert "Hits: 0" in output and "[1]" not in output
    assert "Retrieval completed; resources closed." in output
    assert events[-2:] == [("client_close",), ("model_close",)]


def test_main_without_argv_uses_process_arguments(smoke_boundary, monkeypatch):
    smoke, events, _ = smoke_boundary
    monkeypatch.setattr(sys, "argv", ["smoke_retrieval.py", "退款政策", "--top-k", "1"])
    smoke.main()
    assert events[3:5] == [("query", "退款政策"), ("search", [1.0, 0.0, 0.0], 1)]


def test_help_opens_no_external_resources(smoke_boundary, capsys):
    smoke, events, _ = smoke_boundary
    with pytest.raises(SystemExit) as error:
        smoke.main(["--help"])
    assert error.value.code == 0
    assert "--top-k" in capsys.readouterr().out
    assert events == []


@pytest.mark.parametrize("query", ["", " \t\n", "　"])
def test_blank_query_rejected_before_loading_or_connecting(smoke_boundary, query):
    smoke, events, _ = smoke_boundary
    with pytest.raises(SystemExit) as error:
        smoke.main([query])
    assert error.value.code == 2
    assert events == []


@pytest.mark.parametrize("top_k", ["0", "-1", "bad", "1.5", "true"])
def test_invalid_top_k_rejected_before_loading_or_connecting(smoke_boundary, top_k):
    smoke, events, _ = smoke_boundary
    with pytest.raises(SystemExit) as error:
        smoke.main(["--top-k", top_k])
    assert error.value.code == 2
    assert events == []


@pytest.mark.parametrize("name,value", [("MILVUS_URI", "sensitive-marker.db"), ("MILVUS_COLLECTION", "bad name"), ("EMBEDDING_BATCH_SIZE", "0")])
def test_invalid_settings_are_safe_and_open_no_resources(smoke_boundary, monkeypatch, name, value):
    smoke, events, _ = smoke_boundary
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as error:
        smoke.main([])
    assert error.value.code != 0
    assert value not in str(error.value)
    assert events == []


@pytest.mark.parametrize("stage", ["model", "client", "query", "token", "search", "missing", "incompatible"])
def test_failure_exits_and_releases_every_owned_resource(smoke_boundary, capsys, stage):
    smoke, events, state = smoke_boundary
    state["failure"] = stage
    with pytest.raises(SystemExit) as error:
        smoke.main([])
    assert error.value.code != 0
    assert error.value.__cause__ is state["primary"]
    assert isinstance(error.value.__cause__.__cause__, RuntimeError)
    names = [event[0] for event in events]
    assert names.count("model_close") == (stage != "model")
    assert names.count("client_close") == (stage not in {"model", "client"})
    if stage in {"model", "client", "query", "token"}:
        assert "search" not in names
    output = capsys.readouterr().out
    assert "completed" not in output.lower() and "Hits:" not in output


@pytest.mark.parametrize("target", ["model_close_error", "client_close_error"])
def test_cleanup_failure_prevents_success_footer(smoke_boundary, capsys, target):
    smoke, events, state = smoke_boundary
    state[target] = True
    with pytest.raises(SystemExit) as error:
        smoke.main([])
    assert error.value.code != 0
    assert events[-2:] == [("client_close",), ("model_close",)]
    assert "completed" not in capsys.readouterr().out.lower()


def test_primary_failure_survives_both_cleanup_failures(smoke_boundary):
    smoke, events, state = smoke_boundary
    state.update(failure="query", model_close_error=True, client_close_error=True)
    with pytest.raises(SystemExit) as error:
        smoke.main([])
    assert error.value.__cause__ is state["primary"]
    assert isinstance(error.value.__cause__, EmbeddingError)
    assert len(error.value.__cause__.__notes__) == 2
    assert events[-2:] == [("client_close",), ("model_close",)]


def test_interrupt_still_closes_resources(smoke_boundary, capsys):
    smoke, events, state = smoke_boundary
    state["failure"] = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        smoke.main([])
    assert events[-2:] == [("client_close",), ("model_close",)]
    assert "completed" not in capsys.readouterr().out.lower()


@pytest.mark.parametrize("stage", [None, "client", "search", "close"])
def test_sdk_console_is_scoped_and_other_logging_preserved(smoke_boundary, monkeypatch, capsys, stage):
    smoke, _, state = smoke_boundary
    sdk_output, application_output = StringIO(), StringIO()
    originals = []
    for name in ("pymilvus", "pymilvus.milvus_client", "resolv.smoke_retrieval_test"):
        logger = logging.getLogger(name)
        target = application_output if name.startswith("resolv") else sdk_output
        monkeypatch.setattr(logger, "handlers", [logging.StreamHandler(target)])
        monkeypatch.setattr(logger, "propagate", False)
        originals.append((logger, logger.handlers, logger.propagate))
    state["log_at"] = stage
    if stage in {"client", "search"}:
        state["failure"] = stage
    if stage == "close":
        state["client_close_error"] = True
    if stage:
        with pytest.raises(SystemExit) as error:
            smoke.main([])
        assert "sensitive-marker" not in str(error.value)
        assert "application diagnostic" in application_output.getvalue()
    else:
        smoke.main([])
    assert "sensitive-marker" not in sdk_output.getvalue()
    captured = capsys.readouterr()
    assert "sensitive-marker" not in captured.out + captured.err
    for logger, handlers, propagate in originals:
        assert logger.handlers is handlers
        assert logger.propagate is propagate


def test_import_loads_no_model_and_opens_no_client():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
import resolv.core.config as config
import resolv.db.milvus as db
import resolv.services.embedding as embedding

def unexpected(*args, **kwargs):
    raise AssertionError('import must not construct settings or external resources')

config.EmbeddingSettings = unexpected
config.MilvusSettings = unexpected
embedding.EmbeddingService = unexpected
embedding._load_model = unexpected
db.create_milvus_client = unexpected
import scripts.smoke_retrieval
assert 'FlagEmbedding' not in sys.modules
assert 'torch' not in sys.modules
"""],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
