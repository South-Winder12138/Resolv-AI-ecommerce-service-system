import importlib
import logging

import pytest

from resolv.core.config import EmbeddingSettings, MilvusSettings
from resolv.db.milvus import VectorStoreError
from resolv.knowledge import chunk_document, load_document
from resolv.services.embedding import EmbeddingError


@pytest.fixture
def ingest_boundary(monkeypatch, tmp_path):
    ingest = importlib.import_module("scripts.ingest_knowledge")
    db = importlib.import_module("resolv.db.milvus")
    for name in (
        "MILVUS_URI", "MILVUS_COLLECTION", "BGE_M3_MODEL", "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE", "EMBEDDING_USE_FP16", "EMBEDDING_QUERY_MAX_LENGTH",
        "EMBEDDING_PASSAGE_MAX_LENGTH", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ingest, "MilvusSettings", lambda: MilvusSettings(_env_file=None))
    monkeypatch.setattr(ingest, "EmbeddingSettings", lambda: EmbeddingSettings(_env_file=None))
    events = []
    state = {"failure": None, "model_close_error": False, "client_close_error": False}
    path = tmp_path / "refund.md"
    path.write_text("# 售后政策\n\n## 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。", encoding="utf-8")

    class FakeEmbedding:
        dimension = 3

        def __init__(self, settings):
            events.append(("model", settings.model_name, settings.device, settings.use_fp16))
            if state["failure"] == "model":
                raise EmbeddingError("Embedding model loading failed.")

        def embed_documents(self, texts):
            events.append(("embed", texts))
            if state["failure"] == "embed":
                raise EmbeddingError("Dense embedding inference failed.")
            return [[1.0, 0.0, 0.0] for _ in texts]

        def close(self):
            events.append(("model_close",))
            if state["model_close_error"]:
                raise EmbeddingError("Embedding model cleanup failed.")

    class FakeClient:
        def close(self):
            events.append(("client_close",))
            if state["client_close_error"]:
                raise RuntimeError("private cleanup details")

    client = FakeClient()

    def create_client(settings):
        events.append(("client", settings.milvus_uri))
        if state["failure"] == "client":
            logging.getLogger("pymilvus").error("private-token-one")
            logging.getLogger("pymilvus.milvus_client").error("private-token-two")
            raise VectorStoreError("Milvus connection failed.")
        return client

    class FakeStore:
        def __init__(self, passed_client, collection):
            assert passed_client is client
            events.append(("store", collection))

        def ensure_collection(self, dimension):
            events.append(("ensure", dimension))
            if state["failure"] == "ensure":
                raise VectorStoreError("Milvus collection preparation failed.")

        def upsert(self, chunks, vectors):
            events.append(("upsert", chunks, vectors))
            if state["failure"] == "upsert":
                raise VectorStoreError("Milvus upsert failed.")
            return len(chunks) - (state["failure"] == "partial")

    monkeypatch.setattr(ingest, "EmbeddingService", FakeEmbedding)
    monkeypatch.setattr(db, "create_milvus_client", create_client)
    monkeypatch.setattr(ingest, "MilvusVectorStore", FakeStore)
    return ingest, path, events, state


def arguments(path, *extra):
    return [path.name, "--source-root", str(path.parent), *extra]


def test_ingest_real_chunk_to_fake_vector_store_with_observable_output(ingest_boundary, capsys):
    ingest, path, events, _ = ingest_boundary
    expected_chunks = chunk_document(load_document(path, source_root=path.parent))
    ingest.main(arguments(path))
    assert [event[0] for event in events] == [
        "model", "client", "store", "ensure", "embed", "upsert", "client_close", "model_close",
    ]
    assert events[0] == ("model", "BAAI/bge-m3", "cpu", False)
    assert events[2] == ("store", "resolv_knowledge_v1")
    assert events[3] == ("ensure", 3)
    assert events[4] == ("embed", [chunk.text for chunk in expected_chunks])
    assert events[5] == ("upsert", expected_chunks, [[1.0, 0.0, 0.0]])
    output = capsys.readouterr().out
    for value in (
        "Document:", expected_chunks[0].document_id, "Source: refund.md", "Chunk count: 1",
        "Embedding model: BAAI/bge-m3", "Embedding dimension: 3", "Normalize: True",
        "Milvus collection: resolv_knowledge_v1", "Upsert count: 1", expected_chunks[0].text,
        "Ingestion completed; resources closed.",
    ):
        assert value in output


def test_ingest_passes_chunk_parameters_and_preserves_batch_order(ingest_boundary):
    ingest, path, events, _ = ingest_boundary
    path.write_text("# 政策\n\n" + "中文正文。" * 50, encoding="utf-8")
    expected = chunk_document(load_document(path, source_root=path.parent), max_chars=80, overlap=10)
    ingest.main(arguments(path, "--max-chars", "80", "--overlap", "10"))
    assert len(expected) > 1
    write = next(event for event in events if event[0] == "upsert")
    assert write[1] == expected
    assert write[2] == [[1.0, 0.0, 0.0]] * len(expected)


def test_ingest_chunk_preview_is_truncated(ingest_boundary, capsys):
    ingest, path, _, _ = ingest_boundary
    path.write_text("x" * 250 + "secret-tail", encoding="utf-8")
    ingest.main(arguments(path))
    output = capsys.readouterr().out
    assert "x" * 200 in output
    assert "secret-tail" not in output
    assert "…" in output


def test_ingest_empty_document_needs_no_external_resources(ingest_boundary, capsys):
    ingest, path, events, _ = ingest_boundary
    path.write_text(" \n\t", encoding="utf-8")
    ingest.main(arguments(path))
    assert events == []
    output = capsys.readouterr().out
    assert "Chunk count: 0" in output
    assert "Upsert count: 0" in output
    assert "not loaded" in output


def test_ingest_default_source_root_uses_existing_sample(ingest_boundary, capsys):
    ingest, _, events, _ = ingest_boundary
    ingest.main(["knowledge/refund.md"])
    assert "Source: knowledge/refund.md" in capsys.readouterr().out
    assert next(event for event in events if event[0] == "upsert")[1][0].source == "knowledge/refund.md"


@pytest.mark.parametrize("bad_case", ["missing", "unsupported", "decode", "chunk_size", "overlap", "source_root"])
def test_ingest_bad_input_exits_before_external_resources(ingest_boundary, bad_case):
    ingest, path, events, _ = ingest_boundary
    args = arguments(path)
    if bad_case == "missing":
        path.unlink()
    elif bad_case == "unsupported":
        args[0] = "refund.pdf"
    elif bad_case == "decode":
        path.write_bytes(b"\xff\xfe")
    elif bad_case == "chunk_size":
        args.extend(["--max-chars", "0"])
    elif bad_case == "overlap":
        args.extend(["--overlap", "800"])
    elif bad_case == "source_root":
        args[2] = str(path)
    with pytest.raises(SystemExit) as error:
        ingest.main(args)
    assert error.value.code != 0
    assert events == []


@pytest.mark.parametrize("name,value", [("EMBEDDING_BATCH_SIZE", "0"), ("MILVUS_COLLECTION", "bad name"), ("MILVUS_URI", "secret-token.db")])
def test_ingest_bad_config_is_safe_and_never_loads_resources(ingest_boundary, monkeypatch, name, value):
    ingest, path, events, _ = ingest_boundary
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as error:
        ingest.main(arguments(path))
    assert error.value.code != 0
    assert value not in str(error.value)
    assert events == []


@pytest.mark.parametrize("stage", ["model", "client", "ensure", "embed", "upsert", "partial"])
def test_ingest_failures_exit_and_release_owned_resources(ingest_boundary, capsys, stage):
    ingest, path, events, state = ingest_boundary
    state["failure"] = stage
    with pytest.raises(SystemExit) as error:
        ingest.main(arguments(path))
    assert error.value.code != 0
    names = [event[0] for event in events]
    assert names.count("model_close") == (stage != "model")
    assert names.count("client_close") == (stage not in {"model", "client"})
    assert "completed" not in capsys.readouterr().out.lower()
    if stage in {"model", "client", "ensure", "embed"}:
        assert "upsert" not in names
    if stage == "partial":
        assert isinstance(error.value.__cause__, VectorStoreError)


@pytest.mark.parametrize("close_target", ["model_close_error", "client_close_error"])
def test_ingest_cleanup_failure_is_not_success(ingest_boundary, capsys, close_target):
    ingest, path, events, state = ingest_boundary
    state[close_target] = True
    with pytest.raises(SystemExit) as error:
        ingest.main(arguments(path))
    assert error.value.code != 0
    assert [event[0] for event in events][-2:] == ["client_close", "model_close"]
    assert "completed" not in capsys.readouterr().out.lower()


def test_ingest_cleanup_failures_preserve_primary_error(ingest_boundary):
    ingest, path, events, state = ingest_boundary
    state.update(failure="embed", model_close_error=True, client_close_error=True)
    with pytest.raises(SystemExit) as error:
        ingest.main(arguments(path))
    assert isinstance(error.value.__cause__, EmbeddingError)
    assert str(error.value.__cause__) == "Dense embedding inference failed."
    assert len(error.value.__cause__.__notes__) == 2
    assert [event[0] for event in events][-2:] == ["client_close", "model_close"]


def test_ingest_uses_independent_env_settings(ingest_boundary, monkeypatch, capsys):
    ingest, path, events, _ = ingest_boundary
    monkeypatch.setenv("MILVUS_COLLECTION", "resolv_knowledge_v2")
    monkeypatch.setenv("BGE_M3_MODEL", "local-model")
    ingest.main(arguments(path))
    assert events[0][1] == "local-model"
    assert events[2][1] == "resolv_knowledge_v2"
    assert "resolv_knowledge_v2" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [None, "client"])
def test_ingest_restores_sdk_logging_and_suppresses_raw_credentials(ingest_boundary, capsys, failure):
    ingest, path, _, state = ingest_boundary
    loggers = [logging.getLogger(name) for name in ("pymilvus", "pymilvus.milvus_client", "resolv")]
    original = [(logger.handlers, logger.propagate) for logger in loggers]
    state["failure"] = failure
    if failure:
        with pytest.raises(SystemExit):
            ingest.main(arguments(path))
    else:
        ingest.main(arguments(path))
    captured = capsys.readouterr()
    assert "private-token" not in captured.out + captured.err
    for logger, (handlers, propagate) in zip(loggers, original, strict=True):
        assert logger.handlers is handlers
        assert logger.propagate is propagate


def test_ingest_path_is_required(ingest_boundary):
    ingest, _, events, _ = ingest_boundary
    with pytest.raises(SystemExit) as error:
        ingest.main([])
    assert error.value.code == 2
    assert events == []
