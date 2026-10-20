"""CLI integration uses real local SQL/pipeline and replaces external resources only."""
import importlib
import json
import logging
from contextlib import contextmanager

import pytest
from sqlalchemy import select

from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.knowledge import KnowledgeStoreError
from resolv.db.knowledge_base import KnowledgeBase
from resolv.db.milvus import VectorStoreError
from resolv.models.knowledge import KnowledgeChunkRecord
from resolv.schemas.knowledge import KnowledgeIndexProfile
from resolv.services.embedding import EmbeddingError
from tests.test_milvus import FakeKnowledgeClient

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
COMMANDS = ["index_pending_knowledge", "smoke_knowledge_retrieval", "smoke_knowledge_pipeline"]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'runtime.db'}")
    KnowledgeBase.metadata.create_all(engine)
    path, config = tmp_path / "refund.md", tmp_path / "files.toml"
    path.write_text("# 售后政策\n## 软件商品退款\n软件商品一旦激活，不支持无理由退款。", encoding="utf-8")
    config.write_text('[files."refund.md"]\ncontent_type="refund_policy"\nkey_clause_keywords=["不支持"]', encoding="utf-8")
    state = {"fail": None, "events": []}
    profile = KnowledgeIndexProfile(model_revision=REVISION)

    class Client(FakeKnowledgeClient):
        def __init__(self):
            super().__init__(exists=False)
            self.index = None

        def search(self, **kwargs):
            self.record("search", kwargs)
            return [[{
                "id": row["chunk_id"], "distance": 0.9,
                "entity": {key: row[key] for key in ("chunk_id", "document_id", "index_hash")},
            } for row in self.rows.values()][:kwargs["limit"]]]

        def close(self):
            state["events"].append("client_close")
            if state["fail"] == "client_close":
                raise RuntimeError("private cleanup")

    client = Client()

    @contextmanager
    def db(settings):
        state["events"].append("engine")
        try:
            yield engine
        finally:
            state["events"].append("engine_close")

    class Lock:
        def assert_owned(self):
            if state["fail"] == "lock_lost":
                raise KnowledgeStoreError("Index lock lost")

    @contextmanager
    def lock(_engine):
        state["events"].append("lock")
        if state["fail"] == "lock":
            raise KnowledgeStoreError("Index lock unavailable")
        try:
            yield Lock()
        finally:
            state["events"].append("lock_close")

    class Embedder:
        dimension = 1024

        def __init__(self, requested):
            self.profile = requested

        def embed_documents(self, texts):
            state["events"].append("embed")
            if state["fail"] == "embed":
                raise EmbeddingError("Embedding failed")
            assert engine.pool.checkedout() == 0
            return [[1.0]+[0.0]*1023 for _ in texts]

        def embed_query(self, query):
            state["events"].append("query")
            assert engine.pool.checkedout() == 0
            return [1.0]+[0.0]*1023

    @contextmanager
    def embedding(requested, **kwargs):
        assert "lock" in state["events"] or state.get("query_only")
        state["events"].append("model")
        if state["fail"] == "model":
            raise EmbeddingError("Model loading failed")
        try:
            yield Embedder(requested)
        finally:
            state["events"].append("model_close")

    def client_factory(settings):
        state["events"].append("client")
        if state["fail"] == "client":
            logging.getLogger("pymilvus").error("private-RPC-token")
            raise VectorStoreError("Connection failed")
        return client

    monkeypatch.setattr(importlib.import_module("resolv.db.milvus"), "create_milvus_client", client_factory)
    modules = {}
    for name in COMMANDS:
        module = importlib.import_module("scripts." + name)
        modules[name] = module
        monkeypatch.setattr(module, "knowledge_engine", db)
        monkeypatch.setattr(module, "pinned_knowledge_embedding", embedding)
        monkeypatch.setattr(module, "KnowledgeDatabaseSettings", lambda: KnowledgeDatabaseSettings(
            _env_file=None, knowledge_database_url="mysql+pymysql://test:private@localhost/db?charset=utf8mb4"
        ))
        monkeypatch.setattr(module, "MilvusSettings", lambda: MilvusSettings(_env_file=None))
        if hasattr(module, "knowledge_index_lock"):
            monkeypatch.setattr(module, "knowledge_index_lock", lock)
    yield modules, path, config, profile, engine, state, client
    engine.dispose()


def _pipeline(runtime, **options):
    modules, path, config, _, _, _, _ = runtime
    modules["smoke_knowledge_pipeline"].main([
        str(path), "--config", str(config), "--source-root", str(path.parent),
        "--model-revision", REVISION, *options.get("extra", []),
    ])


def test_complete_cli_stage_index_retrieve_and_repeat(runtime, capsys):
    _pipeline(runtime)
    first = json.loads(capsys.readouterr().out)
    assert first["stage"]["outcome"] == "created"
    assert first["index"]["indexed_count"] == 1
    assert first["hits"][0]["answer"] == "软件商品一旦激活，不支持无理由退款。"
    assert first["hits"][0]["section_path"] == "售后政策 > 软件商品退款"
    assert first["hits"][0]["is_key_clause"] == 1
    assert first["index_profile"]["model_revision"] == REVISION
    _pipeline(runtime)
    second = json.loads(capsys.readouterr().out)
    assert second["stage"]["outcome"] == "unchanged"
    assert second["index"]["indexed_count"] == 0
    assert len(runtime[-1].rows) == 1
    assert runtime[-2]["events"].count("model") == 2
    assert runtime[-2]["events"][-1] == "engine_close"


def test_index_command_no_pending_and_query_read_only(runtime, capsys):
    _pipeline(runtime)
    capsys.readouterr()
    module = runtime[0]["index_pending_knowledge"]
    module.main(["--model-revision", REVISION])
    result = json.loads(capsys.readouterr().out)
    assert result["index"]["indexed_count"] == result["index"]["attempted_count"] == 0
    runtime[-2]["query_only"] = True
    runtime[0]["smoke_knowledge_retrieval"].main(["软件已经激活还能退款吗？", "--model-revision", REVISION, "--top-k", "5"])
    result = json.loads(capsys.readouterr().out)
    assert len(result["hits"]) == 1
    assert result["hits"][0]["score"] == 0.9


@pytest.mark.parametrize("failure", ["lock", "model", "embed", "client", "client_close"])
def test_resource_failures_have_no_success_json_or_private_output(runtime, capsys, failure):
    runtime[-2]["fail"] = failure
    with pytest.raises(SystemExit) as caught:
        _pipeline(runtime)
    output = capsys.readouterr()
    assert output.out == "" and "private" not in output.err
    assert "private" not in str(caught.value)
    assert "engine_close" in runtime[-2]["events"]
    if failure in {"model", "embed", "client", "client_close"}:
        assert "lock_close" in runtime[-2]["events"]


@pytest.mark.parametrize("name", COMMANDS)
def test_invalid_profile_never_opens_resources(runtime, capsys, name):
    module = runtime[0][name]
    args = ["--model-revision", "main"]
    if name == "smoke_knowledge_pipeline":
        args = [str(runtime[1]), "--config", str(runtime[2]), "--source-root", str(runtime[1].parent), *args]
    with pytest.raises(SystemExit):
        module.main(args)
    assert runtime[-2]["events"] == []
    assert capsys.readouterr().out == ""


def test_pipeline_missing_file_config_never_opens_resources(runtime, capsys):
    runtime[2].write_text("[files]", encoding="utf-8")
    with pytest.raises(SystemExit):
        _pipeline(runtime)
    assert runtime[-2]["events"] == []
    assert capsys.readouterr().out == ""


def test_pipeline_bad_chunk_budget_before_resources(runtime, capsys):
    with pytest.raises(SystemExit):
        _pipeline(runtime, extra=["--max-chars", "0"])
    assert runtime[-2]["events"] == []
    assert capsys.readouterr().out == ""


def test_query_invalid_k_before_resources(runtime, capsys):
    with pytest.raises(SystemExit):
        runtime[0]["smoke_knowledge_retrieval"].main(["退款", "--model-revision", REVISION, "--top-k", "0"])
    assert runtime[-2]["events"] == []
    assert capsys.readouterr().out == ""


def test_index_bad_batch_before_resources(runtime, capsys):
    with pytest.raises(SystemExit):
        runtime[0]["index_pending_knowledge"].main(["--model-revision", REVISION, "--batch-size", "0"])
    assert runtime[-2]["events"] == []
    assert capsys.readouterr().out == ""


def test_body_error_not_lost_to_client_cleanup(runtime, capsys):
    runtime[-1].error_at = "upsert"
    runtime[-2]["fail"] = "client_close"
    with pytest.raises(SystemExit):
        _pipeline(runtime)
    assert capsys.readouterr().out == ""
    with create_session_factory(runtime[4])() as session:
        chunk = session.scalar(select(KnowledgeChunkRecord))
        assert chunk.index_status == "pending" and chunk.last_error


@pytest.mark.parametrize("name", COMMANDS)
def test_import_never_creates_resource_or_loads_model(tmp_path, name):
    import subprocess
    import sys

    script = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / (name+".py")
    code = """
import runpy, sys
runpy.run_path(sys.argv[1], run_name="knowledge_import_guard")
assert not {"torch","FlagEmbedding","langchain_openai"} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", code, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
