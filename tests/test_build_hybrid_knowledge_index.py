import pytest
from scripts import build_hybrid_knowledge_index as module

from tests.test_knowledge_hybrid_repository import hybrid_factory

__all__ = ["hybrid_factory"]


def test_help_and_missing_explicit_selection_create_no_resources(monkeypatch, capsys):
    monkeypatch.setattr(
        module, "build", lambda *a, **kw: pytest.fail("created resources")
    )
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "--document-id" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        module.main(["--version", "v3"])
    assert caught.value.code != 0


def test_build_failure_emits_no_success_json_or_credentials(monkeypatch, capsys):
    def fail(*a, **kw):
        raise RuntimeError("private URI password")

    monkeypatch.setattr(module, "build", fail)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--version",
                "v3",
                "--collection",
                "mhhybrid_gate_fixture",
                "--document-id",
                "a" * 64,
                "--file-config",
                "synthetic.toml",
                "--embedding-revision",
                "5617a9f61b028005a4858fdac845db406aefb181",
            ]
        )
    captured = capsys.readouterr()
    assert captured.out == "" and "private" not in captured.err


def test_wrong_dense_revision_refused_before_sql_or_model(monkeypatch):
    from types import SimpleNamespace

    from resolv.schemas.hybrid_knowledge import HybridRetrievalError

    monkeypatch.setattr(module, "load_knowledge_file_configs", lambda *a: {})
    monkeypatch.setattr(
        module, "knowledge_engine", lambda *a: pytest.fail("opened SQL")
    )
    args = SimpleNamespace(
        collection="mhhybrid_gate_fixture",
        embedding_revision="a" * 40,
        passage_max_length=512,
        file_config="unused",
    )
    with pytest.raises(HybridRetrievalError, match="hybrid_dense_revision_mismatch"):
        module.build(args)


def test_owned_model_close_failure_is_not_command_success(
    hybrid_factory, monkeypatch, capsys
):
    monkeypatch.setenv(
        "KNOWLEDGE_DATABASE_URL",
        "mysql+pymysql://offline:offline@127.0.0.1:3307/unused?charset=utf8mb4",
    )
    from contextlib import contextmanager

    from tests.test_hybrid_indexing import Lock
    from tests.test_hybrid_manifest import CONFIG

    factory, s, _ = hybrid_factory
    close_calls = []

    @contextmanager
    def engine_owner(*a):
        yield factory.kw["bind"]

    @contextmanager
    def lock_owner(*a):
        yield Lock()

    @contextmanager
    def client_owner(*a):
        yield object()

    class Embed:
        dimension = 1024

        def __init__(self, p, **kw):
            self.profile = p

        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 1023 for _ in texts]

        def close(self):
            close_calls.append(True)
            raise OSError("private model cleanup credentials")

    class Store:
        def __init__(self, client, p):
            self.profile = p
            self.collection_name = p.collection_name

        def create_or_validate(self):
            pass

        def validate_existing(self):
            pass

        def upsert(self, entries, vectors):
            return len(entries)

    monkeypatch.setattr(module, "knowledge_engine", engine_owner)
    monkeypatch.setattr(module, "hybrid_index_lock", lock_owner)
    monkeypatch.setattr(module, "milvus_client", client_owner)
    monkeypatch.setattr(
        module, "load_knowledge_file_configs", lambda *a: {s.document.source: CONFIG}
    )
    monkeypatch.setattr(module, "HybridEmbeddingAdapter", Embed)
    monkeypatch.setattr(module, "HybridMilvusStore", Store)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--version",
                "v3",
                "--collection",
                "mhhybrid_gate_fixture",
                "--document-id",
                s.document.document_id,
                "--file-config",
                "synthetic.toml",
                "--embedding-revision",
                "5617a9f61b028005a4858fdac845db406aefb181",
            ]
        )
    captured = capsys.readouterr()
    assert close_calls == [True]
    assert captured.out == "" and "private" not in captured.err
