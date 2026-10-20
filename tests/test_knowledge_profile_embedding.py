"""Immutable model binding using local cache and the real embedding adapter."""
import importlib
from pathlib import Path

import pytest

from resolv.schemas.knowledge import KnowledgeIndexProfile
from resolv.services import embedding
from resolv.services.embedding import EmbeddingError
from tests.test_embedding import FakeModel

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "sentencepiece.bpe.model", "pytorch_model.bin", "colbert_linear.pt", "sparse_linear.pt",
]


def _module():
    return importlib.import_module("resolv.services.knowledge_embedding")


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    directory = tmp_path / "models--BAAI--bge-m3" / "snapshots" / REVISION
    directory.mkdir(parents=True)
    for name in FILES:
        (directory / name).write_text("test-only cached file", encoding="utf-8")
    model, loads, requests = FakeModel(), [], []
    def resolve(**kwargs):
        requests.append(kwargs)
        return str(directory)
    def load(settings):
        loads.append(settings)
        return model
    monkeypatch.setattr(embedding, "_load_model", load)
    return directory, model, loads, requests, resolve


def _service(boundary, monkeypatch, **options):
    monkeypatch.setattr(_module(), "_snapshot_download", boundary[-1])
    return _module().KnowledgeEmbeddingService(
        KnowledgeIndexProfile(model_revision=REVISION, passage_max_length=12), **options
    )


def test_binding_loads_exact_snapshot_and_profile_once(boundary, monkeypatch):
    directory, model, loads, requests, _ = boundary
    service = _service(boundary, monkeypatch, query_max_length=8)
    assert service.profile.model_revision == REVISION and service.dimension == 1024
    assert requests == [{
        "repo_id": "BAAI/bge-m3", "revision": REVISION,
        "cache_dir": None, "local_files_only": True,
    }]
    assert len(loads) == 1
    assert loads[0].model_name == str(directory.resolve())
    assert (loads[0].device, loads[0].use_fp16, loads[0].passage_max_length) == ("cpu", False, 12)
    assert len(service.embed_documents(["正文"])[0]) == len(service.embed_query("退款")) == 1024
    assert [kind for kind, _, _ in model.calls] == ["documents", "query"]
    assert len(loads) == 1
    service.close()
    assert model.close_count == 1


def test_environment_cannot_substitute_model_precision_or_limits(boundary, monkeypatch):
    for key, value in {
        "BGE_M3_MODEL": "private/other-model", "EMBEDDING_DEVICE": "mps",
        "EMBEDDING_USE_FP16": "true", "EMBEDDING_PASSAGE_MAX_LENGTH": "8192",
        "EMBEDDING_QUERY_MAX_LENGTH": "8192",
    }.items():
        monkeypatch.setenv(key, value)
    service = _service(boundary, monkeypatch, query_max_length=8)
    settings = boundary[2][0]
    assert settings.model_name == str(boundary[0].resolve())
    assert (settings.device, settings.use_fp16, settings.query_max_length, settings.passage_max_length) == ("cpu", False, 8, 12)
    service.close()


@pytest.mark.parametrize("name", FILES)
def test_incomplete_cache_fails_before_loading_model(boundary, monkeypatch, name):
    (boundary[0] / name).unlink()
    with pytest.raises(EmbeddingError):
        _service(boundary, monkeypatch)
    assert boundary[2] == []


@pytest.mark.parametrize("wrong", ["wrong-revision", "ordinary-directory"])
def test_wrong_snapshot_identity_fails_before_model_load(boundary, monkeypatch, tmp_path, wrong):
    target = tmp_path / wrong
    target.mkdir()
    monkeypatch.setattr(_module(), "_snapshot_download", lambda **kwargs: str(target))
    with pytest.raises(EmbeddingError):
        _module().KnowledgeEmbeddingService(KnowledgeIndexProfile(model_revision=REVISION))
    assert boundary[2] == []


def test_missing_cache_error_safe_and_preserves_cause(boundary, monkeypatch):
    failure = RuntimeError("private cache credentials")
    def resolve(**kwargs):
        raise failure
    monkeypatch.setattr(_module(), "_snapshot_download", resolve)
    with pytest.raises(EmbeddingError) as caught:
        _module().KnowledgeEmbeddingService(KnowledgeIndexProfile(model_revision=REVISION))
    assert caught.value.__cause__ is failure
    assert "private" not in str(caught.value)


def test_query_and_passage_limits_remain_distinct(boundary, monkeypatch):
    service = _service(boundary, monkeypatch, query_max_length=4)
    assert len(service.embed_documents(["123456"])) == 1
    with pytest.raises(ValueError):
        service.embed_query("123456")
    service.close()


@pytest.mark.parametrize("options", [{"batch_size":0}, {"batch_size":True}, {"query_max_length":0}, {"query_max_length":8193}])
def test_bad_runtime_options_never_load_model(boundary, monkeypatch, options):
    with pytest.raises(ValueError):
        _service(boundary, monkeypatch, **options)
    assert boundary[2] == []


def test_context_closes_and_preserves_body_failure(boundary, monkeypatch):
    monkeypatch.setattr(_module(), "_snapshot_download", boundary[-1])
    boundary[1].close_error = RuntimeError("private cleanup")
    original = ValueError("primary failure")
    with pytest.raises(ValueError) as caught, _module().pinned_knowledge_embedding(
        KnowledgeIndexProfile(model_revision=REVISION)
    ) as service:
        assert service.dimension == 1024
        raise original
    assert caught.value is original
    assert boundary[1].close_count == 1
    assert any("cleanup" in note.lower() for note in original.__notes__)


def test_closed_service_rejects_inference(boundary, monkeypatch):
    service = _service(boundary, monkeypatch)
    service.close()
    service.close()
    with pytest.raises(EmbeddingError):
        service.embed_query("退款")


def test_explicit_cache_directory_reaches_local_resolver(boundary, monkeypatch, tmp_path):
    service = _service(boundary, monkeypatch, cache_dir=Path(tmp_path))
    assert boundary[3][0]["cache_dir"] == tmp_path
    service.close()
