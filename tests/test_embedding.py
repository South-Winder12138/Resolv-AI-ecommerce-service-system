import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from resolv.core.config import EmbeddingSettings
from resolv.services import embedding


def unit_vector(index=0):
    vector = np.zeros(1024, dtype=np.float32)
    vector[index] = 1
    return vector


class FakeTokenizer:
    def __init__(self):
        self.calls = []
        self.error = None

    def __call__(self, texts, *, truncation, padding, add_special_tokens):
        self.calls.append((texts, truncation, padding, add_special_tokens))
        if self.error:
            raise self.error
        # One fake token per character, plus two special tokens.
        return {"input_ids": [[0] * (len(text) + 2) for text in texts]}


class FakeModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.result = None
        self.error = None
        self.close_error = None
        self.close_count = 0

    def encode_corpus(self, texts, **kwargs):
        return self._encode("documents", texts, kwargs)

    def encode_queries(self, texts, **kwargs):
        return self._encode("query", texts, kwargs)

    def _encode(self, kind, texts, kwargs):
        self.calls.append((kind, texts, kwargs))
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return {
            "dense_vecs": np.array([unit_vector(i) for i in range(len(texts))]),
            "lexical_weights": None,
            "colbert_vecs": None,
        }

    def stop_self_pool(self):
        self.close_count += 1
        if self.close_error:
            raise self.close_error


@pytest.fixture
def model_service(monkeypatch):
    model = FakeModel()
    loads = []

    def load(settings):
        loads.append(settings)
        return model

    monkeypatch.setattr(embedding, "_load_model", load)
    settings = EmbeddingSettings(
        _env_file=None, query_max_length=8, passage_max_length=12
    )
    service = embedding.EmbeddingService(settings)
    yield model, service, loads
    service.close()


def test_model_factory_explicit_dense_cpu_fp32(monkeypatch):
    created = []
    model = FakeModel()

    def factory(name, **kwargs):
        created.append((name, kwargs))
        return model

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=factory)
    )
    service = embedding.EmbeddingService(EmbeddingSettings(_env_file=None))
    assert service.dimension == 1024
    assert created == [
        (
            "BAAI/bge-m3",
            {
                "devices": "cpu",
                "use_fp16": False,
                "normalize_embeddings": True,
                "batch_size": 8,
                "query_max_length": 512,
                "passage_max_length": 512,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
                "query_instruction_for_retrieval": None,
                "trust_remote_code": False,
            },
        )
    ]
    service.close()


def test_documents_preserve_order_with_plain_python_vectors(model_service):
    model, service, _ = model_service
    result = service.embed_documents(["first", "second"])
    assert len(result) == 2
    assert len(result[0]) == len(result[1]) == 1024
    assert result[0][:3] == [1.0, 0.0, 0.0]
    assert result[1][:3] == [0.0, 1.0, 0.0]
    assert type(result[0][0]) is float
    assert model.calls == [
        (
            "documents",
            ["first", "second"],
            {
                "batch_size": 8,
                "max_length": 12,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            },
        )
    ]


def test_query_uses_query_entrypoint_and_its_own_limit(model_service):
    model, service, _ = model_service
    result = service.embed_query("refund")
    assert len(result) == 1024
    assert result[:3] == [1.0, 0.0, 0.0]
    assert model.calls == [
        (
            "query",
            ["refund"],
            {
                "batch_size": 8,
                "max_length": 8,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            },
        )
    ]


def test_model_reused_across_document_and_query_calls(model_service):
    _, service, loads = model_service
    service.embed_query("a")
    service.embed_documents(["b"])
    service.embed_query("c")
    assert len(loads) == 1


def test_empty_document_batch_does_not_tokenize_or_encode(model_service):
    model, service, _ = model_service
    assert service.embed_documents([]) == []
    assert model.tokenizer.calls == model.calls == []


@pytest.mark.parametrize("texts", ["single string", None, [""], [" \t"], ["valid", 4]])
def test_invalid_document_batch_fails_before_encode(model_service, texts):
    model, service, _ = model_service
    with pytest.raises(ValueError):
        service.embed_documents(texts)
    assert model.calls == model.tokenizer.calls == []


@pytest.mark.parametrize("query", ["", " \n", None, 4, ["query"]])
def test_invalid_query_fails_before_encode(model_service, query):
    model, service, _ = model_service
    with pytest.raises(ValueError):
        service.embed_query(query)
    assert model.calls == model.tokenizer.calls == []


def test_token_count_includes_special_tokens_at_boundary(model_service):
    model, service, _ = model_service
    service.embed_query("123456")  # 6 content + 2 special = query limit 8
    assert model.tokenizer.calls == [(["123456"], False, False, True)]
    assert len(model.calls) == 1
    with pytest.raises(ValueError, match="token"):
        service.embed_query("1234567")
    assert len(model.calls) == 1


def test_document_limit_separate_from_query_limit(model_service):
    model, service, _ = model_service
    service.embed_documents(["1234567890"])  # 10+2 = passage limit 12
    with pytest.raises(ValueError, match="token"):
        service.embed_query("1234567890")
    assert len(model.calls) == 1


def test_overlong_document_rejects_entire_batch(model_service):
    model, service, _ = model_service
    with pytest.raises(ValueError, match="token"):
        service.embed_documents(["short", "12345678901"])
    assert model.calls == []


def test_float64_output_converted_to_float32_values(model_service):
    model, service, _ = model_service
    vector = np.zeros((1, 1024), dtype=np.float64)
    vector[0, :2] = [0.6, 0.8]
    model.result = {"dense_vecs": vector, "lexical_weights": None, "colbert_vecs": None}
    result = service.embed_query("a")
    assert result[:2] == [0.6000000238418579, 0.800000011920929]


@pytest.mark.parametrize(
    "dense",
    [
        None,
        [],
        np.zeros((2, 1024)),
        np.zeros((1, 1023)),
        np.ones(1024),
        np.zeros((1, 1024)),
        np.ones((1, 1024)),
        np.full((1, 1024), np.nan),
        np.full((1, 1024), np.inf),
        np.full((1, 1024), 1e300),
        [["invalid"] * 1024],
    ],
)
def test_invalid_dense_output_is_embedding_error(model_service, dense):
    model, service, _ = model_service
    model.result = {"dense_vecs": dense, "lexical_weights": None, "colbert_vecs": None}
    with pytest.raises(embedding.EmbeddingError) as error:
        service.embed_query("a")
    assert error.value.__cause__ is not None


def test_missing_dense_output_is_embedding_error(model_service):
    model, service, _ = model_service
    model.result = {"lexical_weights": None, "colbert_vecs": None}
    with pytest.raises(embedding.EmbeddingError) as error:
        service.embed_query("a")
    assert isinstance(error.value.__cause__, KeyError)


def test_loading_failure_keeps_cause(monkeypatch):
    failure = RuntimeError("private loader detail")

    def load(settings):
        raise failure

    monkeypatch.setattr(embedding, "_load_model", load)
    with pytest.raises(embedding.EmbeddingError) as error:
        embedding.EmbeddingService(EmbeddingSettings(_env_file=None))
    assert error.value.__cause__ is failure
    assert "private loader detail" not in str(error.value)


@pytest.mark.parametrize("stage", ["tokenizer", "encode"])
def test_model_failures_keep_cause(model_service, stage):
    model, service, _ = model_service
    failure = RuntimeError("private model detail")
    if stage == "tokenizer":
        model.tokenizer.error = failure
    else:
        model.error = failure
    with pytest.raises(embedding.EmbeddingError) as error:
        service.embed_query("a")
    assert error.value.__cause__ is failure
    assert "private model detail" not in str(error.value)


def test_close_idempotent_and_closed_service_never_reloads(model_service):
    model, service, loads = model_service
    service.close()
    service.close()
    assert model.close_count == 1
    for call in (lambda: service.embed_query("a"), lambda: service.embed_documents([])):
        with pytest.raises(embedding.EmbeddingError, match="closed"):
            call()
    assert len(loads) == 1


def test_close_failure_keeps_cause_and_service_stays_closed(model_service):
    model, service, _ = model_service
    failure = RuntimeError("cleanup detail")
    model.close_error = failure
    with pytest.raises(embedding.EmbeddingError) as error:
        service.close()
    assert error.value.__cause__ is failure
    service.close()
    assert model.close_count == 1
    with pytest.raises(embedding.EmbeddingError, match="closed"):
        service.embed_query("a")


def test_module_import_does_not_import_model_framework():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import resolv.services.embedding; "
                "assert 'FlagEmbedding' not in sys.modules; assert 'torch' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
