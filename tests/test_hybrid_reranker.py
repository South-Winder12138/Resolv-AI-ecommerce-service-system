import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from resolv.schemas.hybrid_knowledge import (
    HybridRetrievalError,
    HybridRetrievalProfile,
    PreparedWindow,
)
from resolv.services import hybrid_reranker as module
from tests.test_hybrid_contracts import evidence

REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "model.safetensors",
)


def window(text="pair20"):
    payload = evidence().model_dump()
    payload.pop("rerank_score")
    payload["body_text"] = text
    return PreparedWindow(**payload)


class Tensor:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = shape or (len(values), 1)

    def to(self, *a, **kw):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *a):
        return self

    def tolist(self):
        return self.values


class Tokenizer:
    model_max_length = 8192

    def __init__(self):
        self.calls = []
        self.batches = []

    def __call__(self, query, text=None, **kw):
        self.calls.append((query, text, kw))
        length = int((text or query).removeprefix("pair").removeprefix("query"))
        return {"input_ids": [1] * length, "attention_mask": [1] * length}

    def pad(self, rows, **kw):
        self.batches.append((rows, kw))
        return {"input_ids": Tensor(rows), "attention_mask": Tensor(rows)}


def test_query_count_is_exact_without_forward_or_diagnostic_changes(boundary):
    ranker = module.LocalHybridReranker(HybridRetrievalProfile())
    ranker.query_tokens, ranker.pair_tokens = 7, (9,)
    assert ranker.count_query_tokens("query512") == 512
    assert (ranker.query_tokens, ranker.pair_tokens) == (7, (9,))
    assert boundary.model.forward_calls == 0
    with pytest.raises(HybridRetrievalError, match="query_too_long"):
        ranker.count_query_tokens("query513")
    ranker.close()


class Model:
    config = SimpleNamespace(max_position_embeddings=8194, pad_token_id=1, num_labels=1)

    def __init__(self):
        self.forward_calls = 0
        self.error = None
        self.scores = None
        self.eval_calls = 0
        self.moves = []
        self.kw = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, **kw):
        self.forward_calls += 1
        self.kw.append(kw)
        if self.error:
            raise self.error
        n = len(kw["input_ids"].values)
        vals = (
            self.scores
            if self.scores is not None
            else [float(i + self.forward_calls * 10) for i in range(n)]
        )
        return SimpleNamespace(logits=Tensor(vals))


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    path = tmp_path / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / REV
    path.mkdir(parents=True)
    for name in FILES:
        (path / name).write_text("{}")
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "xlm-roberta",
                "max_position_embeddings": 8194,
                "pad_token_id": 1,
            }
        )
    )
    tok, model, requests, loads = Tokenizer(), Model(), [], []

    def resolve(**kw):
        requests.append(kw)
        return str(path)

    def load_tok(*args, **kw):
        loads.append(("tokenizer", args, kw))
        return tok

    def load_model(*args, **kw):
        loads.append(("model", args, kw))
        return model

    torch = SimpleNamespace(
        float32="fp32",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        inference_mode=nullcontext,
        mps=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setattr(module, "_snapshot_download", resolve)
    monkeypatch.setattr(
        module,
        "_dependencies",
        lambda: (
            torch,
            SimpleNamespace(from_pretrained=load_tok),
            SimpleNamespace(from_pretrained=load_model),
        ),
    )
    return SimpleNamespace(
        path=path, tok=tok, model=model, torch=torch, requests=requests, loads=loads
    )


def create():
    return module.LocalHybridReranker(HybridRetrievalProfile())


def test_runtime_error_is_not_retried(boundary):
    service = create()
    boundary.model.error = RuntimeError("private MPS failure")
    with pytest.raises(HybridRetrievalError) as caught:
        service.score("query20", (window(),) * 8)
    assert boundary.model.forward_calls == 1
    assert caught.value.__cause__ is boundary.model.error
    assert "private" not in str(caught.value)
    service.close()


def test_boundaries_accept_512_and_3072_without_truncation(boundary):
    service = create()
    assert len(service.score("query512", (window("pair3072"),))) == 1
    assert service.query_tokens == 512 and service.pair_tokens == (3072,)
    assert all(kw["truncation"] is False for _, _, kw in boundary.tok.calls)
    assert "attention_mask" in boundary.model.kw[0]
    assert boundary.tok.batches[0][1] == {"padding": True, "return_tensors": "pt"}
    service.close()


@pytest.mark.parametrize(
    "query,windows",
    [("query513", (window(),)), ("query20", (window(), window("pair3073")))],
)
def test_overlong_input_rejected_before_any_forward(boundary, query, windows):
    service = create()
    with pytest.raises(HybridRetrievalError):
        service.score(query, windows)
    assert boundary.model.forward_calls == 0
    service.close()


def test_batches_four_preserve_order_and_empty_does_not_forward(boundary):
    service = create()
    assert service.score("query20", ()) == ()
    assert boundary.model.forward_calls == 0
    assert service.score("query20", (window(),) * 6) == (
        10.0,
        11.0,
        12.0,
        13.0,
        20.0,
        21.0,
    )
    assert [len(rows) for rows, _ in boundary.tok.batches] == [4, 2]
    assert boundary.model.eval_calls == 1
    assert boundary.loads[1][2]["local_files_only"] is True
    assert boundary.loads[1][2]["dtype"] == "fp32"
    assert boundary.model.moves == [(("mps",), {"dtype": "fp32"})]
    service.close()
    service.close()
    with pytest.raises(HybridRetrievalError):
        service.score("query20", (window(),))


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [], [1.0, 2.0]])
def test_bad_logits_fail_whole_call(boundary, scores):
    service = create()
    boundary.model.scores = scores
    with pytest.raises(HybridRetrievalError):
        service.score("query20", (window(),))
    assert boundary.model.forward_calls == 1
    service.close()


def test_mps_unavailable_never_loads_or_switches_model(boundary):
    boundary.torch.backends.mps.is_available = lambda: False
    with pytest.raises(HybridRetrievalError):
        create()
    assert boundary.loads == []


@pytest.mark.parametrize("file", FILES)
def test_snapshot_missing_file_rejected_without_load(boundary, file):
    (boundary.path / file).unlink()
    with pytest.raises(HybridRetrievalError):
        create()
    assert boundary.loads == []


def test_wrong_revision_snapshot_and_short_config_rejected(boundary, monkeypatch):
    monkeypatch.setattr(
        module, "_snapshot_download", lambda **kw: str(boundary.path.parent / "wrong")
    )
    with pytest.raises(HybridRetrievalError):
        create()
    assert boundary.loads == []


def test_small_model_capacity_is_rejected(boundary):
    (boundary.path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "xlm-roberta",
                "max_position_embeddings": 514,
                "pad_token_id": 1,
            }
        )
    )
    with pytest.raises(HybridRetrievalError):
        create()
    assert boundary.loads == []


def test_resolver_local_only_and_import_does_not_load(boundary):
    path = module.resolve_reranker_snapshot(REV)
    assert path == boundary.path
    assert boundary.requests == [
        {
            "repo_id": "BAAI/bge-reranker-v2-m3",
            "revision": REV,
            "cache_dir": None,
            "local_files_only": True,
        }
    ]
    assert boundary.loads == []


def test_enabled_implicit_cpu_fallback_rejected_before_model_load(
    boundary, monkeypatch
):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(HybridRetrievalError, match="reranker_cpu_fallback_enabled"):
        create()
    assert boundary.loads == []


def test_import_does_not_load_models_or_open_network():
    import subprocess
    import sys

    code = """
import builtins
import socket
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','transformers','huggingface_hub','FlagEmbedding'}:
        raise AssertionError('model dependency imported')
    return original(name,*args,**kwargs)
def denied(*a,**kw):
    raise AssertionError('network attempted')
socket.socket.connect=denied
socket.socket.connect_ex=denied
socket.getaddrinfo=denied
builtins.__import__=guarded
import resolv.services.hybrid_reranker
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
