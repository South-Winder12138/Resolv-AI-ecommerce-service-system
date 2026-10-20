import importlib

import pytest

from resolv.core.config import EmbeddingSettings
from resolv.services.embedding import EmbeddingError


@pytest.fixture
def smoke_boundary(monkeypatch):
    smoke = importlib.import_module("scripts.smoke_embedding")
    events = []

    class FakeService:
        dimension = 1024
        error_stage = None
        close_error = False
        close_count = 0

        def __init__(self, settings):
            events.append(("load", settings.device, settings.use_fp16))

        def embed_documents(self, texts):
            events.append(("documents", texts))
            if self.error_stage == "documents":
                raise EmbeddingError("Dense embedding failed.")
            return [[1.0] + [0.0] * 1023 for text in texts]

        def embed_query(self, text):
            events.append(("query", text))
            if self.error_stage == "query":
                raise EmbeddingError("Dense embedding failed.")
            return [1.0] + [0.0] * 1023

        def close(self):
            type(self).close_count += 1
            if self.close_error:
                raise EmbeddingError("Embedding model cleanup failed.")

    monkeypatch.setattr(smoke, "EmbeddingService", FakeService)
    monkeypatch.setattr(
        smoke, "EmbeddingSettings", lambda: EmbeddingSettings(_env_file=None)
    )
    return smoke, FakeService, events


def test_smoke_encodes_actual_refund_chunk_and_query(smoke_boundary, capsys):
    smoke, service, events = smoke_boundary
    smoke.main()
    assert events == [
        ("load", "cpu", False),
        (
            "documents",
            ["售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。"],
        ),
        ("query", "软件已经激活了，还能退款吗？"),
    ]
    output = capsys.readouterr().out
    for expected in (
        "BAAI/bge-m3",
        "cpu",
        "FP16: False",
        "Normalize: True",
        "knowledge/refund.md",
        "Chunk count: 1",
        "dimension=1024",
        "L2 norm=1.000000",
        "软件已经激活了，还能退款吗？",
        "FlagEmbedding",
        "torch",
        "transformers",
    ):
        assert expected in output
    assert service.close_count == 1


@pytest.mark.parametrize("stage", ["documents", "query"])
def test_smoke_inference_failure_exits_and_closes(smoke_boundary, capsys, stage):
    smoke, service, _ = smoke_boundary
    service.error_stage = stage
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code != 0
    assert isinstance(error.value.__cause__, EmbeddingError)
    assert service.close_count == 1
    assert "complete" not in capsys.readouterr().out.lower()


def test_smoke_invalid_settings_exits_before_loading(smoke_boundary, monkeypatch):
    smoke, service, events = smoke_boundary
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code != 0
    assert service.close_count == 0
    assert events == []


def test_smoke_loading_failure_exits_without_owning_model(smoke_boundary, monkeypatch):
    smoke, service, _ = smoke_boundary

    def broken(settings):
        raise EmbeddingError("Embedding model loading failed.")

    monkeypatch.setattr(smoke, "EmbeddingService", broken)
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code != 0
    assert service.close_count == 0


def test_smoke_cleanup_failure_is_nonzero(smoke_boundary):
    smoke, service, _ = smoke_boundary
    service.close_error = True
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code != 0
    assert isinstance(error.value.__cause__, EmbeddingError)
    assert service.close_count == 1


def test_smoke_cleanup_does_not_replace_inference_failure(smoke_boundary):
    smoke, service, _ = smoke_boundary
    service.error_stage = "query"
    service.close_error = True
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert "Dense embedding failed" in str(error.value)
    assert service.close_count == 1
