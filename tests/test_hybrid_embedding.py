import pytest

from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.services import hybrid_embedding as module
from tests.hybrid_fixtures import profile


def test_adapter_maps_only_bge_fields_and_keeps_v3_profile(monkeypatch):
    loads = []

    class Dense:
        dimension = 1024

        def __init__(self, p, **kw):
            loads.append((p, kw))

        def embed_query(self, t):
            return [1.0] + [0.0] * 1023

        def embed_documents(self, t):
            return [self.embed_query(x) for x in t]

        def close(self):
            pass

    monkeypatch.setattr(module, "KnowledgeEmbeddingService", Dense)
    p = profile()
    service = module.HybridEmbeddingAdapter(p)
    assert service.profile == p and service.dimension == 1024
    assert loads[0][0].schema_version == "knowledge-v2"
    assert loads[0][0].model_revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert len(service.embed_query("原query")) == 1024
    assert len(service.embed_documents(("正文",))) == 1
    service.close()


def test_other_dense_revision_refused_before_loading(monkeypatch):
    monkeypatch.setattr(
        module, "KnowledgeEmbeddingService", lambda *a, **kw: pytest.fail("loaded")
    )
    with pytest.raises(HybridRetrievalError):
        module.HybridEmbeddingAdapter(profile(model_revision="a" * 40))
