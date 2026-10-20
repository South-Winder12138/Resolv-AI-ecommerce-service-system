import pytest
from scripts import prepare_hybrid_reranker as module

from resolv.schemas.hybrid_knowledge import HybridRetrievalError

REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def test_prepare_download_is_explicit_and_pinned(tmp_path, monkeypatch):
    calls = []

    def missing(*a, **kw):
        raise HybridRetrievalError("snapshot_missing")

    monkeypatch.setattr(module, "resolve_reranker_snapshot", missing)
    monkeypatch.setattr(
        module, "_snapshot_download", lambda **kw: calls.append(kw) or str(tmp_path)
    )
    results = iter([None, tmp_path])

    def resolve(*a, **kw):
        v = next(results)
        if v is None:
            raise HybridRetrievalError("snapshot_missing")
        return v

    monkeypatch.setattr(module, "resolve_reranker_snapshot", resolve)
    assert module.prepare(REV, cache_dir=tmp_path) == tmp_path
    assert calls[0]["revision"] == REV and calls[0]["local_files_only"] is False
    assert calls[0]["repo_id"] == "BAAI/bge-reranker-v2-m3"


def test_existing_complete_cache_reused_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "resolve_reranker_snapshot", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(
        module, "_snapshot_download", lambda **kw: pytest.fail("downloaded")
    )
    assert module.prepare(REV) == tmp_path


def test_prepare_rejects_unpinned_revision_before_network(monkeypatch):
    monkeypatch.setattr(
        module, "_snapshot_download", lambda **kw: pytest.fail("downloaded")
    )
    with pytest.raises(HybridRetrievalError):
        module.prepare("main")


def test_explicit_preparation_populates_full_snapshot_for_local_resolver(
    tmp_path, monkeypatch
):
    attempts = iter([False, True])

    def resolve(*a, **kw):
        if not next(attempts):
            raise HybridRetrievalError("reranker_snapshot_unavailable")
        return tmp_path

    calls = []
    monkeypatch.setattr(module, "resolve_reranker_snapshot", resolve)
    monkeypatch.setattr(
        module, "_snapshot_download", lambda **kw: calls.append(kw) or str(tmp_path)
    )
    assert module.prepare(REV) == tmp_path
    assert "allow_patterns" not in calls[0]
