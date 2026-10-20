import pytest
from scripts import smoke_hybrid_reranker as module


def test_help_does_not_initialize_model(monkeypatch, capsys):
    monkeypatch.setattr(
        module, "LocalHybridReranker", lambda *a, **kw: pytest.fail("loaded model")
    )
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "--revision" in capsys.readouterr().out


def test_smoke_failure_has_no_success_json(tmp_path, monkeypatch, capsys):
    def fail(*a, **kw):
        raise RuntimeError("private credentials")

    monkeypatch.setattr(module, "run_gate", fail)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--revision",
                "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                "--device",
                "mps",
                "--report-dir",
                str(tmp_path / "gate"),
            ]
        )
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "gate" / "result.json").exists()
    assert "private" not in (tmp_path / "gate" / "failure.json").read_text()


def test_gate_reports_near_limit_one_four_and_fifty_pairs(monkeypatch):
    import sys
    from types import SimpleNamespace

    from resolv.schemas.hybrid_knowledge import HybridRetrievalProfile

    class Reranker:
        snapshot = SimpleNamespace(name="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")

        def __init__(self, profile):
            pass

        def _tokenizer(self, query, text, **kw):
            return {"input_ids": [1] * (21 + len(text.split()))}

        def score(self, query, windows):
            self.query_tokens = 19
            self.pair_tokens = tuple(
                len(self._tokenizer(query, w.body_text)["input_ids"]) for w in windows
            )
            return (1.0,) * len(windows)

        def close(self):
            pass

    monkeypatch.setattr(module, "LocalHybridReranker", Reranker)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            mps=SimpleNamespace(
                synchronize=lambda: None,
                current_allocated_memory=lambda: 1,
                driver_allocated_memory=lambda: 2,
            )
        ),
    )
    report = module.run_gate(HybridRetrievalProfile())
    boundary = [x for x in report["records"] if x["mode"] == "near_limit"]
    assert [x["pairs"] for x in boundary] == [1, 4, 50]
    assert all(
        len(x["pair_tokens"]) == x["score_count"] == x["pairs"] for x in boundary
    )
