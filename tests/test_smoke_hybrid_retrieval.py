import pytest
from scripts import smoke_hybrid_retrieval as module


def test_help_is_resource_free(monkeypatch, capsys):
    monkeypatch.setattr(
        module, "run", lambda *a, **kw: pytest.fail("created resources")
    )
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "--reranker-revision" in capsys.readouterr().out


def test_failure_no_success_json_or_private_text(tmp_path, monkeypatch, capsys):
    def fail(*a, **kw):
        raise OSError("private query credentials")

    monkeypatch.setattr(module, "run", fail)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--version",
                "v3",
                "--collection",
                "mhhybrid_gate_fixture",
                "--reranker-revision",
                "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                "--query",
                "private query",
                "--report-dir",
                str(tmp_path / "gate"),
            ]
        )
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "gate" / "result.json").exists()
    assert "private" not in (tmp_path / "gate" / "failure.json").read_text()
