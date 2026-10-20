import importlib
import json
import subprocess

import pytest


def api():
    try:
        return importlib.import_module("scripts.smoke_three_branch_recovery")
    except ModuleNotFoundError:
        pytest.fail("isolated routed recovery gate missing")


@pytest.mark.parametrize(
    "database,path",
    [
        ("resolv_conversations", "business.db"),
        ("resolv_conversations_smoke_threebranches_20261014", "data/resolv.db"),
        ("resolv_conversations_smoke_other", "business.db"),
    ],
)
def test_invalid_database_or_formal_path_is_refused_before_connect(database, path):
    from pathlib import Path

    with pytest.raises(ValueError):
        api().validate_isolation(database, Path(path))


def test_existing_business_file_requires_exact_owner_marker(tmp_path, monkeypatch):
    module = api()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    path = tmp_path / module.WORKSPACE / "business.db"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"user data")
    with pytest.raises(ValueError):
        module.validate_isolation(module.DATABASE, path)
    module.marker_path(path).write_text(
        json.dumps({"owner": "someone", "database": module.DATABASE, "path": str(path)})
    )
    with pytest.raises(ValueError):
        module.validate_isolation(module.DATABASE, path)
    assert path.read_bytes() == b"user data"


def test_cross_database_state_is_rejected_without_execution(tmp_path):
    module = api()
    state = {
        "owner": module.OWNER,
        "database": "wrong",
        "business_db": str(tmp_path / "business.db"),
    }
    with pytest.raises(ValueError):
        module.validate_state(state, module.DATABASE, tmp_path / "business.db")


def test_pause_exit_23_does_not_count_as_completed_window(tmp_path):
    module = api()
    calls = []

    def runner(command, **kwargs):
        mode = command[command.index("--mode") + 1]
        calls.append(mode)
        return subprocess.CompletedProcess(command, 23, '{"passed":true}', "")

    result = module.run_window(
        "after_sqlite_commit",
        database=module.DATABASE,
        business_db=tmp_path / "business.db",
        state_file=tmp_path / "state.json",
        report=tmp_path / "report.jsonl",
        runner=runner,
    )
    assert result["passed"] is False
    assert calls == ["pause", "resume"] and result["exit_codes"] == [23, 23]


def test_smoke_cli_redacts_initialization_failure(tmp_path, monkeypatch, capsys):
    module = api()
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def failed(*args):
        raise RuntimeError("mysql-password credential-secret")

    monkeypatch.setattr(module, "settings_for_smoke", failed)
    report = tmp_path / "report.jsonl"
    result = module.main(
        [
            "--mode",
            "matrix",
            "--database",
            module.DATABASE,
            "--business-db",
            str(tmp_path / module.WORKSPACE / "business.db"),
            "--report",
            str(report),
        ]
    )
    assert result == 1
    assert "credential-secret" not in capsys.readouterr().out + report.read_text()
