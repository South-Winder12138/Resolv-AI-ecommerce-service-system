import importlib
import json

import pytest


def module():
    try:
        return importlib.import_module("scripts.smoke_understanding_recovery")
    except ModuleNotFoundError:
        pytest.fail("understanding recovery CLI missing")


@pytest.mark.parametrize(
    "url",
    [
        "mysql+pymysql://u:p@localhost/resolv_conversations?charset=utf8mb4",
        "sqlite:///temp.db",
        "mysql+pymysql://u:p@localhost/resolv_knowledge?charset=utf8mb4",
    ],
)
def test_formal_or_wrong_store_is_rejected_before_connect(monkeypatch, url):
    mod = module()
    monkeypatch.setenv("BAD_SMOKE_URL", url)
    monkeypatch.setattr(
        mod, "run_gate", lambda *_args, **_kwargs: pytest.fail("unsafe connect")
    )
    with pytest.raises(SystemExit):
        mod.main(["--database-env", "BAD_SMOKE_URL", "--setup"])


def test_setup_passes_only_validated_isolated_url_to_gate(monkeypatch):
    mod = module()
    monkeypatch.setenv(
        "TEST_SMOKE_URL",
        "mysql+pymysql://u:p@localhost/resolv_conversations_smoke_test?charset=utf8mb4",
    )
    calls = []
    monkeypatch.setattr(
        mod, "run_gate", lambda settings, **kwargs: calls.append(kwargs)
    )
    mod.main(["--database-env", "TEST_SMOKE_URL", "--setup"])
    assert calls == [{"setup": True}]


def test_foreign_manifest_is_rejected_before_connect(monkeypatch, tmp_path):
    mod = module()
    monkeypatch.setenv(
        "TEST_SMOKE_URL",
        "mysql+pymysql://u:p@localhost/resolv_conversations_smoke_test?charset=utf8mb4",
    )
    path = tmp_path / "foreign.json"
    path.write_text(
        json.dumps(
            {"owner": "someone else", "database": "resolv_conversations_smoke_test"}
        )
    )
    monkeypatch.setattr(
        mod, "run_phase", lambda *_: pytest.fail("foreign manifest used")
    )
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--database-env",
                "TEST_SMOKE_URL",
                "--phase",
                "resume",
                "--manifest",
                str(path),
            ]
        )


@pytest.mark.parametrize(
    "args",
    [
        ["--phase", "pause"],
        ["--phase", "verify", "--crash-at", "after_resolution"],
        ["--setup", "--phase", "resume"],
    ],
)
def test_invalid_cli_combinations_are_rejected_before_resources(args, monkeypatch):
    mod = module()
    monkeypatch.setenv(
        "TEST_SMOKE_URL",
        "mysql+pymysql://u:p@localhost/resolv_conversations_smoke_test?charset=utf8mb4",
    )
    with pytest.raises(SystemExit):
        mod.main(["--database-env", "TEST_SMOKE_URL", *args])


def test_phase_crash_exit_is_not_swallowed(monkeypatch, tmp_path):
    mod = module()
    monkeypatch.setenv(
        "TEST_SMOKE_URL",
        "mysql+pymysql://u:p@localhost/resolv_conversations_smoke_test?charset=utf8mb4",
    )

    def phase(*_):
        raise SystemExit(23)

    monkeypatch.setattr(mod, "run_phase", phase)
    with pytest.raises(SystemExit) as caught:
        mod.main(
            [
                "--database-env",
                "TEST_SMOKE_URL",
                "--phase",
                "pause",
                "--manifest",
                str(tmp_path / "owned.json"),
                "--crash-at",
                "after_resolution",
            ]
        )
    assert caught.value.code == 23
