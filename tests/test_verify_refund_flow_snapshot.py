import importlib
import subprocess

from tests.test_verify_hybrid_snapshot import repo, runner
from tests.test_verify_knowledge_branch_snapshot import add_frontend

__all__ = ["repo"]


def verifier():
    assert importlib.util.find_spec("scripts.verify_refund_flow_snapshot"), (
        "missing refund snapshot wrapper"
    )
    return importlib.import_module(
        "scripts.verify_refund_flow_snapshot"
    ).verify_snapshot


def output(root):
    return root / ".superpowers/sdd/2026-10-16-resolv-refund-flow/export"


def test_snapshot_denies_tcp_dns_and_discovers_all_python(repo):
    add_frontend(repo)
    report = verifier()("HEAD", output(repo), root=repo, python_runner=runner())
    assert report["passed"]
    py, front = report["python"], report["frontend"]
    assert py["revision"] == front["revision"] == report["candidate_commit"]
    assert py["tree"] == front["tree"] == report["tree"]
    assert py["network_attempts"] == 0 and py["negative_probe_attempts"] == 2
    assert py["ruff_discovered"] == py["expected_python_files"] > 0


def test_committed_refund_export_ignores_workspace_repairs(repo):
    add_frontend(repo, broken=True)
    (repo / "frontend/src/main.ts").write_text("const value: string = 'workspace fix';")
    report = verifier()("HEAD", output(repo), root=repo, python_runner=runner())
    assert report["passed"] is False
    assert report["frontend"]["checks"]["typecheck"] != 0


def test_committed_python_error_is_not_repaired_by_workspace(repo):
    add_frontend(repo)
    path = repo / "src/resolv/__init__.py"
    path.write_text("def broken(:\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "src/resolv/__init__.py"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "bad python",
        ],
        check=True,
    )
    path.write_text("ORIGIN='workspace repair'\n")
    report = verifier()("HEAD", output(repo), root=repo, python_runner=runner())
    assert (
        report["passed"] is False and report["python"]["checks"]["source_import"] != 0
    )


def test_pytest_cannot_shadow_the_export_network_audit(tmp_path):
    import os
    import sys
    from pathlib import Path

    from scripts.verify_hybrid_snapshot import GUARD

    root = Path(__file__).resolve().parents[1]
    log = tmp_path / "audit.jsonl"
    code = (
        "import importlib.util, types\n"
        + f"exec({GUARD!r})\n"
        + f"spec=importlib.util.spec_from_file_location('gate_conftest',{str(root / 'tests/conftest.py')!r})\nm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\nm.pytest_configure(types.SimpleNamespace())\n"
        + "import socket\ntry:socket.getaddrinfo('audit.invalid',443)\nexcept RuntimeError:pass\n"
    )
    env = os.environ.copy()
    env.update(
        MH_HYBRID_SNAPSHOT_SOURCE=str(root),
        MH_HYBRID_SNAPSHOT_ORIGINAL=str(tmp_path / "original"),
        MH_HYBRID_NETWORK_LOG=str(log),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert log.exists() and len(log.read_text().splitlines()) == 1
