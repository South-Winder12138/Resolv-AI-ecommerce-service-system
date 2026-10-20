import json
from pathlib import Path

from tests.test_logistics_contracts import api
from tests.test_verify_hybrid_snapshot import repo, runner
from tests.test_verify_knowledge_branch_snapshot import add_frontend

__all__ = ["repo"]


def output(root):
    return root / ".superpowers/sdd/2026-10-20-resolv-logistics-mcp/export"


def test_logistics_gate_includes_all_python_and_two_network_probes(repo):
    add_frontend(repo)
    report = api("scripts.verify_logistics_snapshot").verify_logistics_snapshot("HEAD", output(repo), root=repo, python_runner=runner())
    assert report["passed"]
    assert report["python"]["negative_probe_attempts"] == 2 and report["python"]["network_attempts"] == 0
    assert report["python"]["ruff_discovered"] == report["python"]["expected_python_files"] > 0
    assert report["python"]["revision"] == report["frontend"]["revision"] == report["candidate_commit"]


def test_exported_runner_keeps_owned_resource_root(repo):
    add_frontend(repo)
    report = api("scripts.verify_logistics_snapshot").verify_logistics_snapshot("HEAD", output(repo), root=repo, python_runner=runner())
    source = Path(report["exported_source"])
    assert source == (output(repo) / "python/source").resolve()
    marker = json.loads((source / ".logistics-source.json").read_text())
    assert marker["revision"] == report["candidate_commit"]
    assert marker["resource_root"] == str((repo / ".superpowers/sdd/2026-10-20-resolv-logistics-mcp").resolve())
    assert marker["resource_root"] != str(source)


def test_export_never_imports_dirty_worktree(repo):
    add_frontend(repo, broken=True)
    (repo / "frontend/src/main.ts").write_text("const value: string = 'dirty fix';")
    report = api("scripts.verify_logistics_snapshot").verify_logistics_snapshot("HEAD", output(repo), root=repo, python_runner=runner())
    assert report["passed"] is False and report["frontend"]["checks"]["typecheck"] != 0
    assert "dirty fix" not in (Path(report["exported_source"]) / "frontend/src/main.ts").read_text()
