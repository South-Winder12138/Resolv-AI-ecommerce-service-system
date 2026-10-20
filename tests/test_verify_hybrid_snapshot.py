import subprocess
from types import SimpleNamespace

import pytest
from scripts.verify_hybrid_snapshot import verify_snapshot

from resolv.schemas.hybrid_knowledge import HybridRetrievalError


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src/resolv").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts").mkdir()
    (root / "migrations/knowledge").mkdir(parents=True)
    (root / "src/resolv/__init__.py").write_text("ORIGIN='tracked'\n")
    (root / "tests/test_minimal.py").write_text("def test_ok(): assert True\n")
    (root / "scripts/__init__.py").touch()
    (root / "migrations/knowledge/__init__.py").touch()
    (root / ".gitignore").write_text(".superpowers/\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "add",
            "src/resolv/__init__.py",
            "tests/test_minimal.py",
            "scripts/__init__.py",
            "migrations/knowledge/__init__.py",
            ".gitignore",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Snapshot Test",
            "-c",
            "user.email=snapshot@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


def output(root, name="out"):
    return root / ".superpowers/sdd/2026-10-14-resolv-hybrid-retrieval" / name


def runner(codes=None, network=False, real_pytest=False):
    def run(cmd, **kw):
        phase = kw.get("env", {}).get("MH_HYBRID_SNAPSHOT_PHASE")
        if phase in ("pytest", "ruff") and not (phase == "pytest" and real_pytest):
            if network and phase == "pytest":
                from pathlib import Path

                Path(kw["env"]["MH_HYBRID_NETWORK_LOG"]).write_text(
                    '{"method":"connect","family":2}\n'
                )
            return SimpleNamespace(
                returncode=(codes or {}).get(phase, 0), stdout="", stderr=""
            )
        return subprocess.run(cmd, check=kw.pop("check", False), **kw)

    return run


def test_snapshot_rejects_untracked_import_fallback(repo, monkeypatch):
    (repo / "src/resolv/only_untracked.py").write_text("value=1\n")
    (repo / "tests/test_minimal.py").write_text(
        "from resolv.only_untracked import value\ndef test_value(): assert value==1\n"
    )
    subprocess.run(["git", "-C", str(repo), "add", "tests/test_minimal.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Snapshot Test",
            "-c",
            "user.email=snapshot@example.invalid",
            "commit",
            "-qm",
            "tracked import",
        ],
        check=True,
    )
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    report = verify_snapshot(
        "HEAD", output(repo), root=repo, runner=runner(real_pytest=True)
    )
    assert report["passed"] is False and report["checks"]["pytest"] != 0
    assert not (output(repo) / "source/src/resolv/only_untracked.py").exists()


def test_snapshot_contains_only_selected_commit_and_separate_negative_probe(repo):
    (repo / "untracked.txt").write_text("excluded")
    report = verify_snapshot("HEAD", output(repo), root=repo, runner=runner())
    assert report["passed"] is True
    assert not (output(repo) / "source/untracked.txt").exists()
    assert report["network_attempts"] == 0 and report["negative_probe_attempts"] == 1
    assert report["source_import_verified"] is True
    assert len(report["commit"]) == len(report["tree"]) == 40


@pytest.mark.parametrize(
    "codes,network", [({"pytest": 1}, False), ({"ruff": 1}, False), ({}, True)]
)
def test_failure_or_network_attempt_prevents_pass(repo, codes, network):
    report = verify_snapshot(
        "HEAD", output(repo), root=repo, runner=runner(codes, network)
    )
    assert report["passed"] is False


def test_output_path_escape_and_existing_directory_rejected(repo):
    with pytest.raises(HybridRetrievalError):
        verify_snapshot("HEAD", repo.parent / "outside", root=repo, runner=runner())
    p = output(repo)
    p.mkdir(parents=True)
    with pytest.raises(HybridRetrievalError):
        verify_snapshot("HEAD", p, root=repo, runner=runner())


def test_guard_preserves_declared_runtime_dependencies_while_removing_workspace_source(
    repo, tmp_path
):
    import os
    import sys

    from scripts.verify_hybrid_snapshot import GUARD

    deps = repo / ".venv/lib/site-packages"
    deps.mkdir(parents=True)
    (deps / "declared_dependency.py").write_text("value=42\n")
    source = tmp_path / "export"
    (source / "src").mkdir(parents=True)
    code = f"import sys\nsys.prefix={str(repo / '.venv')!r}\nsys.path.append({str(deps)!r})\nexec({GUARD!r})\nimport declared_dependency\nassert declared_dependency.value==42\n"
    env = os.environ.copy()
    env.update(
        MH_HYBRID_SNAPSHOT_SOURCE=str(source),
        MH_HYBRID_SNAPSHOT_ORIGINAL=str(repo),
        MH_HYBRID_NETWORK_LOG=str(tmp_path / "net.log"),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_real_ruff_rejects_lint_error_in_export_beneath_ignored_parent(repo):
    ignored_parent = repo / ".superpowers/sdd"
    ignored_parent.mkdir(parents=True)
    (ignored_parent / ".gitignore").write_text("*\n")
    (repo / "src/resolv/broken.py").write_text("import os\n")
    subprocess.run(["git", "-C", str(repo), "add", "src/resolv/broken.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Snapshot Test",
                    "-c", "user.email=snapshot@example.invalid", "commit", "-qm", "lint error"], check=True)
    fake = runner()

    def real_lint(cmd, **kw):
        if kw["env"]["MH_HYBRID_SNAPSHOT_PHASE"] == "ruff":
            return subprocess.run(cmd, check=kw.pop("check", False), **kw)
        return fake(cmd, **kw)

    report = verify_snapshot("HEAD", output(repo), root=repo, runner=real_lint)
    assert report["passed"] is False
    assert report["checks"]["ruff"] != 0
    assert "F401" in (output(repo) / "ruff.log").read_text()


def test_zero_file_lint_discovery_prevents_snapshot_pass(repo):
    fake = runner()

    def empty_discovery(cmd, **kw):
        if kw["env"]["MH_HYBRID_SNAPSHOT_PHASE"] == "ruff_files":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return fake(cmd, **kw)

    report = verify_snapshot("HEAD", output(repo), root=repo, runner=empty_discovery)
    assert report["passed"] is False
