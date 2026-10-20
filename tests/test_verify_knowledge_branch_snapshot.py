"""Committed exports must prove both sources, network denial and actual compiler errors."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from scripts.verify_knowledge_branch_snapshot import verify_snapshot

from tests.test_verify_hybrid_snapshot import repo, runner

__all__ = ["repo"]
MODULES = Path(
    os.environ.get(
        "MH_SNAPSHOT_FRONTEND_DEPENDENCIES",
        Path(__file__).resolve().parents[1] / "frontend/node_modules",
    )
)


def add_frontend(root, *, broken=False):
    front = root / "frontend"
    (front / "src").mkdir(parents=True)
    scripts = {name: 'node -e "process.exit(0)"' for name in ("test", "build", "lint")}
    scripts["typecheck"] = "tsc -p tsconfig.json --pretty false"
    (front / "package.json").write_text(json.dumps({"scripts": scripts}))
    (front / "tsconfig.json").write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "strict": True,
                    "noEmit": True,
                    "skipLibCheck": True,
                    "types": [],
                },
                "include": ["src"],
            }
        )
    )
    (front / "src/main.ts").write_text(
        "const value: string = " + ("123" if broken else "'ok'") + ";\n"
    )
    subprocess.run(["git", "-C", str(root), "add", "frontend"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "frontend fixture",
        ],
        check=True,
    )
    (front / "node_modules").symlink_to(MODULES, target_is_directory=True)


def output(root, name="snapshot"):
    return root / ".superpowers/sdd/2026-10-15-resolv-knowledge-branch" / name


def test_committed_wrapper_same_tree_tcp_dns_guard_and_full_lint_discovery(repo):
    add_frontend(repo)
    result = verify_snapshot("HEAD", output(repo), root=repo, python_runner=runner())
    assert result["passed"]
    assert (
        result["python"]["revision"]
        == result["frontend"]["revision"]
        == result["candidate_commit"]
    )
    assert result["python"]["tree"] == result["frontend"]["tree"] == result["tree"]
    assert result["python"]["network_attempts"] == 0
    assert result["python"]["negative_probe_attempts"] >= 2
    assert (
        result["python"]["ruff_discovered"]
        == result["python"]["expected_python_files"]
        > 0
    )
    assert set(result["frontend"]["checks"]) == {"test", "typecheck", "build", "lint"}
    for path in (output(repo), repo / "elsewhere"):
        with pytest.raises(ValueError):
            verify_snapshot("HEAD", path, root=repo)


def test_real_typescript_failure_ignores_workspace_fix(repo):
    add_frontend(repo, broken=True)
    (repo / "frontend/src/main.ts").write_text(
        "const value: string = 'uncommitted fix';\n"
    )
    result = verify_snapshot("HEAD", output(repo), root=repo, python_runner=runner())
    assert not result["passed"]
    assert result["frontend"]["checks"]["typecheck"] != 0
    assert "TS2322" in (output(repo) / "frontend/typecheck.log").read_text()


def test_workspace_selector_is_an_enum_not_an_arbitrary_root(repo):
    from scripts.verify_frontend_snapshot import verify_snapshot as frontend
    from scripts.verify_hybrid_snapshot import verify_snapshot as python

    for verifier in (python, frontend):
        with pytest.raises(ValueError):
            verifier("HEAD", output(repo), root=repo, workspace_name="../../escape")


def test_python_export_provides_explicit_readonly_node_dependencies(repo):
    add_frontend(repo)
    base = runner()

    def observed(command, **kwargs):
        if kwargs["env"]["MH_HYBRID_SNAPSHOT_PHASE"] == "pytest":
            assert (
                Path(kwargs["env"]["MH_SNAPSHOT_FRONTEND_DEPENDENCIES"])
                == repo / "frontend/node_modules"
            )
        return base(command, **kwargs)

    result = verify_snapshot("HEAD", output(repo), root=repo, python_runner=observed)
    assert result["passed"]
