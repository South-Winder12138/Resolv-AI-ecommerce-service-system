import importlib
import subprocess
from pathlib import Path

import pytest


def test_snapshot_exports_only_committed_files_and_runs_all_commands(tmp_path):
    try:
        verify = importlib.import_module(
            "scripts.verify_frontend_snapshot"
        ).verify_snapshot
    except ModuleNotFoundError:
        pytest.fail("commit snapshot verification missing")
    root = tmp_path / "repo"
    (root / "frontend/src").mkdir(parents=True)
    (root / "frontend/package.json").write_text('{"name":"fixture"}')
    (root / "frontend/src/main.ts").write_text('import "./uncommitted.ts"')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "frontend"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    (root / "frontend/src/uncommitted.ts").write_text("user work")
    (root / "frontend/node_modules/.tmp").mkdir(parents=True)
    (root / "frontend/node_modules/.tmp/cache").write_bytes(b"original cache")
    output = (
        root / ".superpowers/sdd/2026-10-14-resolv-three-branch-integration/snapshot"
    )
    commands = []

    def runner(command, *, cwd, **kwargs):
        commands.append(command)
        assert not (Path(cwd) / "src/uncommitted.ts").exists()
        assert (Path(cwd) / "src/main.ts").read_text() == 'import "./uncommitted.ts"'
        (Path(cwd) / "node_modules/.tmp/cache").parent.mkdir(exist_ok=True)
        (Path(cwd) / "node_modules/.tmp/cache").write_text("generated cache")
        return subprocess.CompletedProcess(
            command, 1 if command[-1] == "build" else 0, "missing uncommitted import"
        )

    report = verify("HEAD", output, root=root, runner=runner)
    assert report["passed"] is False
    assert commands == [
        ["npm", "test"],
        ["npm", "run", "typecheck"],
        ["npm", "run", "build"],
        ["npm", "run", "lint"],
    ]
    assert (root / "frontend/node_modules/.tmp/cache").read_bytes() == b"original cache"
    assert (root / "frontend/src/uncommitted.ts").read_text() == "user work"


def test_snapshot_rejects_user_output_directory_before_writing(tmp_path):
    try:
        verify = importlib.import_module(
            "scripts.verify_frontend_snapshot"
        ).verify_snapshot
    except ModuleNotFoundError:
        pytest.fail("commit snapshot verification missing")
    with pytest.raises(ValueError):
        verify("HEAD", tmp_path / "frontend", root=tmp_path)
