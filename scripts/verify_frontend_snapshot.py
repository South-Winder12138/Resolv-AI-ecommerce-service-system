"""Build only Git-committed frontend content, using local dependencies without installation."""

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

WORKSPACE = ".superpowers/sdd/2026-10-14-resolv-three-branch-integration"
WORKSPACES = {
    "logistics": ".superpowers/sdd/2026-10-20-resolv-logistics-mcp",
    "refund": ".superpowers/sdd/2026-10-16-resolv-refund-flow",
    "three_branch": WORKSPACE,
    "knowledge": ".superpowers/sdd/2026-10-15-resolv-knowledge-branch",
}


def verify_snapshot(
    revision,
    output_dir,
    *,
    root=None,
    runner=subprocess.run,
    workspace_name="three_branch",
):
    if workspace_name not in WORKSPACES:
        raise ValueError("Unknown snapshot workspace")
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    output_dir = Path(output_dir).resolve()
    workspace = (root / WORKSPACES[workspace_name]).resolve()
    if (
        not output_dir.is_relative_to(workspace)
        or output_dir == workspace
        or output_dir.exists()
    ):
        raise ValueError(
            "snapshot output must be a new directory inside this task's workspace"
        )
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=root, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", sha + "^{tree}"], cwd=root, text=True
    ).strip()
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", sha, "frontend"], cwd=root
    )
    output_dir.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(output_dir, filter="data")
    frontend = output_dir / "frontend"
    modules = frontend / "node_modules"
    modules.mkdir()
    # Reuse packages but keep build caches local to the exported snapshot.
    for item in (root / "frontend/node_modules").iterdir():
        if item.name not in {".tmp", ".vite", ".vite-temp", ".cache"}:
            (modules / item.name).symlink_to(item, target_is_directory=item.is_dir())
    steps = []
    for command in (
        ["npm", "test"],
        ["npm", "run", "typecheck"],
        ["npm", "run", "build"],
        ["npm", "run", "lint"],
    ):
        result = runner(command, cwd=frontend, capture_output=True, text=True)
        log = output_dir / (command[-1] + ".log")
        log.write_text(result.stdout + (result.stderr or ""))
        steps.append(
            {"command": command, "exit_code": result.returncode, "log": str(log)}
        )
    report = {
        "revision": sha,
        "tree": tree,
        "checks": {step["command"][-1]: step["exit_code"] for step in steps},
        "passed": all(step["exit_code"] == 0 for step in steps),
        "steps": steps,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = verify_snapshot(args.revision, args.output_dir)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
