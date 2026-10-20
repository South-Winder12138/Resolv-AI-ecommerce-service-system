"""Verify Python and frontend from one explicit committed refund candidate."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import subprocess
from hashlib import sha256

from scripts.verify_frontend_snapshot import verify_snapshot as verify_frontend
from scripts.verify_hybrid_snapshot import verify_snapshot as verify_python

WORKSPACE = ".superpowers/sdd/2026-10-16-resolv-refund-flow"


def verify_snapshot(
    revision,
    output_dir,
    *,
    root=None,
    python_runner=subprocess.run,
    frontend_runner=subprocess.run,
):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    output_dir = Path(output_dir).resolve()
    workspace = root / WORKSPACE
    if (
        output_dir == workspace
        or not output_dir.is_relative_to(workspace)
        or output_dir.exists()
    ):
        raise ValueError("Refund snapshot requires a new task directory")
    # Resolve once; branch movements cannot select different revisions per export.
    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            "--verify",
            "--end-of-options",
            revision + "^{commit}",
        ],
        text=True,
    ).strip()
    output_dir.mkdir(parents=True)
    python = verify_python(
        commit,
        output_dir / "python",
        root=root,
        runner=python_runner,
        workspace_name="refund",
    )
    frontend = verify_frontend(
        commit,
        output_dir / "frontend",
        root=root,
        runner=frontend_runner,
        workspace_name="refund",
    )
    source = output_dir / "python/source"
    fingerprints = {
        str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest()
        for folder in ("src", "scripts", "tests", "migrations")
        for p in (source / folder).rglob("*.py")
    }
    for name in ("pyproject.toml", "uv.lock", "frontend/package-lock.json"):
        if (source / name).exists():
            fingerprints[name] = sha256((source / name).read_bytes()).hexdigest()
    python.update(gate="snapshot_python", source_fingerprints=fingerprints)
    frontend.update(
        gate="snapshot_frontend",
        source_fingerprints={
            str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest()
            for p in (source / "frontend").rglob("*")
            if p.is_file()
        },
    )
    report = {
        "passed": python["passed"]
        and frontend["passed"]
        and python["revision"] == frontend["revision"] == commit
        and python["tree"] == frontend["tree"],
        "candidate_commit": commit,
        "tree": python["tree"],
        "python": python,
        "frontend": frontend,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_snapshot(args.revision, args.output_dir)
    except Exception as exc:  # noqa: BLE001 -- safe CLI metadata only
        raise SystemExit("Refund snapshot failed: " + type(exc).__name__) from None
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "candidate_commit": report["candidate_commit"],
                "tree": report["tree"],
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
