"""One immutable candidate for both offline Python and frontend verification."""

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

WORKSPACE = ".superpowers/sdd/2026-10-20-resolv-logistics-mcp"


def verify_logistics_snapshot(revision, output_dir, *, root=None, python_runner=subprocess.run, frontend_runner=subprocess.run):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    output_dir = Path(output_dir).absolute()
    workspace = root / WORKSPACE
    if output_dir != output_dir.resolve() or not output_dir.is_relative_to(workspace) or output_dir == workspace or output_dir.exists():
        raise ValueError("Logistics snapshot requires a new owned output directory")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"], text=True).strip()
    output_dir.mkdir(parents=True)
    python = verify_python(commit, output_dir / "python", root=root, runner=python_runner, workspace_name="logistics")
    frontend = verify_frontend(commit, output_dir / "frontend", root=root, runner=frontend_runner, workspace_name="logistics")
    source = output_dir / "python/source"
    fingerprints = {
        str(path.relative_to(source)): sha256(path.read_bytes()).hexdigest()
        for folder in ("src", "scripts", "tests", "migrations", "logistics", "frontend")
        for path in (source / folder).rglob("*") if path.is_file()
    }
    for name in ("pyproject.toml", "uv.lock"):
        if (source / name).is_file():
            fingerprints[name] = sha256((source / name).read_bytes()).hexdigest()
    report = {
        "passed": python["passed"] and frontend["passed"] and python["revision"] == frontend["revision"] == commit and python["tree"] == frontend["tree"],
        "candidate_commit": commit, "tree": python["tree"], "exported_source": str(source),
        "python": python, "frontend": frontend, "source_fingerprints": fingerprints,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = verify_logistics_snapshot(args.revision, args.output_dir)
    print(json.dumps({k: report[k] for k in ("passed", "candidate_commit", "exported_source")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
