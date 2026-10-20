"""Export a committed tree and run source-isolated, socket-blocked offline checks."""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from resolv.schemas.hybrid_knowledge import HybridRetrievalError

WORKSPACE = ".superpowers/sdd/2026-10-14-resolv-hybrid-retrieval"
WORKSPACES = {
    "logistics": ".superpowers/sdd/2026-10-20-resolv-logistics-mcp",
    "refund": ".superpowers/sdd/2026-10-16-resolv-refund-flow",
    "hybrid": WORKSPACE,
    "knowledge": ".superpowers/sdd/2026-10-15-resolv-knowledge-branch",
}
GUARD = r"""
import json, os, socket, sys
from pathlib import Path
source=Path(os.environ['MH_HYBRID_SNAPSHOT_SOURCE']).resolve()
original=Path(os.environ['MH_HYBRID_SNAPSHOT_ORIGINAL']).resolve()
runtime=Path(sys.prefix).resolve()
def forbidden_path(value):
    if not value:return False
    p=Path(value).resolve()
    if p==runtime or runtime in p.parents:return False
    return p==original or original in p.parents
sys.path[:]=[str(source/'src'),str(source)]+[p for p in sys.path if not forbidden_path(p) and p not in (str(source/'src'),str(source))]
def record(method, family=None):
    with open(os.environ['MH_HYBRID_NETWORK_LOG'],'a') as f:
        f.write(json.dumps({'method':method,'family':family})+'\n')
    raise RuntimeError('hybrid snapshot network blocked')
def wrap_connect(method):
    original_method=getattr(socket.socket,method)
    def guarded(self,*a,**kw):
        if self.family in (socket.AF_INET,socket.AF_INET6):record(method,int(self.family))
        return original_method(self,*a,**kw)
    return guarded
socket.socket.connect=wrap_connect('connect')
socket.socket.connect_ex=wrap_connect('connect_ex')
for name in ('getaddrinfo','gethostbyname','gethostbyname_ex','getnameinfo','gethostbyaddr'):
    def denied(*a,_name=name,**kw):record(_name)
    setattr(socket,name,denied)
os.environ['PYTHON_DOTENV_DISABLED']='1'
"""
IMPORT_PROBE = r"""
import os
from pathlib import Path
import resolv
root=Path(os.environ['MH_HYBRID_SNAPSHOT_SOURCE']).resolve()/'src'
assert Path(resolv.__file__).resolve().is_relative_to(root), 'source import fallback'
print(str(Path(resolv.__file__).resolve()))
"""
NEGATIVE_PROBE = r"""
import socket
try:
    socket.socket(socket.AF_INET).connect(('127.0.0.1',9))
except RuntimeError as e:
    assert str(e)=='hybrid snapshot network blocked'
else:
    raise AssertionError('guard did not block TCP')
"""
DNS_PROBE = r"""
try:
    socket.getaddrinfo('snapshot.invalid', 443)
except RuntimeError as e:
    assert str(e)=='hybrid snapshot network blocked'
else:
    raise AssertionError('guard did not block DNS')
"""


def _git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def verify_snapshot(
    revision: str,
    output_dir: Path,
    *,
    root: Path | None = None,
    runner=subprocess.run,
    workspace_name="hybrid",
) -> dict:
    if workspace_name not in WORKSPACES:
        raise ValueError("Unknown snapshot workspace")
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    output_dir = Path(output_dir).resolve()
    workspace = (root / WORKSPACES[workspace_name]).resolve()
    if (
        not output_dir.is_relative_to(workspace)
        or output_dir == workspace
        or output_dir.exists()
    ):
        raise HybridRetrievalError("hybrid_snapshot_output_invalid")
    commit = _git(
        root, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"
    )
    tree = _git(root, "rev-parse", commit + "^{tree}")
    output_dir.mkdir(parents=True)
    source = output_dir / "source"
    source.mkdir()
    archive = output_dir / "commit.tar"
    _git(root, "archive", "--format=tar", "--output=" + str(archive), commit)
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    if workspace_name == "logistics":
        (source / ".logistics-source.json").write_text(json.dumps({
            "revision": commit, "tree": tree, "resource_root": str(workspace),
        }) + "\n")
    guard = output_dir / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(GUARD)
    log = output_dir / "network.jsonl"
    log.touch()
    negative = output_dir / "negative-network.jsonl"
    negative.touch()
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(guard) + os.pathsep + str(source / "src") + os.pathsep + str(source)
    )
    env.update(
        MH_HYBRID_SNAPSHOT_SOURCE=str(source),
        MH_HYBRID_SNAPSHOT_ORIGINAL=str(root),
        MH_HYBRID_NETWORK_LOG=str(log),
    )
    if workspace_name in {"knowledge", "refund", "logistics"}:
        env["MH_SNAPSHOT_FRONTEND_DEPENDENCIES"] = str(root / "frontend/node_modules")
    checks = {}
    outputs = {}

    def run(phase, cmd, *, negative_log=False):
        current = env.copy()
        current["MH_HYBRID_SNAPSHOT_PHASE"] = phase
        if negative_log:
            current["MH_HYBRID_NETWORK_LOG"] = str(negative)
        result = runner(
            cmd, cwd=source, env=current, capture_output=True, text=True, check=False
        )
        checks[phase] = result.returncode
        outputs[phase] = result.stdout + "\n" + result.stderr
        (output_dir / (phase + ".log")).write_text(outputs[phase])
        return result

    probe = NEGATIVE_PROBE + (DNS_PROBE if workspace_name in {"knowledge", "refund", "logistics"} else "")
    run("negative_probe", [sys.executable, "-c", probe], negative_log=True)
    run("source_import", [sys.executable, "-c", IMPORT_PROBE])
    run("pytest", [sys.executable, "-m", "pytest", "-q", "-ra"])
    lint_roots = (
        "src",
        "scripts",
        "tests",
        "migrations" if workspace_name in {"knowledge", "refund", "logistics"} else "migrations/knowledge",
    )
    lint_cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--no-respect-gitignore",
        "--no-cache",
    ]
    discovered = run("ruff_files", [*lint_cmd, "--show-files", *lint_roots])
    expected_files = {
        p.resolve() for name in lint_roots for p in (source / name).rglob("*.py")
    }
    lint_files = {
        Path(p).resolve() for p in discovered.stdout.splitlines() if p.strip()
    }
    lint_inputs_verified = bool(expected_files) and expected_files == lint_files
    run("ruff", [*lint_cmd, *lint_roots])
    attempts = [x for x in log.read_text().splitlines() if x]
    probes = [x for x in negative.read_text().splitlines() if x]
    report = {
        "passed": all(code == 0 for code in checks.values())
        and lint_inputs_verified
        and not attempts
        and len(probes) == (2 if workspace_name in {"knowledge", "refund", "logistics"} else 1),
        "revision": commit,
        "commit": commit,
        "tree": tree,
        "checks": checks,
        "source_import_verified": checks["source_import"] == 0,
        "lint_inputs_verified": lint_inputs_verified,
        "lint_file_count": len(lint_files),
        "expected_lint_file_count": len(expected_files),
        "ruff_discovered": len(lint_files),
        "expected_python_files": len(expected_files),
        "network_attempts": len(attempts),
        "negative_probe_attempts": len(probes),
        "python_executable": sys.executable,
        "pytest_summary": outputs["pytest"].strip().splitlines()[-1]
        if outputs["pytest"].strip()
        else "",
        "guard_scope": "Python socket AF_INET/AF_INET6 connect/connect_ex and DNS, inherited by Python children",
        "exported_source": str(source),
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
    except Exception as exc:  # noqa: BLE001 -- safe command error metadata only
        raise SystemExit(
            "Hybrid snapshot failed: " + getattr(exc, "code", type(exc).__name__)
        ) from None
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
