"""One explicitly authorized retry of C3's zero-work startup failure."""

import argparse
import json
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from time import monotonic

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import refund_gate_support as s

PUBLIC = "docs/superpowers/acceptance/2026-10-16-refund-flow"
APPROVAL = PUBLIC + "/C3-startup-retry-1-authorization.json"
ORIGINAL_APPROVAL = "880abcdc9f764b8a33d6e91c00b02031dff2376b"


def require(value):
    if not value:
        raise ValueError("Unapproved, changed or consumed C3 startup retry")


def read_approval(commit):
    require(re.fullmatch("[0-9a-f]{40}", commit or ""))
    try:
        return json.loads(
            subprocess.check_output(
                ["git", "show", commit + ":" + APPROVAL],
                cwd=s.ROOT,
                stderr=subprocess.DEVNULL,
            )
        )
    except (subprocess.CalledProcessError, ValueError):
        raise ValueError("Missing committed C3 retry approval") from None


def claim_retry(manifest, approved, *, approval_commit, source_hash):
    require(re.fullmatch("[0-9a-f]{40}", approval_commit or ""))
    prior_path = s.ROOT / PUBLIC / "R2-C3.json"
    raw = prior_path.read_bytes()
    prior = json.loads(raw)
    require(
        approved.get("approved") is True
        and approved.get("approval_text") == "重新尝试，继续"
    )
    require(
        approved.get("source_hash") == source_hash
        and re.fullmatch("[0-9a-f]{64}", source_hash or "")
    )
    require(approved.get("prior_report_sha256") == sha256(raw).hexdigest())
    require(approved.get("original_approval_commit") == ORIGINAL_APPROVAL)
    require(
        tuple(
            approved.get(k)
            for k in (
                "max_additional_supplier_slots",
                "max_semantic_attempts",
                "max_additional_retrievals",
            )
        )
        == (72, 8, 8)
    )
    require(
        prior.get("gate") == "R2-C3"
        and prior.get("error_type") == "VectorStoreError"
        and prior.get("passed") is False
    )
    require(
        [c.get("case_id") for c in prior.get("cases", [])]
        == [f"F{i}" for i in range(1, 9)]
        and all(c.get("status") == "not_executed" for c in prior["cases"])
    )
    with s.manifest_lock(manifest) as obj:
        require(
            "c3_startup_retry" not in obj and "C3" not in obj.get("continuations", {})
        )
        require(approved.get("authorized_owner") == obj["owner"])
        require(
            all(
                prior.get("resource_identity", {}).get(k) == obj[k]
                for k in (
                    "owner",
                    "server_fingerprint",
                    "conversation_database",
                    "knowledge_database",
                    "collection",
                    "business_db",
                )
            )
        )
        calls = s.budget_ledger(manifest)
        require(
            len(calls) == 18
            and approved.get("baseline_ledger_hash")
            == s.canonical_hash(calls)
            == s.canonical_hash(prior.get("ledger"))
        )
        require(
            len(obj["semantic_attempts"]) == 11
            and approved.get("baseline_attempts_hash")
            == s.canonical_hash(obj["semantic_attempts"])
        )
        require(
            len(obj["retrievals"]) == 7
            and approved.get("baseline_retrievals_hash")
            == s.canonical_hash(obj["retrievals"])
        )
        obj["c3_startup_retry"] = {
            "approval_commit": approval_commit,
            "approval_hash": s.canonical_hash(approved),
            "source_hash": source_hash,
            "prior_report_sha256": sha256(raw).hexdigest(),
        }
        s.save_json(manifest, obj)


def main(argv=None):
    from scripts.refund_semantic_continuation import fingerprints
    from scripts.smoke_refund_flow import load_cases, run_refund_gate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--approval-retry-commit", required=True)
    args = parser.parse_args(argv)
    public = s.ROOT / PUBLIC
    paths = [
        public / n
        for n in (
            "R2-C3-retry-1.json",
            "real-invocation-ledger-C3-retry-1.json",
            "preservation-R2-C3-retry-1.json",
        )
    ]
    if any(p.exists() for p in paths):
        raise ValueError("Refusing to overwrite immutable C3 retry evidence")
    commit, sources = fingerprints()
    approved = read_approval(args.approval_retry_commit)
    target = s.new_report_dir(args.ownership, args.report_dir)
    claim_retry(
        args.ownership,
        approved,
        approval_commit=args.approval_retry_commit,
        source_hash=s.canonical_hash(sources),
    )
    report = {
        "gate": "R2-C3-retry-1",
        "passed": False,
        "candidate_commit": commit,
        "source_fingerprints": sources,
        "prior_report_sha256": approved["prior_report_sha256"],
        "retry_approval_commit": args.approval_retry_commit,
        "r1_prerequisite_deferred": True,
        "formal_migration_eligible": False,
        "formal_preserved": False,
    }
    before = None
    begin = monotonic()
    try:
        before = s.snapshot_formal_resources()
        s.save_json(target / "preservation-before.json", before)
        observed = run_refund_gate(
            ownership=args.ownership,
            case_ids=tuple(f"F{i}" for i in range(1, 9)),
            report_dir=target / "cases",
            prepare_report=public / "R2-prepare-continuation.json",
            continuation="C3",
            approval_commit=ORIGINAL_APPROVAL,
        )
        report.update(observed, gate="R2-C3-retry-1")
    except Exception as exc:  # noqa: BLE001 — credentials must stay out of reports
        report.update(
            error_code=getattr(exc, "code", "refund_gate_failed"),
            error_type=type(exc).__name__,
            passed=False,
        )
    if before is not None:
        try:
            after = s.snapshot_formal_resources(layout=before)
            s.save_json(target / "preservation-after.json", after)
            report["formal_preserved"] = before == after
        except Exception as exc:  # noqa: BLE001 — preservation absence is failure
            report.update(
                formal_preserved=False, preservation_error_type=type(exc).__name__
            )
    report.setdefault(
        "cases",
        [
            {
                "case_id": c["case_id"],
                "question": c["question"],
                "expected": c["expected"],
                "status": "not_executed",
                "passed": False,
            }
            for c in load_cases()["cases"]
        ],
    )
    report.update(
        elapsed_seconds=monotonic() - begin,
        resource_identity=s.validate_manifest(args.ownership),
        ledger=s.budget_ledger(args.ownership),
        prior_report_preserved=sha256((public / "R2-C3.json").read_bytes()).hexdigest()
        == approved["prior_report_sha256"],
    )
    s.save_json(target / "report.json", report)
    require(not any(p.exists() for p in paths))
    s.save_json(paths[0], report)
    s.save_json(paths[1], report["ledger"])
    s.save_json(
        paths[2], {"before": before, "formal_preserved": report["formal_preserved"]}
    )
    print(
        json.dumps(
            {
                "gate": report["gate"],
                "machine_passed": report.get("machine_passed", False),
                "formal_preserved": report["formal_preserved"],
                "report": str(paths[0]),
            }
        )
    )
    return (
        0
        if report.get("machine_passed")
        and report["formal_preserved"]
        and report["prior_report_preserved"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
