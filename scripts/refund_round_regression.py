"""One explicitly requested F1 regression after the round-control fix."""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import refund_gate_support as s

PUBLIC = "docs/superpowers/acceptance/2026-10-16-refund-flow"
APPROVAL = PUBLIC + "/round-control-authorization.json"
PRIOR = PUBLIC + "/R2-C3-retry-1.json"
REQUEST = "好，现在完善轮次提示和最后一轮结束策略，然后继续测试"
TAG = "round-control"


def require(value):
    if not value:
        raise ValueError("Unapproved, changed or exhausted F1 round-control regression")


def policy(scope):
    require(
        scope
        in {
            TAG,
            "output-repair",
            "output-completeness",
            "context-identity",
            "chain-validation",
        }
    )
    if scope == "chain-validation":
        return {
            "attempt": 10,
            "calls": 61,
            "attempts": 17,
            "searches": 12,
            "request": "对这条链路进行验证，看是否还会出现异常",
            "prior": PUBLIC + "/R2-native-return.json",
            "key": "chain_validation_regression",
            "approval": PUBLIC + "/chain-validation-authorization.json",
        }
    if scope == TAG:
        return {
            "attempt": 5,
            "calls": 27,
            "attempts": 12,
            "searches": 8,
            "request": REQUEST,
            "prior": PRIOR,
            "key": "round_control_regression",
            "approval": APPROVAL,
        }
    if scope == "output-completeness":
        return {
            "attempt": 7,
            "calls": 44,
            "attempts": 14,
            "searches": 10,
            "request": "现在开始进行真实模型验证，进行测试",
            "prior": PUBLIC + "/R2-output-repair-F1.json",
            "key": "output_completeness_regression",
            "approval": PUBLIC + "/output-completeness-authorization.json",
        }
    if scope == "context-identity":
        return {
            "attempt": 8,
            "calls": 53,
            "attempts": 15,
            "searches": 11,
            "request": "好，继续下一步",
            "prior": PUBLIC + "/R2-output-completeness-F1.json",
            "key": "context_identity_regression",
            "approval": PUBLIC + "/context-identity-authorization.json",
        }
    return {
        "attempt": 6,
        "calls": 35,
        "attempts": 13,
        "searches": 9,
        "request": "好，按你这个顺序，进行修复",
        "prior": PUBLIC + "/R2-round-control-F1.json",
        "key": "output_repair_regression",
        "approval": PUBLIC + "/output-repair-authorization.json",
    }


def fingerprints():
    from scripts.refund_semantic_continuation import fingerprints as previous

    _, names = previous()
    return s.committed_sources(tuple(names))


def read_approval(commit, *, scope=TAG):
    p = policy(scope)
    require(re.fullmatch("[0-9a-f]{40}", commit or ""))
    try:
        return json.loads(
            subprocess.check_output(
                ["git", "show", commit + ":" + p["approval"]],
                cwd=s.ROOT,
                stderr=subprocess.DEVNULL,
            )
        )
    except (ValueError, subprocess.CalledProcessError):
        raise ValueError("Missing committed direct regression instruction") from None


def _save(manifest, obj, *, scope=TAG):
    g = obj[policy(scope)["key"]]
    g["grant_hash"] = s.canonical_hash(
        {k: v for k, v in g.items() if k != "grant_hash"}
    )
    s.save_json(manifest, obj)


def _active(obj, source_hash, *, started=True, scope=TAG):
    p = policy(scope)
    g = obj.get(p["key"], {})
    require(
        g.get("owner") == obj["owner"]
        and g.get("source_hash") == source_hash
        and g.get("grant_hash")
        == s.canonical_hash({k: v for k, v in g.items() if k != "grant_hash"})
        and g.get("status") == ("started" if started else "claimed")
    )
    if started:
        row = obj["semantic_attempts"].get(f"F1:{p['attempt']}", {})
        require(
            row.get("source_hash") == source_hash and row.get("continuation") == scope
        )
    return g


def claim_regression(manifest, approved, *, approval_commit, source_hash, scope=TAG):
    p = policy(scope)
    require(re.fullmatch("[0-9a-f]{40}", approval_commit or ""))
    require(re.fullmatch("[0-9a-f]{64}", source_hash or ""))
    require(
        approved.get("authorization_kind") == "direct_user_request"
        and approved.get("approval_text") == p["request"]
        and approved.get("source_hash") == source_hash
    )
    require(
        tuple(
            approved.get(k)
            for k in (
                "case_id",
                "attempt",
                "max_additional_supplier_slots",
                "max_additional_retrievals",
                "max_semantic_attempts",
            )
        )
        == ("F1", p["attempt"], 9, 1, 1)
    )
    raw = (s.ROOT / p["prior"]).read_bytes()
    require(sha256(raw).hexdigest() == approved.get("prior_report_sha256"))
    prior = json.loads(raw)
    preceding = (
        prior.get("probe", {}) if scope == "chain-validation" else prior["cases"][0]
    )
    require(
        preceding.get("case_id") == "F1" and preceding.get("machine_passed") is False
    )
    if scope == "chain-validation":
        from scripts.refund_acceptance_v2 import CASES_SHA256

        require(approved.get("cases_sha256") == CASES_SHA256)
    from scripts.refund_semantic_continuation import grant

    with s.manifest_lock(manifest) as obj:
        require(p["key"] not in obj)
        old = grant(obj, continuation="C3")
        require(old["halted"] == "F1" and old["source_hash"] != source_hash)
        previous_keys = (
            (
                "round_control_regression",
                "output_repair_regression",
                "output_completeness_regression",
                "context_identity_regression",
                "native_return_repair",
            )
            if scope == "chain-validation"
            else (
                "round_control_regression",
                "output_repair_regression",
                "output_completeness_regression",
            )
            if scope == "context-identity"
            else ("round_control_regression", "output_repair_regression")
            if scope == "output-completeness"
            else ("round_control_regression",)
            if scope == "output-repair"
            else ()
        )
        for key in previous_keys:
            previous = obj.get(key, {})
            require(
                previous.get("status") == "halted"
                and previous.get("source_hash") != source_hash
            )
            require(
                previous.get("grant_hash")
                == s.canonical_hash(
                    {k: v for k, v in previous.items() if k != "grant_hash"}
                )
            )
        require(approved.get("authorized_owner") == obj["owner"])
        calls = s.budget_ledger(manifest)
        require(
            len(calls) == p["calls"]
            and approved.get("baseline_ledger_hash") == s.canonical_hash(calls)
        )
        require(
            len(obj["semantic_attempts"]) == p["attempts"]
            and approved.get("baseline_attempts_hash")
            == s.canonical_hash(obj["semantic_attempts"])
        )
        require(
            len(obj["retrievals"]) == p["searches"]
            and approved.get("baseline_retrievals_hash")
            == s.canonical_hash(obj["retrievals"])
        )
        obj[p["key"]] = {
            "owner": obj["owner"],
            "source_hash": source_hash,
            "approval_commit": approval_commit,
            "approval_hash": s.canonical_hash(approved),
            "prior_report_sha256": approved["prior_report_sha256"],
            "status": "claimed",
        }
        _save(manifest, obj, scope=scope)


def reserve_case(manifest, *, case_id, attempt, source_hash, scope=TAG):
    p = policy(scope)
    key = f"F1:{p['attempt']}"
    require(case_id == "F1" and type(attempt) is int and attempt == p["attempt"])
    with s.manifest_lock(manifest) as obj:
        g = _active(obj, source_hash, started=False, scope=scope)
        require(
            key not in obj["semantic_attempts"] and len(obj["semantic_attempts"]) < 19
        )
        obj["semantic_attempts"][key] = {
            "source_hash": source_hash,
            "status": "started",
            "continuation": scope,
        }
        g["status"] = "started"
        _save(manifest, obj, scope=scope)


def reserve_call(manifest, *, purpose, case_id, attempt, source_hash, scope=TAG):
    p = policy(scope)
    require(case_id == "F1" and type(attempt) is int and attempt == p["attempt"])
    require(purpose in {"resolution", "intents", "refund_query", "refund_decision"})
    with s.manifest_lock(manifest) as obj:
        _active(obj, source_hash, scope=scope)
        calls = s.budget_ledger(manifest)
        totals = Counter(c["purpose"] for c in calls)
        require(
            len(calls) < 96
            and totals[purpose]
            < (11 if scope == TAG and purpose == "resolution" else s.LIMITS[purpose])
        )
        used = [c for c in calls if c.get("continuation") == scope]
        require(
            all(
                c["case_id"] == "F1"
                and c["attempt"] == p["attempt"]
                and c["source_hash"] == source_hash
                for c in used
            )
        )
        require(
            len(used) < 9
            and sum(c["purpose"] == purpose for c in used)
            < (6 if purpose == "refund_decision" else 1)
        )
        entry = {
            "invocation_id": s.uuid4().hex,
            "purpose": purpose,
            "case_id": case_id,
            "attempt": attempt,
            "source_hash": source_hash,
            "outcome": "reserved",
            "input_tokens": None,
            "output_tokens": None,
            "continuation": scope,
        }
        calls.append(entry)
        s._save_ledger(manifest, calls, obj)
        return s.InvocationReservation(Path(manifest).resolve(), entry["invocation_id"])


def reserve_retrieval(manifest, *, label, source_hash, scope=TAG):
    require(label == f"F1:{policy(scope)['attempt']}")
    with s.manifest_lock(manifest) as obj:
        _active(obj, source_hash, scope=scope)
        used = obj["retrievals"]
        require(
            label not in used
            and len(used) < 16
            and not any(v.get("continuation") == scope for v in used.values())
        )
        used[label] = {
            "source_hash": source_hash,
            "outcome": "reserved",
            "continuation": scope,
        }
        s.save_json(manifest, obj)


def finish(manifest, *, source_hash, passed, scope=TAG):
    require(type(passed) is bool)
    with s.manifest_lock(manifest) as obj:
        g = obj.get(policy(scope)["key"], {})
        _active(obj, source_hash, started=g.get("status") == "started", scope=scope)
        g["status"] = "completed" if passed else "halted"
        if scope == "chain-validation":
            row = obj["semantic_attempts"].get("F1:10")
            if row is not None:
                row["status"] = "passed" if passed else "failed"
        _save(manifest, obj, scope=scope)


def main(argv=None):
    from scripts.smoke_refund_flow import load_cases, owned_selection, semantic_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--approval-commit", required=True)
    parser.add_argument(
        "--scope",
        choices=(
            TAG,
            "output-repair",
            "output-completeness",
            "context-identity",
            "chain-validation",
        ),
        default=TAG,
    )
    args = parser.parse_args(argv)
    scope = args.scope
    if scope == "chain-validation":
        from scripts.refund_acceptance_v2 import load_cases

    p = policy(scope)
    skw = {} if scope == TAG else {"scope": scope}
    public = s.ROOT / PUBLIC
    output = public / ("R2-" + scope + "-F1.json")
    require(not output.exists())
    commit, sources = fingerprints()
    source_hash = s.canonical_hash(sources)
    approved = read_approval(args.approval_commit, **skw)
    target = s.new_report_dir(args.ownership, args.report_dir)
    claim_regression(
        args.ownership,
        approved,
        approval_commit=args.approval_commit,
        source_hash=source_hash,
        **skw,
    )
    report = {
        "gate": "R2-" + scope + "-F1",
        "candidate_commit": commit,
        "source_fingerprints": sources,
        "approval_commit": args.approval_commit,
        "prior_report_sha256": approved["prior_report_sha256"],
        "passed": False,
        "machine_passed": False,
        "manual_review_required": True,
        "formal_preserved": False,
        "formal_migration_eligible": False,
        "r1_prerequisite_deferred": True,
        "cases": [],
    }
    diagnostic_sink = None
    if scope == "chain-validation":
        report["expectation_revision"] = 2
    if scope in {
        "output-repair",
        "output-completeness",
        "context-identity",
        "chain-validation",
    }:
        from scripts.refund_failure_storage import OwnedRefundDiagnostics

        diagnostic_sink = OwnedRefundDiagnostics(args.ownership, target)
    before = None
    try:
        before = s.snapshot_formal_resources()
        s.save_json(target / "preservation-before.json", before)
        for name in ("R2-prepare-continuation.json", "R2-retrieval.json"):
            require(json.loads((public / name).read_text()).get("passed") is True)
        with s.owned_refund_resources(args.ownership) as resources:
            owned_selection(resources)
            result = semantic_case(
                resources,
                load_cases()["cases"][0],
                args.ownership,
                source_hash,
                target,
                attempt=p["attempt"],
                continuation=scope,
                **({"diagnostic_hook": diagnostic_sink} if diagnostic_sink else {}),
            )
            report["cases"] = [result]
            report["machine_passed"] = result.get("machine_passed") is True
    except Exception as exc:  # noqa: BLE001 -- retain no credentials or private errors
        report.update(
            error_type=type(exc).__name__,
            error_code=getattr(exc, "code", "regression_failed"),
        )
    if before is not None:
        try:
            after = s.snapshot_formal_resources(layout=before)
            s.save_json(target / "preservation-after.json", after)
            report["formal_preserved"] = before == after
        except Exception as exc:  # noqa: BLE001 -- unknown preservation fails the gate
            report["preservation_error_type"] = type(exc).__name__
    finish(
        args.ownership,
        source_hash=source_hash,
        passed=report["machine_passed"] and report["formal_preserved"],
        **skw,
    )
    if diagnostic_sink is not None:
        report["failure_diagnostics"] = diagnostic_sink.entries
    report["ledger"] = s.budget_ledger(args.ownership)
    report["resource_identity"] = s.validate_manifest(args.ownership)
    report["prior_report_preserved"] = (
        sha256((s.ROOT / p["prior"]).read_bytes()).hexdigest()
        == approved["prior_report_sha256"]
    )
    s.save_json(target / "report.json", report)
    require(not output.exists())
    s.save_json(output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "gate",
                    "machine_passed",
                    "formal_preserved",
                    "prior_report_preserved",
                )
            }
        )
    )
    return (
        0
        if report["machine_passed"]
        and report["formal_preserved"]
        and report["prior_report_preserved"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
