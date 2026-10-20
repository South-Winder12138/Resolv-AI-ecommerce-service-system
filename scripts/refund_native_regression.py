"""Two explicitly approved native-return cases; never reopen an older allocation."""

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
from scripts.refund_acceptance_v2 import CASES_SHA256

TAG = "native-return"
KEY = "native_return_repair"
PUBLIC = "docs/superpowers/acceptance/2026-10-16-refund-flow"
PRIOR = PUBLIC + "/R2-context-identity-F1.json"
APPROVAL = PUBLIC + "/native-return-authorization.json"


def require(value):
    if not value:
        raise ValueError("Unapproved, changed or exhausted native-return repair")


def _save(manifest, obj):
    g = obj[KEY]
    g["grant_hash"] = s.canonical_hash(
        {k: v for k, v in g.items() if k != "grant_hash"}
    )
    s.save_json(manifest, obj)


def _active(obj, source_hash, status=None):
    g = obj.get(KEY, {})
    require(
        g.get("owner") == obj["owner"]
        and g.get("source_hash") == source_hash
        and g.get("grant_hash")
        == s.canonical_hash({k: v for k, v in g.items() if k != "grant_hash"})
    )
    if status:
        require(g.get("status") == status)
    return g


def claim_grant(manifest, approved, *, approval_commit, source_hash):
    require(re.fullmatch("[0-9a-f]{40}", approval_commit or ""))
    require(re.fullmatch("[0-9a-f]{64}", source_hash or ""))
    require(
        approved.get("authorization_kind") == "explicit_user_reply"
        and approved.get("approval_text") == "按此方案修复并验证（推荐）"
        and approved.get("expectation_approval_text")
        == "修正 F1 验收，另加明确口径的隔离正向案例（推荐）"
    )
    require(
        [
            approved.get(k)
            for k in (
                "max_additional_supplier_slots",
                "max_semantic_attempts",
                "max_additional_retrievals",
                "probe_attempt",
                "integration_attempt",
            )
        ]
        == [10, 2, 1, 9, 10]
    )
    require(
        approved.get("source_hash") == source_hash
        and approved.get("cases_sha256") == CASES_SHA256
    )
    raw = (s.ROOT / PRIOR).read_bytes()
    require(sha256(raw).hexdigest() == approved.get("prior_report_sha256"))
    require(json.loads(raw)["cases"][0]["machine_passed"] is False)
    with s.manifest_lock(manifest) as obj:
        require(KEY not in obj and approved.get("authorized_owner") == obj["owner"])
        old = obj.get("context_identity_regression", {})
        require(old.get("status") == "halted" and old.get("source_hash") != source_hash)
        require(
            old.get("grant_hash")
            == s.canonical_hash({k: v for k, v in old.items() if k != "grant_hash"})
        )
        calls = s.budget_ledger(manifest)
        require(
            len(calls) == 60
            and len(obj["semantic_attempts"]) == 16
            and len(obj["retrievals"]) == 12
        )
        require(
            approved.get("baseline_ledger_hash") == s.canonical_hash(calls)
            and approved.get("baseline_attempts_hash")
            == s.canonical_hash(obj["semantic_attempts"])
            and approved.get("baseline_retrievals_hash")
            == s.canonical_hash(obj["retrievals"])
        )
        obj[KEY] = {
            "owner": obj["owner"],
            "source_hash": source_hash,
            "approval_commit": approval_commit,
            "approval_hash": s.canonical_hash(approved),
            "status": "claimed",
        }
        _save(manifest, obj)


def reserve_case(manifest, *, case_id, attempt, source_hash):
    require(case_id == "F1" and type(attempt) is int and attempt in {9, 10})
    with s.manifest_lock(manifest) as obj:
        g = _active(obj, source_hash, "claimed" if attempt == 9 else "probe_passed")
        require(
            len(obj["semantic_attempts"]) == (16 if attempt == 9 else 17)
            and f"F1:{attempt}" not in obj["semantic_attempts"]
        )
        obj["semantic_attempts"][f"F1:{attempt}"] = {
            "source_hash": source_hash,
            "status": "started",
            "continuation": TAG,
        }
        g["status"] = "probe_running" if attempt == 9 else "integration_running"
        _save(manifest, obj)


def reserve_call(manifest, *, purpose, case_id, attempt, source_hash):
    require(case_id == "F1" and type(attempt) is int and attempt in {9, 10})
    allowed = (
        {"refund_decision"}
        if attempt == 9
        else {"resolution", "intents", "refund_query", "refund_decision"}
    )
    require(purpose in allowed)
    with s.manifest_lock(manifest) as obj:
        _active(
            obj, source_hash, "probe_running" if attempt == 9 else "integration_running"
        )
        calls = s.budget_ledger(manifest)
        require(
            len(calls) < 96
            and Counter(c["purpose"] for c in calls)[purpose] < s.LIMITS[purpose]
        )
        scoped = [c for c in calls if c.get("continuation") == TAG]
        require(
            len(scoped) < 10
            and all(
                c["source_hash"] == source_hash and c["case_id"] == "F1" for c in scoped
            )
        )
        used = [c for c in scoped if c["attempt"] == attempt]
        require(
            len(used) < (1 if attempt == 9 else 9)
            and sum(c["purpose"] == purpose for c in used)
            < (6 if attempt == 10 and purpose == "refund_decision" else 1)
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
            "continuation": TAG,
        }
        calls.append(entry)
        s._save_ledger(manifest, calls, obj)
        return s.InvocationReservation(Path(manifest).resolve(), entry["invocation_id"])


def reserve_retrieval(manifest, *, label, source_hash):
    require(label == "F1:10")
    with s.manifest_lock(manifest) as obj:
        _active(obj, source_hash, "integration_running")
        used = obj["retrievals"]
        require(
            label not in used
            and len(used) < 16
            and not any(v.get("continuation") == TAG for v in used.values())
        )
        used[label] = {
            "source_hash": source_hash,
            "outcome": "reserved",
            "continuation": TAG,
        }
        s.save_json(manifest, obj)


def finish_phase(manifest, *, source_hash, attempt, passed):
    require(type(passed) is bool and attempt in {9, 10})
    with s.manifest_lock(manifest) as obj:
        g = _active(
            obj, source_hash, "probe_running" if attempt == 9 else "integration_running"
        )
        g["status"] = (
            ("probe_passed" if attempt == 9 else "completed") if passed else "halted"
        )
        obj["semantic_attempts"][f"F1:{attempt}"]["status"] = (
            "passed" if passed else "failed"
        )
        _save(manifest, obj)


def probe(ownership, source_hash, target, diagnostic):
    """One native terminal return over retained isolated observations, no tools executed."""
    from langchain_core.messages import messages_from_dict
    from scripts.smoke_refund_flow import BudgetedRefundClient

    from resolv.core.config import Settings
    from resolv.core.refund_config import default_refund_profile
    from resolv.core.refund_model import RefundModelClient
    from resolv.schemas.refund_flow import RefundRunRecord
    from resolv.services.context import TokenCounter
    from resolv.services.refund_context import build_refund_context
    from resolv.services.refund_decision import validate_refund_decision

    reserve_case(ownership, case_id="F1", attempt=9, source_hash=source_hash)
    original = json.loads((s.ROOT / PRIOR).read_text())["cases"][0]
    record = RefundRunRecord.model_validate_json(json.dumps(original["refund_record"]))
    saved = record.profile
    profile = default_refund_profile(
        understanding_config=saved.query_model,
        business_config=saved.decision_model,
        summary_profile=saved.summary_profile.value,
        adapter_descriptor=saved.adapter,
    ).model_copy(update={"selection": saved.selection, "max_agent_steps": 4})
    record = record.model_copy(update={"profile": profile})
    context = build_refund_context(
        record,
        record.request,
        record.binding,
        record.evidence,
        record.observations,
        history=[],
        summary=None,
        counter=TokenCounter(profile.decision_model.tokenizer_profile),
        profile=profile,
    )
    s.save_json(
        target / "probe-context-private.json",
        {
            "profile": profile.model_dump(mode="json"),
            "input": context.model_dump(mode="json"),
        },
    )
    client = RefundModelClient(Settings(), diagnostic_hook=diagnostic)
    budget = BudgetedRefundClient(
        client, ownership, "F1", 9, source_hash, continuation=TAG
    )
    result = {
        "case_id": "F1",
        "attempt": 9,
        "kind": "native_terminal_probe",
        "machine_passed": False,
        "input_hash": context.input_hash,
        "input_estimate": context.input_estimate,
        "profile": profile.model_dump(mode="json"),
        "archived_read_count": len(record.observations),
        "new_business_reads": 0,
        "new_applications": 0,
    }
    try:
        response = budget.decide(
            messages_from_dict(context.messages.value),
            profile=profile.decision_model,
            model_step=4,
            attempt=1,
            require_tool=False,
            final_round=True,
            audit_hook=lambda _: None,
        )
        value = validate_refund_decision(
            response.decision,
            request=record.request,
            binding=record.binding,
            evidence=record.evidence,
            observations=record.observations,
            profile=profile,
        )
        require(not response.tool_calls and response.input_hash == context.input_hash)
        require(
            value.verdict == "undetermined"
            and value.next_action == "human_draft"
            and bool(value.missing_information)
        )
        require(any("自签收之日起 7 天内" in q.quote for q in value.policy_citations))
        result.update(machine_passed=True, decision=value.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001 -- safe metadata only
        result.update(
            error_code=getattr(exc, "code", "native_probe_failed"),
            error_type=type(exc).__name__,
        )
    finally:
        client.close()
        result.update(audits=budget.audits, model_outputs=budget.outputs)
        s.save_json(target / "probe.json", result)
    return result


def main(argv=None):
    from scripts.refund_acceptance_v2 import load_cases
    from scripts.refund_failure_storage import OwnedRefundDiagnostics
    from scripts.refund_round_regression import fingerprints
    from scripts.smoke_refund_flow import owned_selection, semantic_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--approval-commit", required=True)
    args = parser.parse_args(argv)
    require(re.fullmatch("[0-9a-f]{40}", args.approval_commit))
    output = s.ROOT / PUBLIC / "R2-native-return.json"
    require(not output.exists())
    commit, sources = fingerprints()
    source_hash = s.canonical_hash(sources)
    cases = load_cases()
    approved = json.loads(
        subprocess.check_output(
            ["git", "show", args.approval_commit + ":" + APPROVAL], cwd=s.ROOT
        )
    )
    target = s.new_report_dir(args.ownership, args.report_dir)
    claim_grant(
        args.ownership,
        approved,
        approval_commit=args.approval_commit,
        source_hash=source_hash,
    )
    diagnostic = OwnedRefundDiagnostics(args.ownership, target)
    report = {
        "gate": "R2-native-return",
        "candidate_commit": commit,
        "source_fingerprints": sources,
        "approval_commit": args.approval_commit,
        "passed": False,
        "machine_passed": False,
        "manual_review_required": True,
        "formal_preserved": False,
        "expectation_revision": 2,
        "cases": [],
        "formal_migration_eligible": False,
    }
    before = None
    try:
        before = s.snapshot_formal_resources()
        s.save_json(target / "preservation-before.json", before)
        with s.owned_refund_resources(args.ownership) as resources:
            owned_selection(resources)
            result = probe(args.ownership, source_hash, target, diagnostic)
            report["probe"] = result
            middle = s.snapshot_formal_resources()
            s.save_json(target / "preservation-after-probe.json", middle)
            report["probe_formal_preserved"] = middle == before
            proceed = result["machine_passed"] and report["probe_formal_preserved"]
            finish_phase(
                args.ownership, source_hash=source_hash, attempt=9, passed=proceed
            )
            if proceed:
                result = semantic_case(
                    resources,
                    cases["cases"][0],
                    args.ownership,
                    source_hash,
                    target,
                    attempt=10,
                    continuation=TAG,
                    diagnostic_hook=diagnostic,
                )
                report["cases"] = [result]
                finish_phase(
                    args.ownership,
                    source_hash=source_hash,
                    attempt=10,
                    passed=result.get("machine_passed") is True,
                )
                report["machine_passed"] = result.get("machine_passed") is True
    except Exception as exc:  # noqa: BLE001 -- preserve no credentials/provider internals
        report.update(
            error_code=getattr(exc, "code", "native_regression_failed"),
            error_type=type(exc).__name__,
        )
        with s.manifest_lock(args.ownership) as obj:
            _active(obj, source_hash)["status"] = "halted"
            _save(args.ownership, obj)
    if before is not None:
        try:
            after = s.snapshot_formal_resources()
            s.save_json(target / "preservation-after.json", after)
            report["formal_preserved"] = before == after
        except Exception as exc:  # noqa: BLE001 -- unknown preservation is not success
            report["preservation_error_type"] = type(exc).__name__
    if not report["formal_preserved"]:
        with s.manifest_lock(args.ownership) as obj:
            _active(obj, source_hash)["status"] = "halted"
            _save(args.ownership, obj)
    report.update(
        failure_diagnostics=diagnostic.entries,
        ledger=s.budget_ledger(args.ownership),
        resource_identity=s.validate_manifest(args.ownership),
        prior_report_preserved=sha256((s.ROOT / PRIOR).read_bytes()).hexdigest()
        == approved["prior_report_sha256"],
    )
    s.save_json(target / "report.json", report)
    require(not output.exists())
    s.save_json(output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
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
