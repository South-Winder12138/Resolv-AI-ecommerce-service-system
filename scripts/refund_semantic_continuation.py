"""Explicit finite semantic grants; halted earlier continuations stay closed."""

import json
import re
import subprocess
from collections import Counter
from hashlib import sha256
from pathlib import Path

from scripts import refund_gate_support as s
from scripts.refund_continuation import ATTEMPTS, CASES

PLAN = "docs/superpowers/plans/2026-10-16-resolv-refund-flow-continuation-C2.md"
APPROVAL = "docs/superpowers/acceptance/2026-10-16-refund-flow/continuation-C2-authorization.json"


def require(value):
    if not value:
        raise ValueError("Unapproved or exhausted semantic continuation")


def case_attempts(continuation):
    require(continuation in {"C2", "C3"})
    return ATTEMPTS | ({"F1": 4} if continuation == "C3" else {})


def fingerprints():
    from scripts.refund_continuation import fingerprints as previous

    _, names = previous()
    return s.committed_sources(
        (
            *names,
            "scripts/refund_semantic_continuation.py",
            "tests/test_refund_semantic_continuation.py",
            "tests/test_refund_semantic_continuation_c3.py",
            "tests/test_retry_refund_flow_c3.py",
        )
    )


def _save(manifest, obj, continuation):
    g = obj["continuations"][continuation]
    g["grant_hash"] = s.canonical_hash(
        {k: v for k, v in g.items() if k != "grant_hash"}
    )
    s.save_json(manifest, obj)


def grant(obj, source_hash=None, *, continuation="C2"):
    require(continuation in {"C2", "C3"})
    g = obj.get("continuations", {}).get(continuation)
    require(g is not None and g.get("owner") == obj["owner"])
    require(
        g.get("grant_hash")
        == s.canonical_hash({k: v for k, v in g.items() if k != "grant_hash"})
    )
    if source_hash is not None:
        require(
            re.fullmatch("[0-9a-f]{64}", source_hash or "")
            and g["source_hash"] == source_hash
            and not g["halted"]
        )
    return g


def authorize_c2(manifest, *, approval_commit, source_hash):
    return _authorize(manifest, approval_commit, source_hash, "C2")


def authorize_c3(manifest, *, approval_commit, source_hash):
    return _authorize(manifest, approval_commit, source_hash, "C3")


def _authorize(manifest, approval_commit, source_hash, continuation):
    from scripts.refund_continuation import grant as old_grant

    require(
        re.fullmatch("[0-9a-f]{40}", approval_commit or "")
        and re.fullmatch("[0-9a-f]{64}", source_hash or "")
    )

    def blob(name):
        try:
            return subprocess.check_output(
                ["git", "show", approval_commit + ":" + name],
                cwd=s.ROOT,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise ValueError("Missing committed semantic approval") from None

    approved = json.loads(blob(APPROVAL.replace("C2", continuation)))
    require(
        approved.get("approved") is True
        and approved.get("approval_text") == f"批准 {continuation}，按计划继续"
    )
    require(
        approved.get("plan_sha256")
        == sha256(blob(PLAN.replace("C2", continuation))).hexdigest()
    )
    require(
        approved.get("case_sha256")
        == sha256(blob(CASES)).hexdigest()
        == sha256((s.ROOT / CASES).read_bytes()).hexdigest()
    )
    require(
        approved.get("r1_prerequisite_deferred_for_isolated_R2_only") is True
        and approved.get("R1_retry_authorized") is False
        and approved.get("formal_migration_authorized") is False
    )
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
    if continuation == "C3":
        require(
            approved.get("cumulative_semantic_limit") == 19
            and approved.get("resolution_limit") == 11
        )
    with s.manifest_lock(manifest) as obj:
        require(continuation not in obj.get("continuations", {}))
        old = old_grant(obj)
        require(
            old is not None
            and old["halted"] == "R1"
            and old["r1_passed"] is False
            and not old["completed_cases"]
        )
        if continuation == "C3":
            previous = grant(obj)
            require(previous["halted"] == "F1" and not previous["completed_cases"])
        calls = s.budget_ledger(manifest)
        require(approved.get("authorized_owner") == obj["owner"])
        require(
            approved.get("baseline_ledger_hash") == s.canonical_hash(calls)
            and len(calls) == (18 if continuation == "C3" else 13)
        )
        require(
            approved.get("baseline_attempts_hash")
            == s.canonical_hash(obj["semantic_attempts"])
            and len(obj["semantic_attempts"]) == (11 if continuation == "C3" else 10)
        )
        require(
            approved.get("baseline_retrievals_hash")
            == s.canonical_hash(obj["retrievals"])
            and len(obj["retrievals"]) == (7 if continuation == "C3" else 6)
        )
        obj["continuations"][continuation] = {
            "owner": obj["owner"],
            "approval_commit": approval_commit,
            "approval_hash": s.canonical_hash(approved),
            "source_hash": source_hash,
            "completed_cases": [],
            "halted": None,
            "r1_prerequisite_deferred": True,
        }
        _save(manifest, obj, continuation)


def _case(obj, source_hash, case_id, attempt, *, started=True, continuation="C2"):
    g = grant(obj, source_hash, continuation=continuation)
    attempts = case_attempts(continuation)
    require(
        case_id in attempts and attempts[case_id] == attempt and type(attempt) is int
    )
    require(case_id == f"F{len(g['completed_cases']) + 1}")
    if started:
        row = obj.get("semantic_attempts", {}).get(f"{case_id}:{attempt}")
        require(
            row is not None
            and row.get("source_hash") == source_hash
            and row.get("continuation") == continuation
        )
    return g


def reserve_case(manifest, *, case_id, attempt, source_hash, continuation="C2"):
    with s.manifest_lock(manifest) as obj:
        _case(
            obj, source_hash, case_id, attempt, started=False, continuation=continuation
        )
        attempts = obj["semantic_attempts"]
        key = f"{case_id}:{attempt}"
        require(
            key not in attempts and len(attempts) < (19 if continuation == "C3" else 18)
        )
        attempts[key] = {
            "source_hash": source_hash,
            "status": "started",
            "continuation": continuation,
        }
        s.save_json(manifest, obj)


def reserve_call(
    manifest, *, purpose, case_id, attempt, source_hash, continuation="C2"
):
    with s.manifest_lock(manifest) as obj:
        _case(obj, source_hash, case_id, attempt, continuation=continuation)
        calls = s.budget_ledger(manifest)
        require(purpose in {"resolution", "intents", "refund_query", "refund_decision"})
        total = Counter(c["purpose"] for c in calls)
        limit = (
            11
            if continuation == "C3" and purpose == "resolution"
            else s.LIMITS[purpose]
        )
        require(len(calls) < 96 and total[purpose] < limit)
        require(sum(c.get("continuation") == continuation for c in calls) < 72)
        case = [c for c in calls if c["case_id"] == case_id and c["attempt"] == attempt]
        require(
            all(
                c["source_hash"] == source_hash
                and c.get("continuation") == continuation
                for c in case
            )
        )
        require(
            len(case) < 9
            and sum(c["purpose"] == purpose for c in case)
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
            "continuation": continuation,
        }
        calls.append(entry)
        s._save_ledger(manifest, calls, obj)
        return s.InvocationReservation(Path(manifest).resolve(), entry["invocation_id"])


def reserve_retrieval(manifest, *, label, source_hash, continuation="C2"):
    parts = label.split(":")
    require(len(parts) == 2 and parts[1].isdigit())
    with s.manifest_lock(manifest) as obj:
        _case(obj, source_hash, parts[0], int(parts[1]), continuation=continuation)
        require(label == f"{parts[0]}:{case_attempts(continuation)[parts[0]]}")
        used = obj["retrievals"]
        require(
            label not in used
            and len(used) < 16
            and sum(v.get("continuation") == continuation for v in used.values()) < 8
        )
        used[label] = {
            "source_hash": source_hash,
            "outcome": "reserved",
            "continuation": continuation,
        }
        s.save_json(manifest, obj)


def finish_case(manifest, *, source_hash, case_id, passed, continuation="C2"):
    with s.manifest_lock(manifest) as obj:
        g = _case(
            obj,
            source_hash,
            case_id,
            case_attempts(continuation).get(case_id),
            continuation=continuation,
        )
        require(type(passed) is bool)
        if passed:
            g["completed_cases"].append(case_id)
        else:
            g["halted"] = case_id
        _save(manifest, obj, continuation)
