"""One committed C1 authorization, using the existing owned immutable ledgers."""

import json
import re
import subprocess
from hashlib import sha256

from scripts import refund_gate_support as s

PLAN = "docs/superpowers/plans/2026-10-16-resolv-refund-flow-continuation-C1.md"
APPROVAL = "docs/superpowers/acceptance/2026-10-16-refund-flow/continuation-C1-authorization.json"
CASES = "tests/fixtures/refund_flow/semantic_cases.json"
ATTEMPTS = {f"F{i}": 3 if i in (1, 3) else 2 for i in range(1, 9)}


def _require(value):
    if not value:
        raise ValueError("Unapproved or exhausted C1 continuation")


def fingerprints():
    from scripts.smoke_refund_flow import SOURCES
    from scripts.smoke_refund_flow_model import SOURCES as MODEL_SOURCES

    return s.committed_sources(
        tuple(
            dict.fromkeys(
                (
                    *SOURCES,
                    *MODEL_SOURCES,
                    "scripts/refund_continuation.py",
                    "tests/test_refund_continuation.py",
                    "tests/test_smoke_refund_flow.py",
                    "tests/test_smoke_refund_flow_model.py",
                )
            )
        )
    )


def _save(manifest, obj):
    g = obj["continuations"]["C1"]
    g["grant_hash"] = s.canonical_hash(
        {k: v for k, v in g.items() if k != "grant_hash"}
    )
    s.save_json(manifest, obj)


def grant(obj, source_hash=None):
    g = obj.get("continuations", {}).get("C1")
    if g is None:
        return None
    _require(
        g.get("owner") == obj["owner"]
        and g.get("grant_hash")
        == s.canonical_hash({k: v for k, v in g.items() if k != "grant_hash"})
    )
    if source_hash is not None:
        _require(g["source_hash"] == source_hash and not g["halted"])
    return g


def authorize(manifest, *, approval_commit, source_hash):
    _require(
        re.fullmatch(r"[0-9a-f]{40}", approval_commit or "")
        and re.fullmatch(r"[0-9a-f]{64}", source_hash or "")
    )

    def blob(name):
        try:
            return subprocess.check_output(
                ["git", "show", approval_commit + ":" + name],
                cwd=s.ROOT,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise ValueError("Missing committed C1 approval") from None

    approved = json.loads(blob(APPROVAL))
    _require(
        approved.get("approved") is True
        and approved.get("approval_text") == "批准 C1，按计划继续（推荐）"
        and approved.get("plan_sha256") == sha256(blob(PLAN)).hexdigest()
        and approved.get("case_sha256")
        == sha256(blob(CASES)).hexdigest()
        == sha256((s.ROOT / CASES).read_bytes()).hexdigest()
        and approved.get("semantic_attempts") == ATTEMPTS
        and approved.get("supplier_cap") == 96
        and approved.get("retrieval_cap") == 16
        and approved.get("additional_supplier_calls") == 82
        and approved.get("additional_retrievals") == 8
    )
    with s.manifest_lock(manifest) as obj:
        existing = grant(obj)
        if existing:
            _require(
                existing["approval_commit"] == approval_commit
                and existing["source_hash"] == source_hash
            )
            return existing
        calls = s.budget_ledger(manifest)
        expected = {f"F{i}:1" for i in range(1, 9)} | {"F1:2", "F3:2"}
        _require(
            len(calls) == 6
            and set(obj.get("semantic_attempts", {})) == expected
            and set(obj["retests"]) == {"F1", "F3"}
            and len(obj.get("retrievals", {})) == 6
            and approved.get("authorized_owner") == obj["owner"]
            and approved.get("baseline_ledger_hash") == s.canonical_hash(calls)
            and approved.get("baseline_attempts_hash")
            == s.canonical_hash(obj["semantic_attempts"])
            and source_hash not in {c["source_hash"] for c in calls}
        )
        obj["continuations"] = {
            "C1": {
                "owner": obj["owner"],
                "source_hash": source_hash,
                "approval_commit": approval_commit,
                "approval_hash": s.canonical_hash(approved),
                "baseline_ledger_hash": s.canonical_hash(calls),
                "baseline_calls": 6,
                "transport_started": False,
                "r1_passed": False,
                "halted": None,
                "completed_cases": [],
            }
        }
        _save(manifest, obj)
        return obj["continuations"]["C1"]


def authorized(obj, case_id, attempt, source_hash):
    g = grant(obj, source_hash)
    if g is None:
        return False
    return attempt == (2 if case_id == "R1" else ATTEMPTS.get(case_id))


def start_transport(manifest, *, source_hash):
    with s.manifest_lock(manifest) as obj:
        g = grant(obj, source_hash)
        _require(g and not g["transport_started"])
        g["transport_started"] = True
        _save(manifest, obj)


def finish_r1(manifest, *, source_hash, passed):
    with s.manifest_lock(manifest) as obj:
        g = grant(obj, source_hash)
        _require(
            g and g["transport_started"] and not g["r1_passed"] and type(passed) is bool
        )
        g["r1_passed"] = passed
        g["halted"] = None if passed else "R1"
        _save(manifest, obj)


def finish_case(manifest, *, source_hash, case_id, passed):
    with s.manifest_lock(manifest) as obj:
        g = grant(obj, source_hash)
        _require(
            g
            and g["r1_passed"]
            and case_id == f"F{len(g['completed_cases']) + 1}"
            and f"{case_id}:{ATTEMPTS[case_id]}" in obj.get("semantic_attempts", {})
        )
        if passed:
            g["completed_cases"].append(case_id)
        else:
            g["halted"] = case_id
        _save(manifest, obj)


def halt(manifest, *, source_hash, case_id):
    with s.manifest_lock(manifest) as obj:
        g = grant(obj, source_hash)
        _require(g and case_id in {*ATTEMPTS, "R1"})
        g["halted"] = case_id
        _save(manifest, obj)
