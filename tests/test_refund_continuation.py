"""C1 is a finite grant; old attempts and global ceilings remain authoritative."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest
from scripts import refund_gate_support as support


@pytest.fixture
def c1(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1]
    plan = "docs/superpowers/plans/2026-10-16-resolv-refund-flow-continuation-C1.md"
    auth = "docs/superpowers/acceptance/2026-10-16-refund-flow/continuation-C1-authorization.json"
    approval = json.loads((original / auth).read_text())
    for name in (plan, auth, "tests/fixtures/refund_flow/semantic_cases.json"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((original / name).read_bytes())
    approval["plan_sha256"] = sha256((tmp_path / plan).read_bytes()).hexdigest()
    (tmp_path / auth).write_text(json.dumps(approval))
    monkeypatch.setattr(support, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    manifest = w / "resource-ownership.json"
    obj = support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    obj.update(
        semantic_attempts={
            f"F{i}:1": {"source_hash": "c" * 64, "status": "started"}
            for i in range(1, 9)
        },
        retests={"F1": "d" * 64, "F3": "d" * 64},
        retrievals={f"formal{i}": {} for i in range(6)},
    )
    obj["semantic_attempts"].update(
        {f"F{i}:2": {"source_hash": "d" * 64, "status": "started"} for i in (1, 3)}
    )
    support.save_json(manifest, obj)
    calls = [
        {
            "invocation_id": str(i),
            "purpose": purpose,
            "case_id": "R1" if i < 3 else "F1",
            "attempt": 1,
            "source_hash": "c" * 64,
            "outcome": "failed",
        }
        for i, purpose in enumerate(
            [
                "refund_query",
                "refund_decision",
                "refund_decision",
                "resolution",
                "intents",
                "resolution",
            ]
        )
    ]
    support._save_ledger(manifest, calls, obj)
    approval.update(
        authorized_owner=obj["owner"],
        baseline_ledger_hash=support.canonical_hash(calls),
        baseline_attempts_hash=support.canonical_hash(obj["semantic_attempts"]),
    )
    (tmp_path / auth).write_text(json.dumps(approval))
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "--", plan, auth, "tests"],
        [
            "git",
            "-c",
            "user.name=Offline",
            "-c",
            "user.email=offline@example.invalid",
            "commit",
            "-qm",
            "approved",
        ],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    return manifest, commit, obj, calls


def approve(c1):
    manifest, commit, _, _ = c1
    assert hasattr(support, "authorize_refund_continuation"), "missing finite C1 grant"
    support.authorize_refund_continuation(
        manifest, approval_commit=commit, source_hash="e" * 64
    )


def test_C1_preserves_old_attempts_and_fences_duplicate_source(c1):
    p, commit, before, calls = c1
    with pytest.raises(ValueError):
        support.reserve_refund_case_attempt(
            p, case_id="F1", attempt=3, source_hash="e" * 64
        )
    approve(c1)
    saved = support.validate_manifest(p)
    assert saved["semantic_attempts"] == before["semantic_attempts"]
    assert saved["retests"] == before["retests"] and support.budget_ledger(p) == calls
    with pytest.raises(ValueError):
        support.authorize_refund_continuation(
            p, approval_commit=commit, source_hash="f" * 64
        )
    with pytest.raises(ValueError):
        support.reserve_refund_case_attempt(
            p, case_id="F1", attempt=3, source_hash="f" * 64
        )
    support.reserve_refund_transport_attempt(p, source_hash="e" * 64)
    support.finish_continuation_r1(p, source_hash="e" * 64, passed=True)
    support.reserve_refund_case_attempt(
        p, case_id="F1", attempt=3, source_hash="e" * 64
    )
    with pytest.raises(ValueError):
        support.reserve_refund_case_attempt(
            p, case_id="F1", attempt=3, source_hash="e" * 64
        )
    assert len(support.validate_manifest(p)["semantic_attempts"]) == 11


def test_C1_requires_valid_committed_approval_and_fixed_case_bytes(c1):
    p, commit, _, _ = c1
    assert hasattr(support, "authorize_refund_continuation"), "missing finite C1 grant"
    for bad in ("HEAD", "f" * 40):
        with pytest.raises(ValueError):
            support.authorize_refund_continuation(
                p, approval_commit=bad, source_hash="e" * 64
            )
    path = support.ROOT / "tests/fixtures/refund_flow/semantic_cases.json"
    path.write_text("changed cases")
    with pytest.raises(ValueError):
        support.authorize_refund_continuation(
            p, approval_commit=commit, source_hash="e" * 64
        )


def test_C1_R1_is_one_attempt_and_each_purpose_has_a_hard_cap(c1):
    p, _, _, _ = c1
    approve(c1)
    support.reserve_refund_transport_attempt(p, source_hash="e" * 64)
    with pytest.raises(ValueError):
        support.reserve_refund_transport_attempt(p, source_hash="e" * 64)

    def reserve(_):
        try:
            return support.reserve_refund_call(
                p,
                purpose="refund_decision",
                case_id="R1",
                attempt=2,
                source_hash="e" * 64,
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(reserve, range(16)))
    assert sum(x is not None for x in results) == 7
    assert len(support.budget_ledger(p)) == 13
    with pytest.raises(ValueError):
        support.reserve_refund_case_attempt(
            p, case_id="F2", attempt=2, source_hash="e" * 64
        )


def test_C1_failed_case_stops_later_attempts_and_does_not_reset_history(c1):
    p, _, _, _ = c1
    approve(c1)
    support.reserve_refund_transport_attempt(p, source_hash="e" * 64)
    support.finish_continuation_r1(p, source_hash="e" * 64, passed=True)
    support.reserve_refund_case_attempt(
        p, case_id="F1", attempt=3, source_hash="e" * 64
    )
    support.halt_refund_continuation(p, source_hash="e" * 64, case_id="F1")
    with pytest.raises(ValueError):
        support.reserve_refund_case_attempt(
            p, case_id="F2", attempt=2, source_hash="e" * 64
        )
    assert len(support.validate_manifest(p)["semantic_attempts"]) == 11


def test_C1_approval_cannot_be_transferred_to_another_owner(c1):
    p, commit, before, _ = c1
    before["owner"] = "f" * 32
    # Keep a valid manifest but a different task identity.
    renamed = support.new_manifest(
        p.parent, owner="f" * 32, server_fingerprint="b" * 64
    )
    before.update(
        {
            k: renamed[k]
            for k in (
                "owner",
                "business_db",
                "conversation_database",
                "knowledge_database",
                "collection",
            )
        }
    )
    support.save_json(p, before)
    support._save_ledger(p, c1[3], before)
    with pytest.raises(ValueError):
        support.authorize_refund_continuation(
            p, approval_commit=commit, source_hash="e" * 64
        )


@pytest.mark.parametrize("kind", ["total", "purpose"])
def test_C1_cannot_bypass_original_global_supplier_ceilings(c1, kind):
    p, _, _, _ = c1
    approve(c1)
    support.reserve_refund_transport_attempt(p, source_hash="e" * 64)
    calls = support.budget_ledger(p)
    additional = 96 - len(calls) if kind == "total" else 63 - 2
    calls += [
        {
            "invocation_id": "prior-" + str(i),
            "purpose": "refund_decision",
            "case_id": "other",
            "attempt": 1,
            "source_hash": "c" * 64,
            "outcome": "unknown",
        }
        for i in range(additional)
    ]
    support._save_ledger(p, calls, support.validate_manifest(p))
    with pytest.raises(ValueError):
        support.reserve_refund_call(
            p, purpose="refund_decision", case_id="R1", attempt=2, source_hash="e" * 64
        )
    assert support.budget_ledger(p) == calls
