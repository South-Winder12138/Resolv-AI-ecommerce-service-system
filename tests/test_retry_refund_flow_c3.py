"""An explicit zero-case startup retry never weakens the original report guard."""

import importlib
import json
from hashlib import sha256
from pathlib import Path

import pytest
from scripts import refund_gate_support as s
from scripts import smoke_refund_flow as runner

from tests.test_refund_continuation import c1  # noqa: F401
from tests.test_refund_semantic_continuation import c2  # noqa: F401
from tests.test_refund_semantic_continuation_c3 import c3  # noqa: F401


def helper():
    assert importlib.util.find_spec("scripts.retry_refund_flow_c3"), (
        "missing isolated startup retry"
    )
    return importlib.import_module("scripts.retry_refund_flow_c3")


@pytest.fixture
def retry(c3):  # noqa: F811
    p, _, _, _ = c3
    public = s.ROOT / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    original = Path(__file__).resolve().parents[1]
    prior = public / "R2-C3.json"
    prior.write_bytes(
        (
            original / "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-C3.json"
        ).read_bytes()
    )
    obj = s.validate_manifest(p)
    data = json.loads(prior.read_text())
    data["resource_identity"] = obj
    data["ledger"] = s.budget_ledger(p)
    prior.write_text(json.dumps(data))
    auth = {
        "approved": True,
        "approval_text": "重新尝试，继续",
        "authorized_owner": obj["owner"],
        "source_hash": "2" * 64,
        "prior_report_sha256": sha256(prior.read_bytes()).hexdigest(),
        "baseline_ledger_hash": s.canonical_hash(s.budget_ledger(p)),
        "baseline_attempts_hash": s.canonical_hash(obj["semantic_attempts"]),
        "baseline_retrievals_hash": s.canonical_hash(obj["retrievals"]),
        "original_approval_commit": "880abcdc9f764b8a33d6e91c00b02031dff2376b",
        "max_additional_supplier_slots": 72,
        "max_semantic_attempts": 8,
        "max_additional_retrievals": 8,
    }
    return p, prior, prior.read_bytes(), auth


def test_retry_requires_approval_and_consumes_only_one_startup_marker(retry):
    m = helper()
    p, prior, raw, auth = retry
    with pytest.raises(ValueError):
        m.claim_retry(
            p,
            dict(auth, approved=False),
            approval_commit="3" * 40,
            source_hash="2" * 64,
        )
    before = s.validate_manifest(p)
    m.claim_retry(p, auth, approval_commit="3" * 40, source_hash="2" * 64)
    with pytest.raises(ValueError):
        m.claim_retry(p, auth, approval_commit="3" * 40, source_hash="2" * 64)
    after = s.validate_manifest(p)
    assert after["continuations"] == before["continuations"]
    assert after["semantic_attempts"] == before["semantic_attempts"]
    assert len(s.budget_ledger(p)) == 18 and len(after["retrievals"]) == 7
    assert prior.read_bytes() == raw


@pytest.mark.parametrize(
    "field",
    [
        "approval_text",
        "authorized_owner",
        "source_hash",
        "prior_report_sha256",
        "baseline_ledger_hash",
        "baseline_attempts_hash",
        "baseline_retrievals_hash",
        "max_additional_supplier_slots",
    ],
)
def test_retry_drift_refuses_before_claim_and_preserves_prior(retry, field):
    m = helper()
    p, prior, raw, auth = retry
    auth[field] = "changed"
    with pytest.raises(ValueError):
        m.claim_retry(p, auth, approval_commit="3" * 40, source_hash="2" * 64)
    assert "c3_startup_retry" not in s.validate_manifest(p)
    assert prior.read_bytes() == raw


def test_retry_rejects_prior_with_started_case_or_existing_grant(retry):
    m = helper()
    p, prior, raw, auth = retry
    changed = json.loads(raw)
    changed["cases"][0]["status"] = "failed"
    prior.write_text(json.dumps(changed))
    auth["prior_report_sha256"] = sha256(prior.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        m.claim_retry(p, auth, approval_commit="3" * 40, source_hash="2" * 64)
    prior.write_bytes(raw)
    auth["prior_report_sha256"] = sha256(raw).hexdigest()
    obj = s.validate_manifest(p)
    obj["continuations"]["C3"] = {"halted": "F1"}
    s.save_json(p, obj)
    with pytest.raises(ValueError):
        m.claim_retry(p, auth, approval_commit="3" * 40, source_hash="2" * 64)
    assert prior.read_bytes() == raw


def test_normal_C3_CLI_still_refuses_existing_report(retry, monkeypatch):
    p, prior, raw, _ = retry
    monkeypatch.setattr(runner, "ROOT", s.ROOT)
    with pytest.raises(ValueError, match="Refusing to overwrite R2 evidence"):
        runner.main(
            [
                "--mode",
                "semantic",
                "--ownership",
                str(p),
                "--report-dir",
                str(p.parent / "normal-refusal"),
                "--continuation",
                "C3",
                "--approval-plan-commit",
                "3" * 40,
            ]
        )
    assert prior.read_bytes() == raw


def test_retry_reads_committed_approval_and_keeps_new_startup_failure_separate(
    retry, monkeypatch
):
    m = helper()
    p, prior, raw, auth = retry
    with pytest.raises(ValueError):
        m.read_approval("0" * 40)
    from scripts import refund_semantic_continuation as semantic

    monkeypatch.setattr(
        semantic, "fingerprints", lambda: ("a" * 40, {"fixture": "b" * 64})
    )
    auth["source_hash"] = s.canonical_hash({"fixture": "b" * 64})
    monkeypatch.setattr(m, "read_approval", lambda _: auth)

    def failed_snapshot(*a, **kw):
        raise RuntimeError("owned startup control")

    monkeypatch.setattr(s, "snapshot_formal_resources", failed_snapshot)
    assert (
        m.main(
            [
                "--ownership",
                str(p),
                "--report-dir",
                str(p.parent / "retry1"),
                "--approval-retry-commit",
                "3" * 40,
            ]
        )
        == 1
    )
    out = prior.parent / "R2-C3-retry-1.json"
    report = json.loads(out.read_text())
    assert report["prior_report_sha256"] == sha256(raw).hexdigest()
    assert report["prior_report_preserved"] is True
    assert (
        report["formal_preserved"] is False and report["error_type"] == "RuntimeError"
    )
    assert all(c["status"] == "not_executed" for c in report["cases"])
    assert prior.read_bytes() == raw
    with pytest.raises(ValueError, match="immutable"):
        m.main(
            [
                "--ownership",
                str(p),
                "--report-dir",
                str(p.parent / "retry2"),
                "--approval-retry-commit",
                "3" * 40,
            ]
        )
