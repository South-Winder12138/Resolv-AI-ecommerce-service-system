"""A direct, one-case regression instruction cannot reopen exhausted grants."""

import copy
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import pytest
from scripts import refund_gate_support as s
from scripts import smoke_refund_flow as runner

from tests.test_refund_continuation import c1  # noqa: F401
from tests.test_refund_semantic_continuation import c2  # noqa: F401
from tests.test_refund_semantic_continuation_c3 import (
    authorize,
    c3,  # noqa: F401
    call,
    finish,
    reserve,
)


def helper():
    assert importlib.util.find_spec("scripts.refund_round_regression"), (
        "missing bounded F1 regression"
    )
    return importlib.import_module("scripts.refund_round_regression")


@pytest.fixture
def regression(c3):  # noqa: F811
    authorize(c3)
    reserve(c3)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        call(c3, purpose)
    s.reserve_refund_retrieval(
        c3[0], label="F1:4", source_hash="1" * 64, continuation="C3"
    )
    finish(c3, passed=False)
    public = s.ROOT / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    prior = public / "R2-C3-retry-1.json"
    prior.write_text(
        json.dumps({"cases": [{"case_id": "F1", "machine_passed": False}]})
    )
    obj = s.validate_manifest(c3[0])
    auth = {
        "authorization_kind": "direct_user_request",
        "approval_text": "好，现在完善轮次提示和最后一轮结束策略，然后继续测试",
        "authorized_owner": obj["owner"],
        "source_hash": "2" * 64,
        "prior_report_sha256": sha256(prior.read_bytes()).hexdigest(),
        "baseline_ledger_hash": s.canonical_hash(s.budget_ledger(c3[0])),
        "baseline_attempts_hash": s.canonical_hash(obj["semantic_attempts"]),
        "baseline_retrievals_hash": s.canonical_hash(obj["retrievals"]),
        "case_id": "F1",
        "attempt": 5,
        "max_additional_supplier_slots": 9,
        "max_additional_retrievals": 1,
        "max_semantic_attempts": 1,
    }
    return c3[0], auth, prior, prior.read_bytes(), copy.deepcopy(obj)


def claim(row, approved=None):
    helper().claim_regression(
        row[0], approved or row[1], approval_commit="3" * 40, source_hash="2" * 64
    )


def case(row, **kwargs):
    s.reserve_refund_case_attempt(
        row[0],
        **({"case_id": "F1", "attempt": 5, "source_hash": "2" * 64} | kwargs),
        continuation="round-control",
    )


def provider(row, purpose="refund_decision", **kwargs):
    return s.reserve_refund_call(
        row[0],
        purpose=purpose,
        **({"case_id": "F1", "attempt": 5, "source_hash": "2" * 64} | kwargs),
        continuation="round-control",
    )


def test_missing_claim_and_repeated_claim_or_case_refuse(regression):
    m = helper()
    with pytest.raises(ValueError):
        case(regression)
    claim(regression)
    with pytest.raises(ValueError):
        claim(regression)
    case(regression)
    with pytest.raises(ValueError):
        case(regression)
    after = s.validate_manifest(regression[0])
    assert after["continuations"] == regression[4]["continuations"]
    assert len(after["semantic_attempts"]) == 13
    assert regression[2].read_bytes() == regression[3]
    m.finish(regression[0], source_hash="2" * 64, passed=False)
    with pytest.raises(ValueError):
        provider(regression)


@pytest.mark.parametrize(
    "field",
    [
        "authorization_kind",
        "approval_text",
        "authorized_owner",
        "source_hash",
        "prior_report_sha256",
        "baseline_ledger_hash",
        "baseline_attempts_hash",
        "baseline_retrievals_hash",
        "case_id",
        "attempt",
        "max_additional_supplier_slots",
        "max_additional_retrievals",
        "max_semantic_attempts",
    ],
)
def test_authorization_or_baseline_drift_refuses_before_any_claim(regression, field):
    auth = dict(regression[1], **{field: "changed"})
    with pytest.raises(ValueError):
        claim(regression, auth)
    assert "round_control_regression" not in s.validate_manifest(regression[0])


def test_only_one_named_case_source_and_finite_purposes_allowed(regression):
    claim(regression)
    for changed in [{"case_id": "F2"}, {"attempt": 4}, {"source_hash": "4" * 64}]:
        with pytest.raises(ValueError):
            case(regression, **changed)
    case(regression)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        provider(regression, purpose)
    for purpose in s.LIMITS:
        with pytest.raises(ValueError):
            provider(regression, purpose)
    s.reserve_refund_retrieval(
        regression[0], label="F1:5", source_hash="2" * 64, continuation="round-control"
    )
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            regression[0],
            label="F1:5",
            source_hash="2" * 64,
            continuation="round-control",
        )
    assert len(s.budget_ledger(regression[0])) == 36
    assert len(s.validate_manifest(regression[0])["retrievals"]) == 9


def test_concurrent_seventh_decision_reservation_is_refused(regression):
    claim(regression)
    case(regression)

    def attempt(_):
        try:
            provider(regression)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(8))) == 6


def test_existing_global_call_cap_cannot_be_raised_by_regression(regression):
    claim(regression)
    case(regression)
    obj = s.validate_manifest(regression[0])
    calls = s.budget_ledger(regression[0])
    calls += [dict(calls[0], invocation_id=f"used-{i}") for i in range(96 - len(calls))]
    s._save_ledger(regression[0], calls, obj)
    with pytest.raises(ValueError):
        provider(regression)


@pytest.mark.parametrize("startup_failure", [True, False])
def test_cli_keeps_failed_prior_and_never_reuses_new_report(
    regression, monkeypatch, startup_failure
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    m = helper()
    p, auth, prior, raw, _ = regression
    hashes = {"fixture": "f" * 64}
    auth["source_hash"] = s.canonical_hash(hashes)
    monkeypatch.setattr(m, "fingerprints", lambda: ("a" * 40, hashes))
    monkeypatch.setattr(m, "read_approval", lambda _: auth)

    @contextmanager
    def owned(_):
        yield SimpleNamespace()

    def snapshot(**kwargs):
        if startup_failure:
            raise RuntimeError("secret must not escape")
        return {"formal": "unchanged"}

    def semantic(*args, **kwargs):
        s.reserve_refund_case_attempt(
            p,
            case_id="F1",
            attempt=5,
            source_hash=auth["source_hash"],
            continuation="round-control",
        )
        return {"case_id": "F1", "machine_passed": True}

    public = s.ROOT / m.PUBLIC
    for name in ("R2-prepare-continuation.json", "R2-retrieval.json"):
        (public / name).write_text(json.dumps({"passed": True}))
    monkeypatch.setattr(s, "owned_refund_resources", owned)
    monkeypatch.setattr(s, "snapshot_formal_resources", snapshot)
    monkeypatch.setattr(runner, "owned_selection", lambda _: None)
    monkeypatch.setattr(runner, "load_cases", lambda: {"cases": [{"case_id": "F1"}]})
    monkeypatch.setattr(runner, "semantic_case", semantic)
    argv = [
        "--ownership",
        str(p),
        "--report-dir",
        str(p.parent / "round-regression"),
        "--approval-commit",
        "3" * 40,
    ]
    assert m.main(argv) == (1 if startup_failure else 0)
    report = json.loads((public / "R2-round-control-F1.json").read_text())
    assert report["passed"] is False and report["machine_passed"] is not startup_failure
    assert report["prior_report_preserved"] and prior.read_bytes() == raw
    assert "secret" not in json.dumps(report)
    with pytest.raises(ValueError):
        m.main(argv)


def test_absent_committed_authorization_refuses():
    with pytest.raises(ValueError):
        helper().read_approval("0" * 40)
