"""The approved native probe and F1 share a closed, two-attempt finite grant."""

import copy
import json
from hashlib import sha256

import pytest
from scripts import refund_gate_support as s

from tests.test_refund_continuation import c1  # noqa: F401
from tests.test_refund_output_regression import (
    completeness,  # noqa: F401
    context_identity,  # noqa: F401
    identity_claim,
    identity_reserve,
    repair,  # noqa: F401
)
from tests.test_refund_round_regression import regression  # noqa: F401
from tests.test_refund_semantic_continuation import c2  # noqa: F401
from tests.test_refund_semantic_continuation_c3 import c3  # noqa: F401


@pytest.fixture
def native_row(context_identity):  # noqa: F811
    from scripts.refund_round_regression import PUBLIC, finish

    row = context_identity
    identity_claim(row)
    identity_reserve(row)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 4:
        identity_reserve(row, purpose)
    s.reserve_refund_retrieval(
        row[0], label="F1:8", source_hash="7" * 64, continuation="context-identity"
    )
    finish(row[0], source_hash="7" * 64, passed=False, scope="context-identity")
    prior = s.ROOT / PUBLIC / "R2-context-identity-F1.json"
    prior.write_text(
        json.dumps({"cases": [{"case_id": "F1", "machine_passed": False}]})
    )
    obj = s.validate_manifest(row[0])
    from scripts.refund_acceptance_v2 import CASES_SHA256

    auth = {
        "authorization_kind": "explicit_user_reply",
        "approval_text": "按此方案修复并验证（推荐）",
        "expectation_approval_text": "修正 F1 验收，另加明确口径的隔离正向案例（推荐）",
        "source_hash": "9" * 64,
        "authorized_owner": obj["owner"],
        "max_additional_supplier_slots": 10,
        "max_semantic_attempts": 2,
        "max_additional_retrievals": 1,
        "probe_attempt": 9,
        "integration_attempt": 10,
        "prior_report_sha256": sha256(prior.read_bytes()).hexdigest(),
        "baseline_ledger_hash": s.canonical_hash(s.budget_ledger(row[0])),
        "baseline_attempts_hash": s.canonical_hash(obj["semantic_attempts"]),
        "baseline_retrievals_hash": s.canonical_hash(obj["retrievals"]),
        "cases_sha256": CASES_SHA256,
    }
    return row[0], auth, copy.deepcopy(obj), prior.read_bytes()


def claim(row, **changes):
    from scripts.refund_native_regression import claim_grant

    return claim_grant(
        row[0], row[1] | changes, approval_commit="a" * 40, source_hash="9" * 64
    )


def reserve(row, attempt, purpose=None, **changes):
    args = {
        "case_id": "F1",
        "attempt": attempt,
        "source_hash": "9" * 64,
        "continuation": "native-return",
    } | changes
    return (
        s.reserve_refund_call(row[0], purpose=purpose, **args)
        if purpose
        else s.reserve_refund_case_attempt(row[0], **args)
    )


def test_native_two_stage_grant_caps_and_prior_history(native_row):
    from scripts.refund_native_regression import finish_phase

    r = native_row
    with pytest.raises(ValueError):
        reserve(r, 9)
    claim(r)
    with pytest.raises(ValueError):
        claim(r)
    with pytest.raises(ValueError):
        reserve(r, 10)
    reserve(r, 9)
    reserve(r, 9, "refund_decision")
    for purpose in s.LIMITS:
        with pytest.raises(ValueError):
            reserve(r, 9, purpose)
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            r[0], label="F1:9", source_hash="9" * 64, continuation="native-return"
        )
    finish_phase(r[0], source_hash="9" * 64, attempt=9, passed=True)
    reserve(r, 10)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        reserve(r, 10, purpose)
    s.reserve_refund_retrieval(
        r[0], label="F1:10", source_hash="9" * 64, continuation="native-return"
    )
    for purpose in s.LIMITS:
        with pytest.raises(ValueError):
            reserve(r, 10, purpose)
    finish_phase(r[0], source_hash="9" * 64, attempt=10, passed=True)
    after = s.validate_manifest(r[0])
    assert len(s.budget_ledger(r[0])) == 70
    assert len(after["semantic_attempts"]) == 18 and len(after["retrievals"]) == 13
    for k, v in r[2].items():
        if k not in {"semantic_attempts", "retrievals"}:
            assert after[k] == v
    with pytest.raises(ValueError):
        reserve(r, 11)


@pytest.mark.parametrize(
    "field",
    [
        "approval_text",
        "expectation_approval_text",
        "authorized_owner",
        "source_hash",
        "max_additional_supplier_slots",
        "max_semantic_attempts",
        "max_additional_retrievals",
        "prior_report_sha256",
        "baseline_ledger_hash",
        "baseline_attempts_hash",
        "baseline_retrievals_hash",
        "cases_sha256",
    ],
)
def test_native_grant_rejects_drift_before_claim(native_row, field):
    with pytest.raises(ValueError):
        claim(native_row, **{field: "changed"})
    assert "native_return_repair" not in s.validate_manifest(native_row[0])


def test_native_probe_first_failure_never_allows_integration_or_retry(native_row):
    from scripts.refund_native_regression import finish_phase

    r = native_row
    claim(r)
    reserve(r, 9)
    finish_phase(r[0], source_hash="9" * 64, attempt=9, passed=False)
    for n in (9, 10, 11):
        with pytest.raises(ValueError):
            reserve(r, n)
    assert len(s.budget_ledger(r[0])) == 60


@pytest.mark.parametrize(
    "probe_passed,changed", [(False, False), (True, False), (True, True)]
)
def test_actual_cli_stops_before_integration_on_probe_or_preservation_failure(
    native_row, monkeypatch, probe_passed, changed
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scripts import refund_acceptance_v2 as amended
    from scripts import refund_native_regression as m
    from scripts import refund_round_regression as previous
    from scripts import smoke_refund_flow as flow

    row = native_row
    sources = {"fixture": "c" * 64}
    digest = s.canonical_hash(sources)
    auth = row[1] | {"source_hash": digest}
    monkeypatch.setattr(previous, "fingerprints", lambda: ("a" * 40, sources))
    monkeypatch.setattr(
        m.subprocess, "check_output", lambda *a, **kw: json.dumps(auth).encode()
    )
    monkeypatch.setattr(
        amended,
        "load_cases",
        lambda: {"cases": [{"case_id": "F1", "expected": {"verdict": "undetermined"}}]},
    )
    snapshots = []

    def snapshot(**kw):
        snapshots.append(1)
        return {"value": "changed" if changed and len(snapshots) > 1 else "original"}

    monkeypatch.setattr(s, "snapshot_formal_resources", snapshot)

    @contextmanager
    def owned(_):
        yield SimpleNamespace()

    monkeypatch.setattr(s, "owned_refund_resources", owned)
    monkeypatch.setattr(flow, "owned_selection", lambda _: None)
    invoked = []

    def probe(path, source_hash, *args):
        m.reserve_case(path, case_id="F1", attempt=9, source_hash=source_hash)
        return {"machine_passed": probe_passed}

    monkeypatch.setattr(m, "probe", probe)

    def integration(resources, case, path, source_hash, target, **kw):
        invoked.append(1)
        assert case["expected"]["verdict"] == "undetermined" and kw["attempt"] == 10
        m.reserve_case(path, case_id="F1", attempt=10, source_hash=source_hash)
        return {"case_id": "F1", "machine_passed": True}

    monkeypatch.setattr(flow, "semantic_case", integration)
    argv = [
        "--ownership",
        str(row[0]),
        "--report-dir",
        str(row[0].parent / "native-cli"),
        "--approval-commit",
        "a" * 40,
    ]
    assert m.main(argv) == (0 if probe_passed and not changed else 1)
    assert bool(invoked) == (probe_passed and not changed)
    with pytest.raises(ValueError):
        m.main(argv)


@pytest.fixture
def chain_row(native_row):
    from scripts.refund_native_regression import finish_phase

    claim(native_row)
    reserve(native_row, 9)
    reserve(native_row, 9, "refund_decision")
    finish_phase(native_row[0], source_hash="9" * 64, attempt=9, passed=False)
    prior = (
        s.ROOT
        / "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-native-return.json"
    )
    s.save_json(
        prior, {"probe": {"case_id": "F1", "machine_passed": False}, "cases": []}
    )
    obj = s.validate_manifest(native_row[0])
    from scripts.refund_acceptance_v2 import CASES_SHA256

    auth = {
        "authorization_kind": "direct_user_request",
        "approval_text": "对这条链路进行验证，看是否还会出现异常",
        "authorized_owner": obj["owner"],
        "source_hash": "e" * 64,
        "case_id": "F1",
        "attempt": 10,
        "max_additional_supplier_slots": 9,
        "max_additional_retrievals": 1,
        "max_semantic_attempts": 1,
        "cases_sha256": CASES_SHA256,
        "prior_report_sha256": sha256(prior.read_bytes()).hexdigest(),
        "baseline_ledger_hash": s.canonical_hash(s.budget_ledger(native_row[0])),
        "baseline_attempts_hash": s.canonical_hash(obj["semantic_attempts"]),
        "baseline_retrievals_hash": s.canonical_hash(obj["retrievals"]),
    }
    return native_row[0], auth, copy.deepcopy(obj), prior, prior.read_bytes()


def chain_claim(row, **changes):
    from scripts.refund_round_regression import claim_regression

    return claim_regression(
        row[0],
        row[1] | changes,
        approval_commit="b" * 40,
        source_hash="e" * 64,
        scope="chain-validation",
    )


def chain_reserve(row, purpose=None, **changes):
    args = {
        "case_id": "F1",
        "attempt": 10,
        "source_hash": "e" * 64,
        "continuation": "chain-validation",
    } | changes
    return (
        s.reserve_refund_call(row[0], purpose=purpose, **args)
        if purpose
        else s.reserve_refund_case_attempt(row[0], **args)
    )


def test_new_chain_instruction_consumes_only_unused_attempt_without_reopening_native(
    chain_row,
):
    from scripts.refund_round_regression import finish

    r = chain_row
    assert len(s.budget_ledger(r[0])) == 61
    with pytest.raises(ValueError):
        chain_reserve(r)
    chain_claim(r)
    with pytest.raises(ValueError):
        chain_claim(r)
    for change in ({"case_id": "F2"}, {"attempt": 9}, {"source_hash": "d" * 64}):
        with pytest.raises(ValueError):
            chain_reserve(r, **change)
    chain_reserve(r)
    with pytest.raises(ValueError):
        chain_reserve(r)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        chain_reserve(r, purpose)
    for purpose in s.LIMITS:
        with pytest.raises(ValueError):
            chain_reserve(r, purpose)
    s.reserve_refund_retrieval(
        r[0], label="F1:10", source_hash="e" * 64, continuation="chain-validation"
    )
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            r[0], label="F1:10", source_hash="e" * 64, continuation="chain-validation"
        )
    finish(r[0], source_hash="e" * 64, passed=False, scope="chain-validation")
    after = s.validate_manifest(r[0])
    assert len(s.budget_ledger(r[0])) == 70
    assert len(after["semantic_attempts"]) == 18 and len(after["retrievals"]) == 13
    assert after["semantic_attempts"]["F1:10"]["status"] == "failed"
    assert after["native_return_repair"] == r[2]["native_return_repair"]
    assert after["continuations"] == r[2]["continuations"]
    assert r[3].read_bytes() == r[4]
    for scope in ("native-return", "chain-validation"):
        with pytest.raises(ValueError):
            s.reserve_refund_call(
                r[0],
                purpose="refund_decision",
                case_id="F1",
                attempt=10,
                source_hash="e" * 64,
                continuation=scope,
            )


@pytest.mark.parametrize(
    "field",
    [
        "approval_text",
        "authorization_kind",
        "authorized_owner",
        "source_hash",
        "case_id",
        "attempt",
        "max_additional_supplier_slots",
        "max_additional_retrievals",
        "max_semantic_attempts",
        "cases_sha256",
        "prior_report_sha256",
        "baseline_ledger_hash",
        "baseline_attempts_hash",
        "baseline_retrievals_hash",
    ],
)
def test_chain_validation_drift_refuses_before_claim(chain_row, field):
    with pytest.raises(ValueError):
        chain_claim(chain_row, **{field: "changed"})
    assert "chain_validation_regression" not in s.validate_manifest(chain_row[0])


@pytest.mark.parametrize("outcome", ["startup", "failed", "passed"])
def test_chain_CLI_uses_amended_case_private_failure_sink_and_immutable_report(
    chain_row, monkeypatch, outcome
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scripts import refund_acceptance_v2 as amended
    from scripts import refund_round_regression as m
    from scripts import smoke_refund_flow as flow

    from tests.test_refund_failure_storage import diagnostic

    path, auth, old, prior, raw = chain_row
    sources = {"fixture": "c" * 64}
    auth["source_hash"] = s.canonical_hash(sources)
    monkeypatch.setattr(m, "fingerprints", lambda: ("a" * 40, sources))
    monkeypatch.setattr(m, "read_approval", lambda *a, **kw: auth)

    @contextmanager
    def owned(_):
        yield SimpleNamespace()

    def snapshot(**kw):
        if outcome == "startup":
            raise RuntimeError("private startup message")
        return {"unchanged": True}

    def semantic(resources, case, manifest, source_hash, target, **kw):
        s.reserve_refund_case_attempt(
            manifest,
            case_id="F1",
            attempt=10,
            source_hash=source_hash,
            continuation="chain-validation",
        )
        if outcome == "failed":
            kw["diagnostic_hook"](diagnostic())
        return {
            "case_id": "F1",
            "expected": case["expected"],
            "machine_passed": outcome == "passed",
        }

    monkeypatch.setattr(s, "snapshot_formal_resources", snapshot)
    monkeypatch.setattr(s, "owned_refund_resources", owned)
    monkeypatch.setattr(flow, "owned_selection", lambda _: None)
    monkeypatch.setattr(
        flow,
        "load_cases",
        lambda: {"cases": [{"case_id": "F1", "expected": {"verdict": "eligible"}}]},
    )
    monkeypatch.setattr(
        amended,
        "load_cases",
        lambda: {"cases": [{"case_id": "F1", "expected": {"verdict": "undetermined"}}]},
    )
    monkeypatch.setattr(flow, "semantic_case", semantic)
    public = s.ROOT / m.PUBLIC
    for name in ("R2-prepare-continuation.json", "R2-retrieval.json"):
        s.save_json(public / name, {"passed": True})
    target = path.parent / "chain-cli"
    argv = [
        "--ownership",
        str(path),
        "--report-dir",
        str(target),
        "--approval-commit",
        "b" * 40,
        "--scope",
        "chain-validation",
    ]
    assert m.main(argv) == (0 if outcome == "passed" else 1)
    report = json.loads((public / "R2-chain-validation-F1.json").read_text())
    assert report["passed"] is False and report["expectation_revision"] == 2
    assert report["prior_report_preserved"] and prior.read_bytes() == raw
    assert (
        s.validate_manifest(path)["native_return_repair"] == old["native_return_repair"]
    )
    assert "private startup message" not in json.dumps(report)
    if outcome != "startup":
        assert report["cases"][0]["expected"]["verdict"] == "undetermined"
    if outcome == "failed":
        assert report["failure_diagnostics"][0]["stage"] == "json"
        assert "PRIVATE_PUBLIC_PAYLOAD" not in json.dumps(report)
        assert len(list((target / "failed-responses").glob("*.json"))) == 1
    with pytest.raises(ValueError):
        m.main(argv)
