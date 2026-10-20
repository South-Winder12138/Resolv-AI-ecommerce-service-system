"""The approved output repair gets one separate F1 attempt; old failures stay closed."""

import copy
import json
from hashlib import sha256

import pytest
from scripts import refund_gate_support as s
from scripts import refund_round_regression as m
from scripts import smoke_refund_flow as runner

from tests.test_refund_continuation import c1  # noqa: F401
from tests.test_refund_round_regression import (
    case,
    claim,
    provider,
    regression,  # noqa: F401
)
from tests.test_refund_semantic_continuation import c2  # noqa: F401
from tests.test_refund_semantic_continuation_c3 import c3  # noqa: F401


@pytest.fixture
def repair(regression):  # noqa: F811
    claim(regression)
    case(regression)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 5:
        provider(regression, purpose)
    s.reserve_refund_retrieval(
        regression[0], label="F1:5", source_hash="2" * 64, continuation="round-control"
    )
    m.finish(regression[0], source_hash="2" * 64, passed=False)
    prior = s.ROOT / m.PUBLIC / "R2-round-control-F1.json"
    prior.write_text(
        json.dumps({"cases": [{"case_id": "F1", "machine_passed": False}]})
    )
    obj = s.validate_manifest(regression[0])
    auth = dict(
        regression[1],
        approval_text="好，按你这个顺序，进行修复",
        source_hash="3" * 64,
        attempt=6,
        prior_report_sha256=sha256(prior.read_bytes()).hexdigest(),
        baseline_ledger_hash=s.canonical_hash(s.budget_ledger(regression[0])),
        baseline_attempts_hash=s.canonical_hash(obj["semantic_attempts"]),
        baseline_retrievals_hash=s.canonical_hash(obj["retrievals"]),
    )
    return regression[0], auth, copy.deepcopy(obj), prior.read_bytes()


def authorize(row, **kwargs):
    m.claim_regression(
        row[0],
        row[1] | kwargs,
        approval_commit="4" * 40,
        source_hash="3" * 64,
        scope="output-repair",
    )


def reserve(row, purpose=None, **kw):
    args = {
        "case_id": "F1",
        "attempt": 6,
        "source_hash": "3" * 64,
        "continuation": "output-repair",
    } | kw
    if purpose:
        return s.reserve_refund_call(row[0], purpose=purpose, **args)
    return s.reserve_refund_case_attempt(row[0], **args)


def test_repair_grant_is_one_shot_and_preserves_halted_grants(repair):
    with pytest.raises(ValueError):
        reserve(repair)
    authorize(repair)
    with pytest.raises(ValueError):
        authorize(repair)
    reserve(repair)
    with pytest.raises(ValueError):
        reserve(repair)
    after = s.validate_manifest(repair[0])
    assert after["continuations"] == repair[2]["continuations"]
    assert after["round_control_regression"] == repair[2]["round_control_regression"]
    assert len(after["semantic_attempts"]) == 14
    m.finish(repair[0], source_hash="3" * 64, passed=False, scope="output-repair")
    with pytest.raises(ValueError):
        reserve(repair, "refund_decision")


@pytest.mark.parametrize(
    "field",
    [
        "approval_text",
        "source_hash",
        "attempt",
        "max_additional_supplier_slots",
        "prior_report_sha256",
        "baseline_ledger_hash",
        "baseline_attempts_hash",
        "baseline_retrievals_hash",
    ],
)
def test_repair_drift_refuses_before_reservation(repair, field):
    with pytest.raises(ValueError):
        authorize(repair, **{field: "changed"})
    assert "output_repair_regression" not in s.validate_manifest(repair[0])


def test_only_F1_six_and_nine_calls_one_retrieval_allowed(repair):
    authorize(repair)
    for kwargs in [{"case_id": "F2"}, {"attempt": 5}, {"source_hash": "2" * 64}]:
        with pytest.raises(ValueError):
            reserve(repair, **kwargs)
    reserve(repair)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        reserve(repair, purpose)
    for purpose in s.LIMITS:
        with pytest.raises(ValueError):
            reserve(repair, purpose)
    s.reserve_refund_retrieval(
        repair[0], label="F1:6", source_hash="3" * 64, continuation="output-repair"
    )
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            repair[0], label="F1:6", source_hash="3" * 64, continuation="output-repair"
        )
    assert len(s.budget_ledger(repair[0])) == 44
    assert len(s.validate_manifest(repair[0])["retrievals"]) == 10


@pytest.mark.parametrize(
    "scope,attempt,fixture,prior_name",
    [
        ("output-repair", 6, "repair", "R2-round-control-F1.json"),
        ("output-completeness", 7, "completeness", "R2-output-repair-F1.json"),
        ("context-identity", 8, "context_identity", "R2-output-completeness-F1.json"),
    ],
)
def test_output_CLI_installs_private_diagnostics_without_reopening_previous_scope(
    request, monkeypatch, scope, attempt, fixture, prior_name
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tests.test_refund_failure_storage import diagnostic

    path, auth, _, prior_raw = request.getfixturevalue(fixture)
    hashes = {"fixture": "e" * 64}
    auth["source_hash"] = s.canonical_hash(hashes)
    monkeypatch.setattr(m, "fingerprints", lambda: ("a" * 40, hashes))
    monkeypatch.setattr(m, "read_approval", lambda *a, **kw: auth)

    @contextmanager
    def owned(_):
        yield SimpleNamespace()

    def semantic(*args, **kw):
        assert kw["attempt"] == attempt and kw["continuation"] == scope
        s.reserve_refund_case_attempt(
            path,
            case_id="F1",
            attempt=attempt,
            source_hash=auth["source_hash"],
            continuation=scope,
        )
        kw["diagnostic_hook"](diagnostic())
        return {
            "case_id": "F1",
            "machine_passed": False,
            "error_code": "refund_model_output_invalid",
        }

    public = s.ROOT / m.PUBLIC
    for name in ("R2-prepare-continuation.json", "R2-retrieval.json"):
        (public / name).write_text(json.dumps({"passed": True}))
    monkeypatch.setattr(s, "owned_refund_resources", owned)
    monkeypatch.setattr(
        s, "snapshot_formal_resources", lambda **kw: {"unchanged": True}
    )
    monkeypatch.setattr(runner, "owned_selection", lambda _: None)
    monkeypatch.setattr(runner, "load_cases", lambda: {"cases": [{"case_id": "F1"}]})
    monkeypatch.setattr(runner, "semantic_case", semantic)
    target = path.parent / scope
    argv = [
        "--ownership",
        str(path),
        "--report-dir",
        str(target),
        "--approval-commit",
        "4" * 40,
        "--scope",
        scope,
    ]
    assert m.main(argv) == 1
    report = json.loads((public / ("R2-" + scope + "-F1.json")).read_text())
    assert report["failure_diagnostics"][0]["stage"] == "json"
    assert "PRIVATE_PUBLIC_PAYLOAD" not in json.dumps(report)
    assert len(list((target / "failed-responses").glob("*.json"))) == 1
    assert (public / prior_name).read_bytes() == prior_raw
    with pytest.raises(ValueError):
        m.main(argv)


@pytest.fixture
def completeness(repair):
    authorize(repair)
    reserve(repair)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        reserve(repair, purpose)
    s.reserve_refund_retrieval(
        repair[0], label="F1:6", source_hash="3" * 64, continuation="output-repair"
    )
    m.finish(repair[0], source_hash="3" * 64, passed=False, scope="output-repair")
    prior = s.ROOT / m.PUBLIC / "R2-output-repair-F1.json"
    prior.write_text(
        json.dumps({"cases": [{"case_id": "F1", "machine_passed": False}]})
    )
    obj = s.validate_manifest(repair[0])
    auth = dict(
        repair[1],
        approval_text="现在开始进行真实模型验证，进行测试",
        source_hash="5" * 64,
        attempt=7,
        prior_report_sha256=sha256(prior.read_bytes()).hexdigest(),
        baseline_ledger_hash=s.canonical_hash(s.budget_ledger(repair[0])),
        baseline_attempts_hash=s.canonical_hash(obj["semantic_attempts"]),
        baseline_retrievals_hash=s.canonical_hash(obj["retrievals"]),
    )
    return repair[0], auth, copy.deepcopy(obj), prior.read_bytes()


def complete_claim(row, **kw):
    return m.claim_regression(
        row[0],
        row[1] | kw,
        approval_commit="6" * 40,
        source_hash="5" * 64,
        scope="output-completeness",
    )


def complete_reserve(row, purpose=None, **kw):
    args = {
        "case_id": "F1",
        "attempt": 7,
        "source_hash": "5" * 64,
        "continuation": "output-completeness",
    } | kw
    if purpose:
        return s.reserve_refund_call(row[0], purpose=purpose, **args)
    return s.reserve_refund_case_attempt(row[0], **args)


def test_complete_output_one_shot_caps_and_old_grants_remain_closed(completeness):
    with pytest.raises(ValueError):
        complete_reserve(completeness)
    complete_claim(completeness)
    with pytest.raises(ValueError):
        complete_claim(completeness)
    complete_reserve(completeness)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        complete_reserve(completeness, purpose)
    for purpose in s.LIMITS:
        with pytest.raises(ValueError):
            complete_reserve(completeness, purpose)
    s.reserve_refund_retrieval(
        completeness[0],
        label="F1:7",
        source_hash="5" * 64,
        continuation="output-completeness",
    )
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            completeness[0],
            label="F1:7",
            source_hash="5" * 64,
            continuation="output-completeness",
        )
    m.finish(
        completeness[0], source_hash="5" * 64, passed=False, scope="output-completeness"
    )
    after = s.validate_manifest(completeness[0])
    for key in (
        "continuations",
        "round_control_regression",
        "output_repair_regression",
    ):
        assert after[key] == completeness[2][key]
    assert (
        s.ROOT / m.PUBLIC / "R2-output-repair-F1.json"
    ).read_bytes() == completeness[3]
    assert len(s.budget_ledger(completeness[0])) == 53
    assert len(after["semantic_attempts"]) == 15 and len(after["retrievals"]) == 11
    with pytest.raises(ValueError):
        complete_reserve(completeness, "refund_decision")


@pytest.mark.parametrize(
    "field",
    [
        "approval_text",
        "source_hash",
        "attempt",
        "baseline_ledger_hash",
        "prior_report_sha256",
    ],
)
def test_complete_output_drift_is_rejected_before_grant(completeness, field):
    with pytest.raises(ValueError):
        complete_claim(completeness, **{field: "changed"})
    assert "output_completeness_regression" not in s.validate_manifest(completeness[0])


def test_complete_output_only_bound_F1_seven_is_reserved(completeness):
    complete_claim(completeness)
    for kw in ({"case_id": "F2"}, {"attempt": 6}, {"source_hash": "3" * 64}):
        with pytest.raises(ValueError):
            complete_reserve(completeness, **kw)
    assert len(s.validate_manifest(completeness[0])["semantic_attempts"]) == 14


@pytest.fixture
def context_identity(completeness):
    complete_claim(completeness)
    complete_reserve(completeness)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        complete_reserve(completeness, purpose)
    s.reserve_refund_retrieval(
        completeness[0],
        label="F1:7",
        source_hash="5" * 64,
        continuation="output-completeness",
    )
    m.finish(
        completeness[0], source_hash="5" * 64, passed=False, scope="output-completeness"
    )
    prior = s.ROOT / m.PUBLIC / "R2-output-completeness-F1.json"
    prior.write_text(
        json.dumps({"cases": [{"case_id": "F1", "machine_passed": False}]})
    )
    obj = s.validate_manifest(completeness[0])
    auth = dict(
        completeness[1],
        approval_text="好，继续下一步",
        source_hash="7" * 64,
        attempt=8,
        prior_report_sha256=sha256(prior.read_bytes()).hexdigest(),
        baseline_ledger_hash=s.canonical_hash(s.budget_ledger(completeness[0])),
        baseline_attempts_hash=s.canonical_hash(obj["semantic_attempts"]),
        baseline_retrievals_hash=s.canonical_hash(obj["retrievals"]),
    )
    return completeness[0], auth, copy.deepcopy(obj), prior.read_bytes()


def identity_claim(row, **kw):
    return m.claim_regression(
        row[0],
        row[1] | kw,
        approval_commit="8" * 40,
        source_hash="7" * 64,
        scope="context-identity",
    )


def identity_reserve(row, purpose=None, **kw):
    args = {
        "case_id": "F1",
        "attempt": 8,
        "source_hash": "7" * 64,
        "continuation": "context-identity",
    } | kw
    if purpose:
        return s.reserve_refund_call(row[0], purpose=purpose, **args)
    return s.reserve_refund_case_attempt(row[0], **args)


def test_identity_scope_is_one_shot_with_same_caps_and_all_old_grants_preserved(
    context_identity,
):
    row = context_identity
    with pytest.raises(ValueError):
        identity_reserve(row)
    identity_claim(row)
    with pytest.raises(ValueError):
        identity_claim(row)
    for kw in ({"case_id": "F2"}, {"attempt": 7}, {"source_hash": "5" * 64}):
        with pytest.raises(ValueError):
            identity_reserve(row, **kw)
    identity_reserve(row)
    for purpose in ["resolution", "intents", "refund_query"] + ["refund_decision"] * 6:
        identity_reserve(row, purpose)
    for purpose in s.LIMITS:
        with pytest.raises(ValueError):
            identity_reserve(row, purpose)
    s.reserve_refund_retrieval(
        row[0], label="F1:8", source_hash="7" * 64, continuation="context-identity"
    )
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            row[0], label="F1:8", source_hash="7" * 64, continuation="context-identity"
        )
    m.finish(row[0], source_hash="7" * 64, passed=False, scope="context-identity")
    after = s.validate_manifest(row[0])
    for key in (
        "continuations",
        "round_control_regression",
        "output_repair_regression",
        "output_completeness_regression",
    ):
        assert after[key] == row[2][key]
    assert (s.ROOT / m.PUBLIC / "R2-output-completeness-F1.json").read_bytes() == row[3]
    assert (
        len(s.budget_ledger(row[0])) == 62
        and len(after["semantic_attempts"]) == 16
        and len(after["retrievals"]) == 12
    )
    with pytest.raises(ValueError):
        identity_reserve(row, "refund_decision")


@pytest.mark.parametrize(
    "field",
    ["approval_text", "source_hash", "baseline_ledger_hash", "prior_report_sha256"],
)
def test_identity_scope_drift_does_not_create_grant(context_identity, field):
    with pytest.raises(ValueError):
        identity_claim(context_identity, **{field: "changed"})
    assert "context_identity_regression" not in s.validate_manifest(context_identity[0])
