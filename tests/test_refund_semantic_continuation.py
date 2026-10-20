"""C2 consumes unused isolated cases while preserving the failed C1 grant."""

import copy
import importlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest
from scripts import refund_gate_support as s

from tests.test_refund_continuation import approve, c1  # noqa: F401


def module():
    assert importlib.util.find_spec("scripts.refund_semantic_continuation"), (
        "missing explicit C2 semantic authorization"
    )
    return importlib.import_module("scripts.refund_semantic_continuation")


@pytest.fixture
def c2(c1):  # noqa: F811 — pytest injects the imported shared fixture
    manifest, _, _, _ = c1
    approve(c1)
    s.reserve_refund_transport_attempt(manifest, source_hash="e" * 64)
    for purpose in ["refund_query", *["refund_decision"] * 6]:
        s.reserve_refund_call(
            manifest, case_id="R1", attempt=2, source_hash="e" * 64, purpose=purpose
        )
    s.finish_continuation_r1(manifest, source_hash="e" * 64, passed=False)
    original = Path(__file__).resolve().parents[1]
    plan = "docs/superpowers/plans/2026-10-16-resolv-refund-flow-continuation-C2.md"
    auth = "docs/superpowers/acceptance/2026-10-16-refund-flow/continuation-C2-authorization.json"
    for name in (plan, auth):
        target = s.ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    obj = s.validate_manifest(manifest)
    approval = json.loads((s.ROOT / auth).read_text())
    approval.update(
        plan_sha256=sha256((s.ROOT / plan).read_bytes()).hexdigest(),
        authorized_owner=obj["owner"],
        baseline_ledger_hash=s.canonical_hash(s.budget_ledger(manifest)),
        baseline_attempts_hash=s.canonical_hash(obj["semantic_attempts"]),
        baseline_retrievals_hash=s.canonical_hash(obj["retrievals"]),
    )
    (s.ROOT / auth).write_text(json.dumps(approval))
    for cmd in (
        ["git", "add", "--", plan, auth],
        [
            "git",
            "-c",
            "user.name=Offline",
            "-c",
            "user.email=offline@example.invalid",
            "commit",
            "-qm",
            "C2 approved",
        ],
    ):
        subprocess.run(cmd, cwd=s.ROOT, check=True, capture_output=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=s.ROOT, text=True
    ).strip()
    return manifest, commit, copy.deepcopy(obj), s.budget_ledger(manifest)


def authorize(c2):
    p, commit, _, _ = c2
    module().authorize_c2(p, approval_commit=commit, source_hash="f" * 64)


def reserve_case(c2, case_id="F1", attempt=3):
    s.reserve_refund_case_attempt(
        c2[0], case_id=case_id, attempt=attempt, source_hash="f" * 64, continuation="C2"
    )


def reserve_call(c2, purpose="refund_decision", **kwargs):
    return s.reserve_refund_call(
        c2[0],
        case_id="F1",
        attempt=3,
        source_hash="f" * 64,
        purpose=purpose,
        continuation="C2",
        **kwargs,
    )


def test_C2_requires_explicit_approval_and_never_reopens_C1(c2):
    m = module()
    with pytest.raises(ValueError):
        reserve_case(c2)
    authorize(c2)
    p, commit, before, calls = c2
    saved = s.validate_manifest(p)
    assert saved["continuations"]["C1"] == before["continuations"]["C1"]
    assert saved["semantic_attempts"] == before["semantic_attempts"]
    assert s.budget_ledger(p) == calls and len(calls) == 13
    with pytest.raises(ValueError):
        m.authorize_c2(p, approval_commit=commit, source_hash="f" * 64)
    with pytest.raises(ValueError):
        s.reserve_refund_case_attempt(p, case_id="F1", attempt=3, source_hash="f" * 64)
    with pytest.raises(ValueError):
        s.reserve_refund_transport_attempt(p, source_hash="f" * 64)
    assert m.grant(saved)["r1_prerequisite_deferred"] is True


@pytest.mark.parametrize(
    "tamper", ["owner", "ledger", "attempts", "retrievals", "cases", "commit"]
)
def test_C2_rejects_changed_approval_inputs_before_reserving_any_work(c2, tamper):
    m = module()
    p, commit, _, calls = c2
    if tamper == "commit":
        commit = "0" * 40
    elif tamper == "cases":
        (s.ROOT / m.CASES).write_text("changed")
    else:
        obj = s.validate_manifest(p)
        if tamper == "owner":
            obj["continuations"]["C1"]["owner"] = "b" * 32
        elif tamper == "ledger":
            calls = [*calls, dict(calls[-1], invocation_id="extra")]
            s._save_ledger(p, calls, obj)
        elif tamper == "attempts":
            obj["semantic_attempts"]["F1:3"] = {}
        else:
            obj["retrievals"]["extra"] = {}
        s.save_json(p, obj)
    with pytest.raises(ValueError):
        m.authorize_c2(p, approval_commit=commit, source_hash="f" * 64)
    assert "C2" not in s.validate_manifest(p)["continuations"]
    assert s.budget_ledger(p) == calls


def test_C2_fences_sources_order_duplicate_and_parallel_reservations(c2):
    authorize(c2)
    with pytest.raises(ValueError):
        reserve_case(c2, "F2", 2)
    with pytest.raises(ValueError):
        s.reserve_refund_case_attempt(
            c2[0], case_id="F1", attempt=3, source_hash="a" * 64, continuation="C2"
        )
    reserve_case(c2)
    with pytest.raises(ValueError):
        reserve_case(c2)

    def reserve(_):
        try:
            return reserve_call(c2)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(reserve, range(20)))
    assert sum(row is not None for row in rows) == 6
    for purpose in ("resolution", "intents", "refund_query"):
        reserve_call(c2, purpose)
        with pytest.raises(ValueError):
            reserve_call(c2, purpose)
    assert len(s.budget_ledger(c2[0])) == 22
    with pytest.raises(ValueError):
        reserve_call(c2, "refund_summary")


@pytest.mark.parametrize("limit", ["total", "purpose", "C2"])
def test_C2_obeys_existing_global_and_new_round_call_caps(c2, limit):
    authorize(c2)
    reserve_case(c2)
    p = c2[0]
    calls = s.budget_ledger(p)
    count = {"total": 96 - len(calls), "purpose": 63 - 8, "C2": 72}[limit]
    purposes = (
        (
            ["resolution"] * 8
            + ["intents"] * 8
            + ["refund_query"] * 8
            + ["refund_decision"] * 48
        )
        if limit == "C2"
        else ["refund_decision"] * count
    )
    calls.extend(
        {
            "invocation_id": f"extra-{i}",
            "purpose": purposes[i],
            "case_id": "other",
            "attempt": 1,
            "source_hash": "f" * 64,
            "outcome": "unknown",
            "continuation": "C2" if limit == "C2" else "old",
        }
        for i in range(count)
    )
    s._save_ledger(p, calls, s.validate_manifest(p))
    with pytest.raises(ValueError):
        reserve_call(c2)
    assert s.budget_ledger(p) == calls


def test_C2_retrieval_is_reserved_once_and_first_failure_stops_the_round(c2):
    m = module()
    authorize(c2)
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            c2[0], label="F1:3", source_hash="f" * 64, continuation="C2"
        )
    reserve_case(c2)
    s.reserve_refund_retrieval(
        c2[0], label="F1:3", source_hash="f" * 64, continuation="C2"
    )
    with pytest.raises(ValueError):
        s.reserve_refund_retrieval(
            c2[0], label="F1:3", source_hash="f" * 64, continuation="C2"
        )
    m.finish_case(c2[0], source_hash="f" * 64, case_id="F1", passed=False)
    for action in (lambda: reserve_case(c2, "F2", 2), lambda: reserve_call(c2)):
        with pytest.raises(ValueError):
            action()
    obj = s.validate_manifest(c2[0])
    assert obj["continuations"]["C2"]["halted"] == "F1"
    assert obj["continuations"]["C1"] == c2[2]["continuations"]["C1"]
    assert len(obj["semantic_attempts"]) == 11


def test_C2_all_eight_cases_use_exact_historical_attempts_and_stop_at_eight(c2):
    m = module()
    authorize(c2)
    for case_id, attempt in m.ATTEMPTS.items():
        reserve_case(c2, case_id, attempt)
        s.reserve_refund_retrieval(
            c2[0], label=f"{case_id}:{attempt}", source_hash="f" * 64, continuation="C2"
        )
        m.finish_case(c2[0], source_hash="f" * 64, case_id=case_id, passed=True)
    obj = s.validate_manifest(c2[0])
    assert len(obj["semantic_attempts"]) == 18 and len(obj["retrievals"]) == 14
    assert m.grant(obj)["completed_cases"] == list(m.ATTEMPTS)
    with pytest.raises(ValueError):
        reserve_case(c2)


@pytest.mark.parametrize("limit", ["attempts", "retrieval_global", "retrieval_C2"])
def test_C2_cannot_exceed_attempt_or_retrieval_limits_after_authorization(c2, limit):
    authorize(c2)
    p = c2[0]
    if limit != "attempts":
        reserve_case(c2)
    obj = s.validate_manifest(p)
    if limit == "attempts":
        obj["semantic_attempts"].update({f"foreign-{i}": {} for i in range(8)})
    else:
        count = 10 if limit == "retrieval_global" else 8
        obj["retrievals"].update(
            {
                f"foreign-{i}": {
                    "continuation": "C2" if limit == "retrieval_C2" else "old"
                }
                for i in range(count)
            }
        )
    s.save_json(p, obj)
    with pytest.raises(ValueError):
        if limit == "attempts":
            reserve_case(c2)
        else:
            s.reserve_refund_retrieval(
                p, label="F1:3", source_hash="f" * 64, continuation="C2"
            )
