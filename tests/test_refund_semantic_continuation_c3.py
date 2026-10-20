"""C3 admits one new fixed round without reopening halted earlier grants."""

import copy
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest
from scripts import refund_gate_support as s

from tests.test_refund_continuation import c1  # noqa: F401
from tests.test_refund_semantic_continuation import (
    authorize as authorize_c2,
)
from tests.test_refund_semantic_continuation import (
    c2,  # noqa: F401
    module,
)
from tests.test_refund_semantic_continuation import (
    reserve_call as c2_call,
)
from tests.test_refund_semantic_continuation import (
    reserve_case as c2_case,
)

ATTEMPTS = {"F1": 4, "F2": 2, "F3": 3, "F4": 2, "F5": 2, "F6": 2, "F7": 2, "F8": 2}


@pytest.fixture
def c3(c2):  # noqa: F811
    authorize_c2(c2)
    c2_case(c2)
    for purpose in (
        "resolution",
        "intents",
        "refund_query",
        "refund_decision",
        "refund_decision",
    ):
        c2_call(c2, purpose)
    s.reserve_refund_retrieval(
        c2[0], label="F1:3", source_hash="f" * 64, continuation="C2"
    )
    module().finish_case(c2[0], source_hash="f" * 64, case_id="F1", passed=False)
    original = Path(__file__).resolve().parents[1]
    plan = "docs/superpowers/plans/2026-10-16-resolv-refund-flow-continuation-C3.md"
    auth = "docs/superpowers/acceptance/2026-10-16-refund-flow/continuation-C3-authorization.json"
    for name in (plan, auth):
        p = s.ROOT / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((original / name).read_bytes())
    obj = s.validate_manifest(c2[0])
    approval = json.loads((s.ROOT / auth).read_text())
    approval.update(
        authorized_owner=obj["owner"],
        plan_sha256=sha256((s.ROOT / plan).read_bytes()).hexdigest(),
        baseline_ledger_hash=s.canonical_hash(s.budget_ledger(c2[0])),
        baseline_attempts_hash=s.canonical_hash(obj["semantic_attempts"]),
        baseline_retrievals_hash=s.canonical_hash(obj["retrievals"]),
    )
    (s.ROOT / auth).write_text(json.dumps(approval))
    subprocess.run(
        ["git", "add", "--", plan, auth], cwd=s.ROOT, check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Offline",
            "-c",
            "user.email=offline@example.invalid",
            "commit",
            "-qm",
            "C3 approved",
        ],
        cwd=s.ROOT,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=s.ROOT, text=True
    ).strip()
    return c2[0], commit, copy.deepcopy(obj), s.budget_ledger(c2[0])


def authorize(c3):
    m = module()
    assert hasattr(m, "authorize_c3"), "missing explicit C3 authorization"
    m.authorize_c3(c3[0], approval_commit=c3[1], source_hash="1" * 64)


def reserve(c3, case_id="F1", attempt=4, source_hash="1" * 64):
    s.reserve_refund_case_attempt(
        c3[0],
        case_id=case_id,
        attempt=attempt,
        source_hash=source_hash,
        continuation="C3",
    )


def call(c3, purpose="refund_decision", case_id="F1", attempt=4):
    return s.reserve_refund_call(
        c3[0],
        purpose=purpose,
        case_id=case_id,
        attempt=attempt,
        source_hash="1" * 64,
        continuation="C3",
    )


def finish(c3, case_id="F1", passed=True):
    module().finish_case(
        c3[0], source_hash="1" * 64, case_id=case_id, passed=passed, continuation="C3"
    )


def test_C3_keeps_halted_C1_C2_and_default_entry_unchanged(c3):
    with pytest.raises(ValueError):
        reserve(c3)
    authorize(c3)
    obj = s.validate_manifest(c3[0])
    assert obj["continuations"]["C1"] == c3[2]["continuations"]["C1"]
    assert obj["continuations"]["C2"] == c3[2]["continuations"]["C2"]
    assert obj["semantic_attempts"] == c3[2]["semantic_attempts"]
    assert s.budget_ledger(c3[0]) == c3[3]
    with pytest.raises(ValueError):
        authorize(c3)
    for kwargs in ({}, {"continuation": "C2"}):
        with pytest.raises(ValueError):
            s.reserve_refund_case_attempt(
                c3[0], case_id="F1", attempt=3, source_hash="1" * 64, **kwargs
            )


@pytest.mark.parametrize(
    "tamper",
    [
        "approval",
        "owner",
        "ledger",
        "attempts",
        "retrievals",
        "cases",
        "commit",
        "C2",
        "caps",
    ],
)
def test_C3_rejects_drift_before_any_reservation(c3, tamper):
    assert hasattr(module(), "authorize_c3"), "missing explicit C3 authorization"
    p, commit, _, _ = c3
    if tamper == "commit":
        commit = "0" * 40
    elif tamper in ("approval", "caps", "owner"):
        name = "docs/superpowers/acceptance/2026-10-16-refund-flow/continuation-C3-authorization.json"
        data = json.loads((s.ROOT / name).read_text())
        data[
            {
                "approval": "approval_text",
                "caps": "resolution_limit",
                "owner": "authorized_owner",
            }[tamper]
        ] = "unapproved"
        (s.ROOT / name).write_text(json.dumps(data))
        subprocess.run(
            ["git", "add", name], cwd=s.ROOT, check=True, capture_output=True
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Offline",
                "-c",
                "user.email=offline@example.invalid",
                "commit",
                "-qm",
                "invalid",
            ],
            cwd=s.ROOT,
            check=True,
            capture_output=True,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=s.ROOT, text=True
        ).strip()
    elif tamper == "cases":
        (s.ROOT / module().CASES).write_text("changed")
    else:
        obj = s.validate_manifest(p)
        if tamper == "owner":
            obj["owner"] = "b" * 32
        elif tamper == "ledger":
            s._save_ledger(
                p, [*s.budget_ledger(p), dict(c3[3][-1], invocation_id="extra")], obj
            )
        elif tamper == "attempts":
            obj["semantic_attempts"]["extra"] = {}
        elif tamper == "retrievals":
            obj["retrievals"]["extra"] = {}
        else:
            obj["continuations"]["C2"]["halted"] = None
        s.save_json(p, obj)
    before = s.budget_ledger(p)
    with pytest.raises(ValueError):
        module().authorize_c3(p, approval_commit=commit, source_hash="1" * 64)
    assert "C3" not in s.validate_manifest(p)["continuations"]
    assert s.budget_ledger(p) == before


def test_C3_source_attempt_order_and_parallel_single_case(c3):
    authorize(c3)
    for case_id, attempt, source in (
        ("F2", 2, "1" * 64),
        ("F1", 3, "1" * 64),
        ("F1", 4, "2" * 64),
    ):
        with pytest.raises(ValueError):
            reserve(c3, case_id, attempt, source)

    def once(_):
        try:
            reserve(c3)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(once, range(16))) == 1
    for purpose in ("resolution", "intents", "refund_query"):
        call(c3, purpose)
        with pytest.raises(ValueError):
            call(c3, purpose)
    for _ in range(6):
        call(c3)
    with pytest.raises(ValueError):
        call(c3)


def test_C3_all_eight_exact_attempts_and_retrievals(c3):
    authorize(c3)
    for case_id, attempt in ATTEMPTS.items():
        reserve(c3, case_id, attempt)
        s.reserve_refund_retrieval(
            c3[0], label=f"{case_id}:{attempt}", source_hash="1" * 64, continuation="C3"
        )
        with pytest.raises(ValueError):
            s.reserve_refund_retrieval(
                c3[0],
                label=f"{case_id}:{attempt}",
                source_hash="1" * 64,
                continuation="C3",
            )
        finish(c3, case_id)
    obj = s.validate_manifest(c3[0])
    assert len(obj["semantic_attempts"]) == 19 and len(obj["retrievals"]) == 15
    assert module().grant(obj, continuation="C3")["completed_cases"] == list(ATTEMPTS)
    with pytest.raises(ValueError):
        reserve(c3, "F8", 2)


def test_C3_resolution_11_allowed_12_rejected_without_changing_default_limits(c3):
    authorize(c3)
    reserve(c3)
    obj = s.validate_manifest(c3[0])
    rows = s.budget_ledger(c3[0])
    rows += [
        dict(rows[0], invocation_id=f"filler{i}", purpose="resolution")
        for i in range(7)
    ]
    s._save_ledger(c3[0], rows, obj)
    call(c3, "resolution")
    finish(c3)
    reserve(c3, "F2", 2)
    with pytest.raises(ValueError):
        call(c3, "resolution", "F2", 2)
    assert s.LIMITS["resolution"] == 10


@pytest.mark.parametrize(
    "limit",
    [
        "global_calls",
        "round_calls",
        "purpose",
        "attempts",
        "global_retrievals",
        "round_retrievals",
    ],
)
def test_C3_cannot_exceed_global_or_round_caps(c3, limit):
    authorize(c3)
    p = c3[0]
    obj = s.validate_manifest(p)
    if limit == "attempts":
        for i in range(7):
            obj["semantic_attempts"][f"extra{i}"] = {}
        s.save_json(p, obj)
        reserve(c3)
        finish(c3)
        with pytest.raises(ValueError):
            reserve(c3, "F2", 2)
        return
    reserve(c3)
    obj = s.validate_manifest(p)
    if "retrievals" in limit:
        count = 9 if limit == "global_retrievals" else 8
        obj["retrievals"].update(
            {
                f"extra{i}": {
                    "continuation": "C3" if limit == "round_retrievals" else "old"
                }
                for i in range(count)
            }
        )
        s.save_json(p, obj)
        with pytest.raises(ValueError):
            s.reserve_refund_retrieval(
                p, label="F1:4", source_hash="1" * 64, continuation="C3"
            )
    else:
        rows = s.budget_ledger(p)
        count = {"global_calls": 96 - len(rows), "round_calls": 72, "purpose": 63 - 10}[
            limit
        ]
        rows += [
            dict(
                rows[0],
                invocation_id=f"extra{i}",
                purpose="refund_decision",
                case_id="other",
                continuation="C3" if limit == "round_calls" else "old",
            )
            for i in range(count)
        ]
        s._save_ledger(p, rows, obj)
        with pytest.raises(ValueError):
            call(c3)


def test_C3_first_failure_halts_without_mutating_older_grants(c3):
    authorize(c3)
    reserve(c3)
    finish(c3, passed=False)
    for case_id, attempt in (("F1", 4), ("F2", 2)):
        with pytest.raises(ValueError):
            reserve(c3, case_id, attempt)
    with pytest.raises(ValueError):
        call(c3)
    obj = s.validate_manifest(c3[0])
    assert obj["continuations"]["C3"]["halted"] == "F1"
    assert obj["continuations"]["C1"] == c3[2]["continuations"]["C1"]
    assert obj["continuations"]["C2"] == c3[2]["continuations"]["C2"]
