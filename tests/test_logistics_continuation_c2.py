"""The final three fixed cases have their own once-only, offline-tested budget."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.test_logistics_contracts import api
from tests.test_logistics_gate_support import prepared

__all__ = ["prepared"]


@pytest.fixture
def final_cases(prepared):
    module = api("scripts.smoke_logistics_model")
    fixture = json.loads((Path(__file__).parent / "fixtures/logistics_l1_receipt.json").read_text())
    root = prepared.workspace
    l1 = fixture["evidence"]
    l2, l3 = deepcopy(l1), deepcopy(l1)
    l3["model_calls"] = l3["model_calls"][:2] + l3["model_calls"][-2:]
    l3["business_audits"] = l3["business_audits"][1:]
    l3["actions"] = l3["actions"][1:]
    l3["actions"][0]["arguments"] = {"tracking_no": "MOCKSF002"}
    envelope = l3["attempts"][0]["envelope"]
    envelope["tracking_no"] = "MOCKSF002"
    envelope["shipment"].update(status="delivered", location=None, events=[], signed_at=None)
    l3["answer"] = "【模拟物流】该运单已签收，暂无轨迹记录。"
    revision = "b" * 40
    for case, evidence in (("L2", l2), ("L3", l3)):
        for index, call in enumerate(evidence["model_calls"]):
            call.update(case_id=case, revision=revision, id=sha256(f"{case}:{index}".encode()).hexdigest()[:32])
    parent = {"state": "closed", "source_revision": fixture["source_revision"], "max_model_calls": 48,
        "http_calls": 5, "retrievals": 0, "calls": l1["model_calls"], "cases": {"L1": {"state": "failed"}, **{f"L{i}": {"state": "skipped"} for i in range(2, 7)}}}
    child = {"state": "closed", "source_revision": revision, "max_model_calls": 40,
        "http_calls": 9, "retrievals": 0, "calls": l2["model_calls"] + l3["model_calls"],
        "cases": {"L2": {"state": "passed"}, "L3": {"state": "failed"}, **{f"L{i}": {"state": "skipped"} for i in range(4, 7)}}}
    (root / "model-once").mkdir()
    (root / "model-C1").mkdir()
    write = prepared.module.save_json
    write(root / "budget.json", parent)
    write(root / "model-once/report.json", {"status": "failed", "source_revision": parent["source_revision"], "budget": parent, "cases": [{"case": "L1", "evidence": l1}]})
    fixture["original_report_sha256"] = sha256((root / "model-once/report.json").read_bytes()).hexdigest()
    write(root / "L1-adjudication.json", fixture)
    write(root / "budget-C1.json", child)
    write(root / "model-C1/report.json", {"status": "failed", "source_revision": revision, "budget": child, "cases": [{"case": "L2", "passed": True, "evidence": l2}, {"case": "L3", "passed": False, "evidence": l3}]})
    paths = [root / p for p in ("budget.json", "model-once/report.json", "L1-adjudication.json", "budget-C1.json", "model-C1/report.json")]
    write(root / "L1-L3-final-adjudication.json", {"format": "logistics-receipt-adjudication-v1", "model_http_requests_added": 0,
        "cumulative_model_http_requests": 14, "retrievals": 0, "preserved_hashes": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths},
        "cases": [{"case": case, "passed": True, "evidence": e} for case, e in (("L1", l1), ("L2", l2), ("L3", l3))]})
    paths.append(root / "L1-L3-final-adjudication.json")
    assert hasattr(module, "LogisticsFinalCasesBudget"), "missing independent C2 ledger"
    ledger = module.LogisticsFinalCasesBudget(prepared.manifest, prepared.identity["source_revision"])
    return SimpleNamespace(module=module, ledger=ledger, h=prepared, preserved={p: p.read_bytes() for p in paths})


def test_C2_only_final_cases_and_twenty_four_physical_sends(final_cases):
    c = final_cases
    c.ledger.prepare()
    c.ledger.open()
    assert [case["id"] for case in c.ledger.cases] == ["L4", "L5", "L6"]
    for forbidden in ("L1", "L2", "L3", "L5"):
        with pytest.raises(ValueError):
            c.ledger.begin_case(forbidden)
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with c.module.provider_http_guard(c.ledger, "https://provider.invalid/v1"):
        for case in c.ledger.cases:
            c.ledger.begin_case(case["id"])
            for purpose in ("resolution", "intents", *(["business"] * 6)):
                with c.module.model_permit(c.ledger.reserve(purpose)):
                    http.post("https://provider.invalid/v1/chat/completions", json={})
            with pytest.raises(ValueError):
                c.ledger.reserve("business")
            with pytest.raises(ValueError):
                c.ledger.reserve("retrieval")
            c.ledger.finish_case(case["id"], passed=True)
    assert c.ledger.read()["http_calls"] == 24
    assert c.ledger.parent_http_calls + c.ledger.read()["http_calls"] == 38
    assert all(p.read_bytes() == raw for p, raw in c.preserved.items())
    for operation in (c.ledger.prepare, c.ledger.open):
        with pytest.raises(ValueError):
            operation()


def test_C2_first_failure_stops_remaining_cases(final_cases):
    c = final_cases
    c.ledger.prepare()
    c.ledger.open()
    rows = c.module.execute_fixed_cases(c.ledger, lambda case: (_ for _ in ()).throw(ValueError("offline failure")))
    assert [row["case"] for row in rows] == ["L4"]
    assert c.ledger.read()["state"] == "closed"
    assert all(c.ledger.read()["cases"][case]["state"] == "skipped" for case in ("L5", "L6"))


@pytest.mark.parametrize("target", ["budget.json", "model-once/report.json", "L1-adjudication.json", "budget-C1.json", "model-C1/report.json", "L1-L3-final-adjudication.json"])
def test_C2_changed_reference_blocks_actual_send(final_cases, target):
    c = final_cases
    c.ledger.prepare()
    c.ledger.open()
    c.ledger.begin_case("L4")
    permit = c.ledger.reserve("resolution")
    path = c.h.workspace / target
    path.write_bytes(path.read_bytes() + b"\n")
    sent = []
    http = httpx.Client(transport=httpx.MockTransport(lambda request: sent.append(True) or httpx.Response(200, json={})))
    with c.module.provider_http_guard(c.ledger, "https://provider.invalid/v1"), c.module.model_permit(permit), pytest.raises(ValueError):
        http.post("https://provider.invalid/v1/chat/completions", json={})
    assert sent == []


def test_C2_invalid_previous_receipt_cannot_prepare(final_cases):
    c = final_cases
    target = c.h.workspace / "L1-L3-final-adjudication.json"
    receipt = json.loads(target.read_text())
    receipt["cases"][-1]["passed"] = False
    c.h.module.save_json(target, receipt)
    with pytest.raises(ValueError):
        c.module.LogisticsFinalCasesBudget(c.h.manifest, c.h.identity["source_revision"])
    assert not (c.h.workspace / "budget-C2.json").exists()


def test_source_handoff_rejects_unfinished_C2(final_cases, monkeypatch):
    c = final_cases
    c.ledger.prepare()
    monkeypatch.setattr(c.h.module, "source_revision", lambda: "d" * 40)
    with pytest.raises(ValueError, match="C2 must be terminal"):
        c.h.module.rebind_logistics_source(c.h.manifest, previous_revision=c.h.identity["source_revision"],
            previous_source_root=c.h.identity["source_root"], allow_terminal_budget=True)


def test_C1_C2_modes_are_mutually_exclusive(prepared):
    module = api("scripts.smoke_logistics_model")
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.run_logistics_model_cases(prepared.manifest, prepared.identity["source_revision"], prepared.workspace / "bad",
            execute_approved_budget=True, continuation_c1=True, continuation_c2=True)


def test_C2_controller_passes_explicit_mode_to_worker_and_preserves_history(final_cases, monkeypatch):
    from contextlib import contextmanager

    c = final_cases
    children, cases = [], []

    @contextmanager
    def child(*args, role, **kwargs):
        children.append((role, kwargs.get("extra", ())))
        yield

    class Execute:
        def __init__(self, *args, **kwargs):
            assert kwargs["conversation"] is None  # All three cases start fresh.

        def __call__(self, case):
            cases.append(case["id"])
            return {"offline_mode_probe": case["id"]}

    validate = c.module.validate_model_case

    def check(case, evidence):
        if case["id"] in ("L1", "L2", "L3"):
            validate(case, evidence)
        else:
            assert evidence == {"offline_mode_probe": case["id"]}

    monkeypatch.setattr(c.module, "validate_export_origin", lambda: None)
    monkeypatch.setattr(c.module, "validate_model_prerequisites", lambda *args: {})
    monkeypatch.setattr(c.module, "snapshot_logistics_protection", lambda **kwargs: {"offline": "unchanged"})
    monkeypatch.setattr(c.module, "logistics_child", child)
    monkeypatch.setattr(c.module, "LiveModelCases", Execute)
    monkeypatch.setattr(c.module, "validate_model_case", check)
    report = c.module.run_logistics_model_cases(c.h.manifest, c.h.identity["source_revision"], c.h.workspace / "mode-probe",
        execute_approved_budget=True, continuation_c2=True)
    assert cases == ["L4", "L5", "L6"]
    assert children[-1] == ("resolv", ("--real-worker", "--execute-approved-budget", "--continuation-C2"))
    assert report["format"] == "logistics-model-C2-v1" and report["status"] == "passed"
    assert report["cumulative_model_http_calls"] == 14  # Wiring probe sends nothing.
    assert all(p.read_bytes() == raw for p, raw in c.preserved.items())
