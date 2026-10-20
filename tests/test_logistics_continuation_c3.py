"""The approved harness exception never replays L4 or reopens prior ledgers."""

import json
from hashlib import sha256
from types import SimpleNamespace

import httpx
import pytest

from tests.test_logistics_continuation import result_evidence
from tests.test_logistics_continuation_c2 import final_cases
from tests.test_logistics_gate_support import prepared

__all__ = ["final_cases", "prepared"]


@pytest.fixture
def last_two(final_cases):
    c = final_cases
    root = c.h.workspace
    _, _, evidence = result_evidence("L4", "没有找到该运单，请确认是否已发货。")
    revision = "c" * 40
    for index, call in enumerate(evidence["model_calls"]):
        call.update(case_id="L4", revision=revision, id=sha256(f"L4:{index}".encode()).hexdigest()[:32])
    parent = {"state": "closed", "source_revision": revision, "max_model_calls": 24, "retrievals": 0,
        "http_calls": 4, "calls": evidence["model_calls"], "parent_hashes": c.ledger.references,
        "cases": {"L4": {"state": "failed"}, "L5": {"state": "skipped"}, "L6": {"state": "skipped"}}}
    (root / "model-C2").mkdir()
    c.h.module.save_json(root / "budget-C2.json", parent)
    c.h.module.save_json(root / "model-C2/report.json", {"source_revision": revision, "status": "failed", "budget": parent,
        "cumulative_model_http_calls": 18, "cases": [{"case": "L4", "passed": False, "evidence": evidence}]})
    c.h.module.save_json(root / "L4-adjudication.json", {"format": "logistics-case-adjudication-v1", "case": "L4", "status": "passed_from_existing_receipt", "new_model_calls": 0, "model_calls": 4,
        "original_report_sha256": sha256((root / "model-C2/report.json").read_bytes()).hexdigest(), "evidence": evidence})
    assert hasattr(c.module, "LogisticsLastTwoCasesBudget"), "missing approved last-two budget"
    ledger = c.module.LogisticsLastTwoCasesBudget(c.h.manifest, c.h.identity["source_revision"])
    return SimpleNamespace(module=c.module, ledger=ledger, h=c.h,
        preserved={**c.preserved, **{p: p.read_bytes() for p in (root / "budget-C2.json", root / "model-C2/report.json", root / "L4-adjudication.json")}})


def test_C3_only_L5_L6_and_sixteen_requests_with_thirty_four_cumulative(last_two):
    c = last_two
    c.ledger.prepare()
    c.ledger.open()
    assert [r["id"] for r in c.ledger.cases] == ["L5", "L6"]
    assert c.ledger.read()["max_model_calls"] == 16
    assert c.ledger.continuation_flag == "--continuation-C3"
    assert c.ledger.report_format == "logistics-model-C3-v1"
    for case in ("L1", "L2", "L3", "L4", "L6"):
        with pytest.raises(ValueError):
            c.ledger.begin_case(case)
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with c.module.provider_http_guard(c.ledger, "https://provider.invalid/v1"):
        for case in ("L5", "L6"):
            c.ledger.begin_case(case)
            for purpose in ("resolution", "intents", *(["business"] * 6)):
                with c.module.model_permit(c.ledger.reserve(purpose)):
                    http.post("https://provider.invalid/v1/chat/completions", json={})
            with pytest.raises(ValueError):
                c.ledger.reserve("business")
            c.ledger.finish_case(case, passed=True)
    assert c.ledger.read()["http_calls"] == 16
    assert c.ledger.parent_http_calls + c.ledger.read()["http_calls"] == 34
    assert all(p.read_bytes() == raw for p, raw in c.preserved.items())
    for operation in (c.ledger.open, c.ledger.prepare):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("target", ["budget-C2.json", "model-C2/report.json", "L4-adjudication.json"])
def test_C3_changed_L4_reference_blocks_before_provider_send(last_two, target):
    c = last_two
    c.ledger.prepare()
    c.ledger.open()
    c.ledger.begin_case("L5")
    permit = c.ledger.reserve("resolution")
    path = c.h.workspace / target
    path.write_bytes(path.read_bytes() + b"\n")
    sends = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: sends.append(True) or httpx.Response(200, json={})))
    with c.module.provider_http_guard(c.ledger, "https://provider.invalid/v1"), c.module.model_permit(permit), pytest.raises(ValueError):
        client.post("https://provider.invalid/v1/chat/completions", json={})
    assert sends == []


def test_C3_requires_validated_original_L4_and_does_not_rewrite_it(last_two):
    c = last_two
    path = c.h.workspace / "L4-adjudication.json"
    receipt = json.loads(path.read_text())
    receipt["evidence"]["answer"] = "【模拟物流】未找到运单，但包裹已送达。"
    c.h.module.save_json(path, receipt)
    with pytest.raises(ValueError):
        c.module.LogisticsLastTwoCasesBudget(c.h.manifest, c.h.identity["source_revision"])
    assert not (c.h.workspace / "budget-C3.json").exists()


def test_C3_real_failure_stops_L6(last_two):
    c = last_two
    c.ledger.prepare()
    c.ledger.open()
    rows = c.module.execute_fixed_cases(c.ledger, lambda case: (_ for _ in ()).throw(ValueError("offline failure")))
    assert [r["case"] for r in rows] == ["L5"]
    assert c.ledger.read()["state"] == "closed" and c.ledger.read()["cases"]["L6"]["state"] == "skipped"


def test_C3_unfinished_budget_prevents_source_handoff(last_two, monkeypatch):
    c = last_two
    c.ledger.prepare()
    monkeypatch.setattr(c.h.module, "source_revision", lambda: "e" * 40)
    with pytest.raises(ValueError, match="C3 must be terminal"):
        c.h.module.rebind_logistics_source(c.h.manifest, previous_revision=c.h.identity["source_revision"],
            previous_source_root=c.h.identity["source_root"], allow_terminal_budget=True)


def test_C3_C2_modes_cannot_be_combined(prepared):
    from tests.test_logistics_contracts import api

    module = api("scripts.smoke_logistics_model")
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.run_logistics_model_cases(prepared.manifest, prepared.identity["source_revision"], prepared.workspace / "bad",
            execute_approved_budget=True, continuation_c2=True, continuation_c3=True)
