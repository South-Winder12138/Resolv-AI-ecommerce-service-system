"""C1 spends only unused cases; all network sends below use MockTransport."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.test_logistics_contracts import api
from tests.test_logistics_gate_support import prepared

__all__ = ["prepared"]


@pytest.fixture
def continuation(prepared):
    module = api("scripts.smoke_logistics_model")
    fixture = json.loads((Path(__file__).parent / "fixtures/logistics_l1_receipt.json").read_text())
    parent = {"state": "closed", "source_revision": fixture["source_revision"], "max_model_calls": 48,
        "http_calls": 5, "retrievals": 0, "calls": fixture["evidence"]["model_calls"], "active_case": None,
        "cases": {"L1": {"state": "failed"}, **{f"L{i}": {"state": "skipped", "reason": "first_unexpected_failure"} for i in range(2, 7)}}}
    report = {"status": "failed", "source_revision": fixture["source_revision"], "budget": parent}
    root = prepared.workspace
    (root / "model-once").mkdir()
    prepared.module.save_json(root / "budget.json", parent)
    prepared.module.save_json(root / "model-once/report.json", report)
    fixture["original_report_sha256"] = hashlib.sha256((root / "model-once/report.json").read_bytes()).hexdigest()
    prepared.module.save_json(root / "L1-adjudication.json", fixture)
    paths = [root / "budget.json", root / "model-once/report.json", root / "L1-adjudication.json"]
    assert hasattr(module, "LogisticsContinuationBudget"), "missing independent C1 budget"
    ledger = module.LogisticsContinuationBudget(prepared.manifest, prepared.identity["source_revision"])
    return SimpleNamespace(module=module, ledger=ledger, parent_bytes={p: p.read_bytes() for p in paths}, h=prepared)


def test_C1_cannot_replay_L1_and_parent_is_never_reopened(continuation):
    c = continuation
    c.ledger.prepare()
    c.ledger.open()
    with pytest.raises(ValueError):
        c.ledger.begin_case("L1")
    c.ledger.begin_case("L2")
    assert [r["id"] for r in c.ledger.cases] == ["L2", "L3", "L4", "L5", "L6"]
    assert all(p.read_bytes() == b for p, b in c.parent_bytes.items())


def test_C1_caps_forty_new_sends_and_cumulative_forty_five(continuation):
    c = continuation
    c.ledger.prepare()
    c.ledger.open()
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with c.module.provider_http_guard(c.ledger, "https://provider.invalid/v1"):
        for case in c.ledger.cases:
            c.ledger.begin_case(case["id"])
            for purpose in ("resolution", "intents", *(["business"] * 6)):
                with c.module.model_permit(c.ledger.reserve(purpose)):
                    http.post("https://provider.invalid/v1/chat/completions", json={})
            with pytest.raises(ValueError):
                c.ledger.reserve("business")
            c.ledger.finish_case(case["id"], passed=True)
    assert c.ledger.read()["http_calls"] == 40
    cumulative = c.ledger.read()["http_calls"] + 5
    assert cumulative == 45 and cumulative <= 48
    with pytest.raises(ValueError):
        c.ledger.open()
    with pytest.raises(ValueError):
        c.ledger.prepare()
    assert all(p.read_bytes() == b for p, b in c.parent_bytes.items())


def test_C1_first_failure_closes_all_remaining_cases(continuation):
    ledger = continuation.ledger
    ledger.prepare()
    ledger.open()
    rows = continuation.module.execute_fixed_cases(ledger, lambda case: (_ for _ in ()).throw(ValueError("test failure")))
    assert [r["case"] for r in rows] == ["L2"]
    assert ledger.read()["state"] == "closed" and ledger.read()["cases"]["L3"]["state"] == "skipped"


@pytest.mark.parametrize("target", ["budget.json", "model-once/report.json", "L1-adjudication.json"])
def test_C1_changed_parent_or_receipt_blocks_before_send(continuation, target):
    c = continuation
    c.ledger.prepare()
    c.ledger.open()
    c.ledger.begin_case("L2")
    permit = c.ledger.reserve("resolution")
    path = c.h.workspace / target
    path.write_bytes(path.read_bytes() + b"\n")
    sent = []
    http = httpx.Client(transport=httpx.MockTransport(lambda request: sent.append(True) or httpx.Response(200, json={})))
    with c.module.provider_http_guard(c.ledger, "https://provider.invalid/v1"), c.module.model_permit(permit), pytest.raises(ValueError):
        http.post("https://provider.invalid/v1/chat/completions", json={})
    assert sent == []


def test_C1_wrong_receipt_rejected_before_new_ledger_is_created(prepared):
    module = api("scripts.smoke_logistics_model")
    assert hasattr(module, "LogisticsContinuationBudget"), "missing C1 receipt preflight"
    with pytest.raises((ValueError, FileNotFoundError)):
        module.LogisticsContinuationBudget(prepared.manifest, prepared.identity["source_revision"])
    assert not (prepared.workspace / "budget-C1.json").exists()


def test_L2_requires_the_persisted_L1_conversation(continuation):
    c = continuation
    assert hasattr(c.module, "validate_persisted_l1"), "missing original conversation preflight"
    with pytest.raises(ValueError):
        c.module.validate_persisted_l1(c.h.manifest, c.ledger)


def test_L2_reuses_only_matching_durable_L1_answer_and_observation(continuation):
    from resolv.db.engine import create_session_factory
    from resolv.models.conversation import (
        Conversation,
        ConversationAction,
        ConversationMessage,
        ConversationRun,
    )
    from resolv.models.logistics import ConversationLogisticsAttempt
    from resolv.repositories.conversation import fingerprint

    c = continuation
    evidence = c.ledger.previous_evidence
    factory = create_session_factory(c.h.engine)
    profile = c.h.module._read_json(c.h.workspace / "profile.json")
    with factory.begin() as session:
        session.add(Conversation(conversation_id="original-L1", owner_id=c.ledger.identity["owner_id"], thread_id="original-L1-thread"))
        session.flush()
        session.add(ConversationRun(run_id=evidence["run_id"], conversation_id="original-L1", request_id="original-L1", request_fingerprint="f" * 64,
            question_message_id="original-question", answer_message_id="original-answer", status="completed",
            business_config={"logistics_backend": profile}, model_audits=evidence["business_audits"], result_json={"branch": "business"}))
        session.flush()
        session.add(ConversationMessage(message_id="original-answer", conversation_id="original-L1", run_id=evidence["run_id"], role="assistant", seq=1, text=evidence["answer"]))
        action_ids = []
        for i, action in enumerate(evidence["actions"], 1):
            action_id = "original-action-" + str(i)
            action_ids.append(action_id)
            session.add(ConversationAction(action_id=action_id, run_id=evidence["run_id"], model_step=i, tool_call_id="call-" + str(i),
                arguments_fingerprint=fingerprint(action["arguments"]), **action))
        session.flush()
        proof = evidence["attempts"][0]
        session.add(ConversationLogisticsAttempt(attempt_id=proof["attempt_id"], action_id=action_ids[-1], run_id=evidence["run_id"],
            conversation_id="original-L1", owner_id=c.ledger.identity["owner_id"], execution_epoch=1, attempt_no=1,
            config_hash="f" * 64, arguments_hash="e" * 64, tracking_no="MOCKSF001", state="completed", envelope=proof["envelope"], result_hash=fingerprint(proof["envelope"])))
    assert c.module.validate_persisted_l1(c.h.manifest, c.ledger) == "original-L1"
    with factory.begin() as session:
        session.get(ConversationMessage, "original-answer").text = "different saved answer"
    with pytest.raises(ValueError, match="answer/audit changed"):
        c.module.validate_persisted_l1(c.h.manifest, c.ledger)
    with factory.begin() as session:
        session.get(ConversationMessage, "original-answer").text = evidence["answer"]
        session.add(ConversationMessage(message_id="unexpected-message", conversation_id="original-L1", run_id=evidence["run_id"], role="user", seq=2, text="查询另一笔订单"))
    with pytest.raises(ValueError, match="latest L1 turn changed"):
        c.module.validate_persisted_l1(c.h.manifest, c.ledger)


def test_source_handoff_rejects_active_C1(continuation, monkeypatch):
    c = continuation
    c.ledger.prepare()
    c.ledger.open()
    before = c.h.manifest.read_bytes()
    monkeypatch.setattr(c.h.module, "source_revision", lambda: "c" * 40)
    with pytest.raises(ValueError, match="C1 must be terminal"):
        c.h.module.rebind_logistics_source(c.h.manifest, previous_revision=c.h.identity["source_revision"],
            previous_source_root=c.h.identity["source_root"], allow_terminal_budget=True)
    assert c.h.manifest.read_bytes() == before


def test_terminal_source_handoff_preserves_budget_and_does_not_reopen_it(prepared, monkeypatch):
    module = prepared.module
    module.save_json(prepared.workspace / "budget.json", {"state": "closed", "calls": [], "retrievals": 0, "http_calls": 0})
    before = (prepared.workspace / "budget.json").read_bytes()
    previous = prepared.identity["source_revision"]
    monkeypatch.setattr(module, "source_revision", lambda: "d" * 40)
    result = module.rebind_logistics_source(prepared.manifest, previous_revision=previous,
        previous_source_root=prepared.identity["source_root"], allow_terminal_budget=True)
    assert result["source_revision"] == "d" * 40
    assert (prepared.workspace / "budget.json").read_bytes() == before


@pytest.mark.parametrize("case_id,answer", [("L4", "未查询到该运单的物流信息。"), ("L4", "没有找到该运单。"), ("L4", "找不到该运单。"), ("L5", "订单仍在处理中，目前尚无运单号。")])
def test_acceptance_recognizes_equivalent_not_found_and_no_tracking_phrases(case_id, answer):
    module, case, evidence = result_evidence(case_id, answer)
    module.validate_model_case(case, evidence)


def result_evidence(case_id, answer):
    import copy

    module = api("scripts.smoke_logistics_model")
    evidence = copy.deepcopy(json.loads((Path(__file__).parent / "fixtures/logistics_l1_receipt.json").read_text())["evidence"])
    case = next(c for c in json.loads(module.CASES_PATH.read_text())["cases"] if c["id"] == case_id)
    evidence["answer"] = "【模拟物流】" + answer
    evidence["business_audits"] = [evidence["business_audits"][1 if case_id == "L4" else 0], evidence["business_audits"][-1]]
    evidence["model_calls"] = evidence["model_calls"][:2] + evidence["model_calls"][-2:]
    if case_id == "L4":
        evidence["actions"] = evidence["actions"][-1:]
        evidence["actions"][0]["arguments"]["tracking_no"] = "MOCKSF404"
        evidence["attempts"][0]["envelope"].update(tracking_no="MOCKSF404", outcome="not_found", shipment=None)
    else:
        evidence.update(mcp_calls=0, api_requests=0, attempts=[])
        evidence["actions"] = evidence["actions"][:1]
        evidence["actions"][0]["arguments"]["order_id"] = "MOCK002"
        evidence["actions"][0]["result"]["tracking_no"] = None
    return module, case, evidence


@pytest.mark.parametrize("case_id,answer", [
    ("L4", "暂无更新，包裹已签收。"),
    ("L4", "没有取消订单，正在运输中。"),
    ("L4", "没有找到该运单，但包裹已送达。"),
    ("L5", "没有取消订单，包裹已经发货。"),
    ("L5", "暂无新消息，包裹已送达。"),
    ("L5", "尚无运单号，但已签收。"),
])
def test_no_result_requires_relevant_missing_fact_without_invented_status(case_id, answer):
    module, case, evidence = result_evidence(case_id, answer)
    with pytest.raises(ValueError):
        module.validate_model_case(case, evidence)


@pytest.mark.parametrize("case_id,answer", [
    ("L4", "查询结果显示，运单号 MOCKSF404 不存在或暂无物流记录。请检查运单号是否输入正确，或确认该订单是否已发货。如果您有对应的订单号，我可以先通过订单号查询正确的运单信息。"),
    ("L4", "没有找到该运单。请确认订单是否已发货。"),
    ("L5", "该订单尚无运单号，请联系商家确认是否已发货。"),
    ("L4", "没有找到该运单，如果已发货请联系商家核对运单号。"),
])
def test_no_result_does_not_treat_shipping_verification_questions_as_observations(case_id, answer):
    module, case, evidence = result_evidence(case_id, answer)
    module.validate_model_case(case, evidence)


@pytest.mark.parametrize("answer", [
    "没有找到该运单。请确认是否已发货，但包裹已送达。",
    "没有找到该运单。是否已发货？目前包裹已签收。",
])
def test_shipping_questions_cannot_hide_a_separate_invented_observation(answer):
    module, case, evidence = result_evidence("L4", answer)
    with pytest.raises(ValueError):
        module.validate_model_case(case, evidence)
