from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tests.test_logistics_contracts import api
from tests.test_logistics_gate_support import prepared

__all__ = ["prepared"]


def budget(h):
    module = api("scripts.smoke_logistics_model")
    ledger = module.LogisticsModelBudget(h.manifest, h.identity["source_revision"])
    ledger.open()
    return module, ledger


def test_budget_reserves_before_send_and_counts_failure(prepared):
    module, ledger = budget(prepared)
    ledger.begin_case("L1")
    permit = ledger.reserve("resolution")
    assert ledger.read()["calls"][0]["state"] == "reserved"
    observed = []

    def fail(request):
        observed.append(ledger.read()["calls"][0]["state"])
        raise httpx.ConnectError("private provider unavailable")

    client = httpx.Client(transport=httpx.MockTransport(fail))
    with module.provider_http_guard(ledger, "https://provider.invalid/v1"), module.model_permit(permit), pytest.raises(httpx.ConnectError):
        client.post("https://provider.invalid/v1/chat/completions", json={})
    assert observed == ["http_started"]
    call = ledger.read()["calls"][0]
    assert call["state"] == "transport_failed" and call["http_sends"] == 1
    assert "private provider" not in str(ledger.read())


def test_hidden_second_send_and_unreserved_provider_call_are_rejected(prepared):
    module, ledger = budget(prepared)
    ledger.begin_case("L1")
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    permit = ledger.reserve("resolution")
    with module.provider_http_guard(ledger, "https://provider.invalid/v1"):
        with pytest.raises(ValueError):
            client.post("https://provider.invalid/v1/chat/completions", json={})
        with module.model_permit(permit):
            client.post("https://provider.invalid/v1/chat/completions", json={})
            with pytest.raises(ValueError):
                client.post("https://provider.invalid/v1/chat/completions", json={})
    assert ledger.read()["http_calls"] == 1


def test_first_unexpected_failure_closes_remaining_cases(prepared):
    _, ledger = budget(prepared)
    ledger.begin_case("L1")
    ledger.finish_case("L1", passed=False)
    assert ledger.read()["state"] == "closed"
    assert ledger.read()["cases"]["L2"]["state"] == "skipped"
    with pytest.raises(ValueError):
        ledger.begin_case("L2")


def test_no_reuse_of_terminal_ledger(prepared):
    module, ledger = budget(prepared)
    ledger.begin_case("L1")
    ledger.finish_case("L1", passed=False)
    with pytest.raises(ValueError):
        module.LogisticsModelBudget(prepared.manifest, prepared.identity["source_revision"]).open()


def test_concurrent_budget_reservations_allow_one_resolution(prepared):
    _, ledger = budget(prepared)
    ledger.begin_case("L1")

    def reserve(_):
        try:
            return ledger.reserve("resolution")
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        permits = list(pool.map(reserve, range(4)))
    assert sum(p is not None for p in permits) == 1 and len(ledger.read()["calls"]) == 1


def test_budget_caps_48_actual_sends_and_rejects_forbidden_purposes(prepared):
    module, ledger = budget(prepared)
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with module.provider_http_guard(ledger, "https://provider.invalid/v1"):
        for index in range(1, 7):
            ledger.begin_case(f"L{index}")
            for purpose in ["resolution", "intents", *(["business"] * 6)]:
                with module.model_permit(ledger.reserve(purpose)):
                    assert client.post("https://provider.invalid/v1/chat/completions", json={}).status_code == 200
            with pytest.raises(ValueError):
                ledger.reserve("business")
            for purpose in ("summary", "knowledge_query", "refund_decision", "retrieval"):
                with pytest.raises(ValueError):
                    ledger.reserve(purpose)
            ledger.finish_case(f"L{index}", passed=True)
    assert ledger.read()["http_calls"] == 48 and ledger.read()["state"] == "completed"
    with pytest.raises(ValueError):
        ledger.reserve("business")


def test_wrong_route_stops_before_retrieval(prepared):
    from tests.understanding_fixtures import make_classification

    module, ledger = budget(prepared)
    ledger.begin_case("L1")

    class WrongRoute:
        def invoke(self, stage, *args, **kwargs):
            return make_classification(["商品咨询"], ["product_consultation"]), object()

    guard = module.LogisticsModelGuard(ledger).wrap("understanding", WrongRoute())
    with pytest.raises(ValueError, match="route"):
        guard.invoke("intents")
    assert ledger.read()["http_calls"] == 0


def test_case_order_and_once_only_are_enforced(prepared):
    _, ledger = budget(prepared)
    with pytest.raises(ValueError):
        ledger.begin_case("L2")
    ledger.begin_case("L1")
    with pytest.raises(ValueError):
        ledger.begin_case("L1")
    ledger.finish_case("L1", passed=True)
    with pytest.raises(ValueError):
        ledger.begin_case("L1")


def test_runner_refuses_without_explicit_execution_flag(tmp_path):
    module = api("scripts.smoke_logistics_model")
    with pytest.raises(ValueError, match="execute-approved-budget"):
        module.run_logistics_model_cases(tmp_path / "foreign.json", "a" * 40, tmp_path / "report")


def test_fixed_case_runner_stops_after_first_unexpected_failure(prepared):
    module, ledger = budget(prepared)
    called = []

    def failure(case):
        called.append(case["id"])
        raise ValueError("fixed case failed")

    assert hasattr(module, "execute_fixed_cases"), "missing bounded case executor"
    rows = module.execute_fixed_cases(ledger, failure)
    assert called == ["L1"] and rows[0]["passed"] is False
    assert ledger.read()["state"] == "closed"


@pytest.mark.parametrize("code", ["logistics_transport_failed", "model_unavailable", None])
def test_l6_cannot_treat_other_failures_as_expected_timeout(code):
    module = api("scripts.smoke_logistics_model")
    assert hasattr(module, "validate_model_case"), "missing layered model acceptance"
    case = {"id": "L6", "error_code": "logistics_timeout", "tracking_no": "MOCKSF003", "mcp_calls": 1}
    evidence = {"error_code": code, "run_status": "interrupted", "assistant_count": 0, "mcp_calls": 1, "api_requests": 1}
    with pytest.raises(ValueError):
        module.validate_model_case(case, evidence)


def test_actual_openai_httpx2_send_is_guarded(prepared):
    import httpx2

    module, ledger = budget(prepared)
    ledger.begin_case("L1")
    client = httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json={})))
    with module.provider_http_guard(ledger, "https://provider.invalid/v1"), module.model_permit(ledger.reserve("resolution")):
        client.post("https://provider.invalid/v1/chat/completions", json={})
    assert ledger.read()["http_calls"] == 1


def test_live_collector_reads_saved_message_and_replays_sse_offline(prepared, monkeypatch):
    from contextlib import contextmanager

    import httpx2
    from fastapi.testclient import TestClient

    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.test_logistics_demo import DemoClient

    module, ledger = budget(prepared)
    ledger.begin_case("L1")
    demo = api("resolv.services.logistics_demo")
    original = prepared.module.owned_logistics_resources

    @contextmanager
    def resources(path):
        with original(path) as owned:
            owned.checkpoint_factory = MemoryCheckpoints()
            yield owned

    remote = DemoClient()
    remote.profile = prepared.module.owned_profile(prepared.manifest)
    monkeypatch.setattr(demo, "owned_logistics_resources", resources)
    monkeypatch.setattr(demo, "MCPLogisticsClient", lambda profile: remote)
    client_type = httpx2.Client
    with TestClient(demo.create_logistics_demo_app(prepared.manifest)) as http:
        def dispatch(request):
            response = http.request(request.method, request.url.path, content=request.content, headers={"content-type": "application/json"})
            return httpx2.Response(response.status_code, content=response.content)

        monkeypatch.setattr(module.httpx2, "Client", lambda **kwargs: client_type(transport=httpx2.MockTransport(dispatch)))
        evidence = module.LiveModelCases(prepared.manifest, ledger, prepared.workspace)(ledger.cases[0])
    assert evidence["answer"].startswith("【模拟物流】")
    assert evidence["replay_equal"] is True and evidence["sse_equal"] is True
    assert evidence["assistant_count"] == 1 and evidence["refresh_extra_calls"] == 0
    assert evidence["understanding_status"] == "ready" and evidence["intent"] == "logistics"


def test_timeout_collector_accepts_existing_interrupted_snapshot_protocol(prepared, monkeypatch):
    from contextlib import contextmanager

    import httpx2
    from fastapi.testclient import TestClient

    from resolv.schemas.logistics import LogisticsFailure
    from tests.conversation_fixtures import MemoryCheckpoints

    module, ledger = budget(prepared)
    ledger.begin_case("L1")  # No actual model sends; the collector uses fake models.
    demo = api("resolv.services.logistics_demo")
    original = prepared.module.owned_logistics_resources

    @contextmanager
    def resources(path):
        with original(path) as owned:
            owned.checkpoint_factory = MemoryCheckpoints()
            yield owned

    class TimeoutClient:
        profile = prepared.module.owned_profile(prepared.manifest)

        def __init__(self):
            self.calls = []

        def query(self, tracking, *, request_id):
            self.calls.append(tracking)
            raise LogisticsFailure("logistics_timeout", True)

    remote = TimeoutClient()
    monkeypatch.setattr(demo, "owned_logistics_resources", resources)
    monkeypatch.setattr(demo, "MCPLogisticsClient", lambda profile: remote)
    client_type = httpx2.Client
    with TestClient(demo.create_logistics_demo_app(prepared.manifest)) as http:
        def dispatch(request):
            response = http.request(request.method, request.url.path, content=request.content, headers={"content-type": "application/json"})
            return httpx2.Response(response.status_code, content=response.content)

        monkeypatch.setattr(module.httpx2, "Client", lambda **kwargs: client_type(transport=httpx2.MockTransport(dispatch)))
        evidence = module.LiveModelCases(prepared.manifest, ledger, prepared.workspace)(ledger.cases[-1])
    assert evidence["error_code"] == "logistics_timeout" and evidence["http_status"] == 503
    assert evidence["run_status"] == "interrupted" and evidence["assistant_count"] == 0
    assert evidence["replay_equal"] is True and evidence["sse_equal"] is True
    assert evidence["refresh_extra_calls"] == 0 and remote.calls == ["MOCKSF003"]
    proof = prepared.module._read_json(prepared.workspace / "L6-replay.json")
    assert proof["replay"]["status"] == 200 and proof["replay"]["body"] == {"status": "interrupted", "run_id": evidence["run_id"]}


@pytest.mark.parametrize("mutation", ["wrong_run", "wrong_state", "message", "missing_done"])
def test_interrupted_replay_cannot_pass_wrong_run_state_or_completion(mutation):
    module = api("scripts.smoke_logistics_model")
    assert hasattr(module, "model_replay_proof"), "missing existing replay contract verifier"
    import json

    evidence = {"run_id": "r", "run_status": "interrupted", "error_code": "logistics_timeout", "answer": None}
    frames = [{"type": "run_snapshot", "run_id": "r", "status": "interrupted"}, {"type": "done", "run_id": "r", "status": "interrupted", "snapshot": True}]
    if mutation == "wrong_run":
        frames[0]["run_id"] = "other"
    elif mutation == "wrong_state":
        frames[0]["status"] = "completed"
    elif mutation == "message":
        frames.append({"type": "message", "run_id": "r", "text": "退款成功"})
    else:
        frames.pop()
    proof = {"initial": {"status": 503, "body": {"detail": {"code": "logistics_timeout", "run_id": "r", "retryable": True}}},
        "detail": {"status": 200, "body": {"messages": []}}, "replay": {"status": 200, "body": {"status": "interrupted", "run_id": "r"}},
        "sse": {"status": 200, "text": "\n".join("data: " + json.dumps(frame) for frame in frames)}}
    assert module.model_replay_proof(evidence, proof)["sse_equal"] is False


def test_live_runner_requires_export_before_opening_budget(prepared, monkeypatch):
    module = api("scripts.smoke_logistics_model")
    # Model an unexported tree explicitly, also when this test itself is exported.
    monkeypatch.setattr(module, "SOURCE_ROOT", prepared.workspace)
    with pytest.raises(ValueError, match="immutable export"):
        module.run_logistics_model_cases(prepared.manifest, prepared.identity["source_revision"], prepared.workspace / "live", execute_approved_budget=True)
    assert module.LogisticsModelBudget(prepared.manifest, prepared.identity["source_revision"]).read()["state"] == "unopened"


@pytest.mark.parametrize("library_name", ["httpx", "httpx2"])
def test_redirect_inside_send_cannot_hide_a_second_provider_request(prepared, library_name):
    import importlib

    module, ledger = budget(prepared)
    ledger.begin_case("L1")
    library = importlib.import_module(library_name)
    sent = []

    def redirect(request):
        sent.append(str(request.url))
        if len(sent) == 1:
            return library.Response(307, headers={"location": "/v1/redirected"})
        return library.Response(200, json={})

    client = library.Client(transport=library.MockTransport(redirect), follow_redirects=True)
    with module.provider_http_guard(ledger, "https://provider.invalid/v1"), module.model_permit(ledger.reserve("resolution")), pytest.raises(ValueError, match="Additional provider send"):
        client.post("https://provider.invalid/v1/chat/completions", json={})
    assert len(sent) == ledger.read()["http_calls"] == 1


def delivered_evidence():
    evidence = {"error_code": None, "mcp_calls": 1, "api_requests": 1, "understanding_status": "ready", "intent": "logistics",
        "refresh_extra_calls": 0, "replay_equal": True, "sse_equal": True, "versions": ["routed-v4", 4, 3, 2], "input_budget": 60000,
        "model_calls": [{"purpose": purpose, "http_sends": 1, "http_status": 200, "state": "http_completed", "invocation_outcome": "completed"} for purpose in ("resolution", "intents", "business", "business")],
        "business_audits": [{"tool_policy": {"phase": phase, "allowed_tool_names": names}, "returned_tool_names": names, "actual_input": 100, "actual_output": 50, "finish_reason": finish} for phase, names, finish in (("logistics", ["query_logistics"], "tool_calls"), ("answer", [], "stop"))],
        "actions": [{"tool_name": "query_logistics", "effect": "read", "arguments": {"tracking_no": "MOCKSF002"}}],
        "attempts": [{"state": "completed", "hash_verified": True, "envelope": {"tracking_no": "MOCKSF002", "outcome": "found", "observed_at": "2026-10-20T07:04:52Z", "shipment": {"status": "delivered", "signed_at": None, "shipped_at": "2026-10-19T10:00:00+08:00", "events": []}}}],
        "run_status": "completed", "assistant_count": 1, "http_status": 200, "reply_kind": "answer", "answer": "【模拟物流】该运单已签收，轨迹为空。"}
    return evidence


def test_delivered_answer_without_a_time_does_not_invent_unknown_time():
    module = api("scripts.smoke_logistics_model")
    case = next(c for c in __import__("json").loads(module.CASES_PATH.read_text())["cases"] if c["id"] == "L3")
    evidence = delivered_evidence()
    module.validate_model_case(case, evidence)
    # OpenAI-compatible streams can finish with stop while carrying validated native calls.
    evidence["business_audits"][0]["finish_reason"] = "stop"
    module.validate_model_case(case, evidence)
    evidence["answer"] = "【模拟物流】昨日已签收。"
    with pytest.raises(ValueError):
        module.validate_model_case(case, evidence)


@pytest.mark.parametrize("answer", [
    "【模拟物流】运单 MOCKSF002 的物流状态显示为“已签收”（delivered），发货时间为 2026-10-19 10:00。目前暂无详细的轨迹节点记录。",
    "【模拟物流】发货时间：2026年10月19日10:00；当前已签收，签收时间未知。",
    "【模拟物流】已签收。系统查询时间为2026-10-20 07:04:52 UTC。",
])
def test_l3_allows_verified_shipping_and_observation_times(answer):
    module = api("scripts.smoke_logistics_model")
    case = next(c for c in __import__("json").loads(module.CASES_PATH.read_text())["cases"] if c["id"] == "L3")
    evidence = delivered_evidence()
    evidence["answer"] = answer
    module.validate_model_case(case, evidence)


@pytest.mark.parametrize("answer", [
    "【模拟物流】已签收，签收时间为2026-10-19 10:00。",
    "【模拟物流】发货时间为2026-10-19 10:00，签收时间也为2026-10-19 10:00。",
    "【模拟物流】发货时间为2026-10-19 11:00，当前已签收。",
    "【模拟物流】今天已签收，发货时间为2026-10-19 10:00。",
    "【模拟物流】已签收，发货时间为2026-10-19 10:00，送达于2026-10-19 10:00。",
])
def test_l3_rejects_invented_or_misattributed_times(answer):
    module = api("scripts.smoke_logistics_model")
    case = next(c for c in __import__("json").loads(module.CASES_PATH.read_text())["cases"] if c["id"] == "L3")
    evidence = delivered_evidence()
    evidence["answer"] = answer
    with pytest.raises(ValueError):
        module.validate_model_case(case, evidence)


@pytest.mark.parametrize("answer", [
    "【模拟物流】运单 MOCKSF002 尚未签收。",
    "【模拟物流】运单 MOCKSF002 已于上周一签收。",
    "【模拟物流】运单 MOCKSF002 已签收，签收地点是北京。",
    "【模拟物流】运单 MOCKSF002 已签收，轨迹显示包裹先后经过广州和深圳。",
])
def test_l3_cannot_pass_negated_status_or_unknown_location_and_events(answer):
    module = api("scripts.smoke_logistics_model")
    case = next(c for c in __import__("json").loads(module.CASES_PATH.read_text())["cases"] if c["id"] == "L3")
    evidence = delivered_evidence()
    evidence["answer"] = answer
    with pytest.raises(ValueError):
        module.validate_model_case(case, evidence)


@pytest.mark.parametrize("answer", [
    "【模拟物流】包裹不在上海，也没有在运输中。",
    "【模拟物流】包裹正在运输，目前位于上海；预计今天送达。",
])
def test_l1_keywords_cannot_prove_negated_or_invented_facts(answer):
    import json

    module = api("scripts.smoke_logistics_model")
    evidence = json.loads((__import__("pathlib").Path(__file__).parent / "fixtures/logistics_l1_receipt.json").read_text())["evidence"]
    case = next(c for c in json.loads(module.CASES_PATH.read_text())["cases"] if c["id"] == "L1")
    evidence["answer"] = answer
    with pytest.raises(ValueError):
        module.validate_model_case(case, evidence)


def test_budget_created_before_source_handoff_cannot_open_after_it(prepared, monkeypatch):
    module = api("scripts.smoke_logistics_model")
    previous = prepared.identity["source_revision"]
    stale = module.LogisticsModelBudget(prepared.manifest, previous)
    monkeypatch.setattr(prepared.module, "source_revision", lambda: "b" * 40)
    prepared.module.rebind_logistics_source(prepared.manifest, previous_revision=previous, previous_source_root=prepared.identity["source_root"])
    with pytest.raises(ValueError):
        stale.open()
    assert stale.read()["state"] == "unopened"
