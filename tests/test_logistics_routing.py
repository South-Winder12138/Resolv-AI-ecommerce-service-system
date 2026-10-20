"""Actual routed graphs and public HTTP use the explicitly injected backend."""

import inspect

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from resolv.models.conversation import ConversationAction
from resolv.models.logistics import ConversationLogisticsAttempt as Attempt
from resolv.schemas.logistics import LogisticsFailure
from resolv.services.conversation import ConversationService
from tests.branch_fixtures import branch_factory, make_routed_service
from tests.conversation_fixtures import MemoryCheckpoints
from tests.test_conversation_graph import Model
from tests.test_logistics_execution import Client
from tests.test_logistics_mcp_server import profile
from tests.test_logistics_policy import config
from tests.test_routing_graph import decision, tool
from tests.understanding_fixtures import make_classification, make_resolution

__all__ = ["branch_factory"]


def logistics_call():
    return {"id": "shipment", "name": "query_logistics", "args": {"tracking_no": "SF001"}}


def harness(factory, business, *, responses=None, text="查询订单 MH001 的物流", mode="agent", fault_hook=None, client=None):
    assert "logistics_read_service" in inspect.signature(ConversationService).parameters, "missing injected logistics runtime"
    from resolv.services.logistics_execution import LogisticsReadService

    model = Model(responses if responses is not None else [decision(tool()), decision(logistics_call()), decision(text="当前在上海转运中心。")])
    service = make_routed_service(factory, MemoryCheckpoints(), business_model=model, business_factory=business, stage_outputs=[make_resolution(resolved_text=text), make_classification([text], ["logistics"])])
    service.business_run_config = config()
    remote = client or Client()
    service.logistics_read_service = LogisticsReadService(factory, remote, profile(), fault_hook=fault_hook)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "req-logistics", text, mode)
    return service, run, model, remote


def test_routed_logistics_uses_injected_client_not_local_shipment(branch_factory, tool_database):
    service, run, model, remote = harness(branch_factory, tool_database)
    events = list(service.run(run.run_id))
    saved = service.get_run(run.run_id)
    assert saved.status == "completed", events
    assert len(remote.calls) == 1 and remote.calls[0][0] == "SF001"
    tools = [e.data["result"] for e in events if e.type == "tool_result"]
    assert tools[-1]["location"] == "上海转运中心" and tools[-1]["provenance"]["simulated"] is True
    assert service.detail(run.conversation_id).messages[-1].text == "【模拟物流】当前在上海转运中心。"
    assert next(e.data["text"] for e in events if e.type == "message") == "【模拟物流】当前在上海转运中心。"
    assert len(model.calls) == 3
    assert [a["tool_policy"]["phase"] for a in saved.model_audits] == ["order", "logistics", "answer"]


def test_replay_get_and_same_request_id_do_not_connect(branch_factory, tool_database):
    service, run, _, remote = harness(branch_factory, tool_database)
    list(service.run(run.run_id))
    before = len(remote.calls)
    for _ in range(2):
        list(service.snapshot_events(run.run_id))
        service.detail(run.conversation_id)
        same, created = service.accept(run.conversation_id, run.request_id, run.question, "agent")
        assert not created and same.run_id == run.run_id
    assert len(remote.calls) == before == 1


def test_resume_after_result_commit_uses_original_tool_policy(branch_factory, tool_database):
    once = []

    def crash(stage, attempt):
        if stage == "after_result_commit" and not once:
            once.append(attempt.attempt_id)
            raise RuntimeError("private interrupted checkpoint")

    service, run, model, remote = harness(branch_factory, tool_database, fault_hook=crash)
    # A real process exit is exercised by Task9; an Exception after committed
    # results is reconciled here without a second network call.
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert len(remote.calls) == 1 and len(model.calls) == 3 and once
    list(service.resume(run.conversation_id, run.run_id))
    assert len(remote.calls) == 1


def test_chat_never_connects(branch_factory, tool_database):
    service, run, model, remote = harness(branch_factory, tool_database, mode="chat")
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert remote.calls == [] and model.calls == []


def test_technical_failure_and_refresh_never_create_completed_answer(branch_factory, tool_database):
    remote = Client()
    remote.error = LogisticsFailure("logistics_timeout", True)
    service, run, _, remote = harness(branch_factory, tool_database, client=remote)
    events = list(service.run(run.run_id))
    assert any(e.type == "error" and e.data["code"] == "logistics_timeout" for e in events), events
    assert service.get_run(run.run_id).status == "interrupted"
    assert len(service.detail(run.conversation_id).messages) == 1
    list(service.snapshot_events(run.run_id))
    assert len(remote.calls) == 1
    list(service.run(run.run_id))
    assert len(remote.calls) == 1
    remote.error = None
    list(service.resume(run.conversation_id, run.run_id))
    assert service.get_run(run.run_id).status == "completed" and len(remote.calls) == 2


def test_s01b_historical_tracking_cannot_skip_current_order_stage(branch_factory, tool_database):
    service, run, _, remote = harness(branch_factory, tool_database, responses=[decision(logistics_call())])
    events = list(service.run(run.run_id))
    assert any(e.type == "error" and e.data["code"] == "business_scope_violation" for e in events), events
    assert remote.calls == []
    with branch_factory() as session:
        assert session.scalar(select(ConversationAction)) is None


def test_http_details_and_sse_replay_keep_same_simulated_text(branch_factory, tool_database):
    import resolv.api.conversations as routes

    service, run, _, remote = harness(branch_factory, tool_database)
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    app = FastAPI()
    app.state.conversation_runtime = service
    app.include_router(routes.router)
    with TestClient(app) as http:
        detail = http.get(f"/conversations/{run.conversation_id}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["messages"][-1]["text"] == "【模拟物流】当前在上海转运中心。"
        replay = http.post(f"/conversations/{run.conversation_id}/runs/{run.run_id}/resume")
        assert replay.status_code == 200 and "【模拟物流】当前在上海转运中心。" in replay.text
    assert len(remote.calls) == 1


def test_checkpoint_replay_after_remote_commit_does_not_reject_original_read_policy(branch_factory, tool_database):
    # Crash outside the read service, after it returned but before the node can
    # checkpoint; the saved policy must still allow the original logistics call.
    service, run, _, remote = harness(branch_factory, tool_database)
    original = service.logistics_read_service.read
    reached = []

    def interrupted(*args, **kwargs):
        value = original(*args, **kwargs)
        if not reached:
            reached.append(True)
            raise RuntimeError("checkpoint gap")
        return value

    service.logistics_read_service.read = interrupted
    first = list(service.run(run.run_id))
    assert any(e.type == "error" for e in first), first
    with branch_factory() as session:
        assert session.scalar(select(Attempt)).state == "completed"
    resumed = list(service.resume(run.conversation_id, run.run_id))
    assert service.get_run(run.run_id).status == "completed", resumed
    assert len(remote.calls) == 1


def test_v4_logistics_keeps_reply_schema_three(branch_factory, tool_database, tmp_path):
    from resolv.core.refund_config import default_refund_profile
    from resolv.schemas.refund import AdapterDescriptor
    from tests.knowledge_branch_fixtures import branch_profile
    from tests.test_refund_model import profile as refund_model_profile

    service, _, _, remote = harness(branch_factory, tool_database)
    service.orchestration_version = "routed-v4"
    service.knowledge_profile = branch_profile()
    service.release_selector_path = tmp_path / "absent.json"
    service.refund_profile = default_refund_profile(
        understanding_config=refund_model_profile("refund_query"),
        business_config=refund_model_profile(), summary_profile={},
        adapter_descriptor=AdapterDescriptor(
            namespace="unconnected", implementation_version="1", authorization_version="1",
        ),
    )
    c = service.create()
    run, _ = service.accept(c.conversation_id, "v4", "查询订单 MH001 的物流", "agent")
    events = list(service.run(run.run_id))
    saved = service.get_run(run.run_id)
    assert saved.status == "completed", events
    assert saved.result_json["schema_version"] == 3 and saved.state_schema_version == 4
    assert service.detail(c.conversation_id).messages[-1].text == "【模拟物流】当前在上海转运中心。"
    list(service.resume(c.conversation_id, run.run_id))
    assert len(remote.calls) == 1


def test_completed_logistics_action_must_match_verified_audit_on_resume(branch_factory, tool_database):
    service, run, _, remote = harness(branch_factory, tool_database)
    original = service.logistics_read_service.read
    reached = []

    def gap(*args, **kwargs):
        value = original(*args, **kwargs)
        if not reached:
            reached.append(True)
            raise RuntimeError("checkpoint gap")
        return value

    service.logistics_read_service.read = gap
    list(service.run(run.run_id))
    with branch_factory.begin() as session:
        row = session.scalar(select(ConversationAction).where(ConversationAction.tool_name == "query_logistics"))
        row.result = {**row.result, "location": "unverified replacement"}
    events = list(service.resume(run.conversation_id, run.run_id))
    assert any(e.type == "error" and e.data["code"] == "logistics_store_failed" for e in events), events
    assert len(remote.calls) == 1 and len(service.detail(run.conversation_id).messages) == 1
