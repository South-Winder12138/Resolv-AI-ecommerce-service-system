"""Version dispatch and durable END recoveries using actual in-memory checkpoints."""

import pytest

from tests.knowledge_branch_fixtures import knowledge_factory
from tests.test_knowledge_branch_recovery import fail_at
from tests.test_knowledge_routing_graph import harness

__all__ = ["knowledge_factory"]


@pytest.mark.parametrize(
    "point",
    [
        "knowledge_after_query_saved",
        "knowledge_after_retrieval_saved",
        "knowledge_after_check_saved",
        "knowledge_after_answer_input_saved",
        "knowledge_after_answer_saved",
    ],
)
def test_checkpoint_replay_reuses_committed_knowledge_stages(
    knowledge_factory, tmp_path, point
):
    h = harness(knowledge_factory, tmp_path)
    h.fault_hook = fail_at(point)
    first = list(h.conversation.run(h.run.run_id))
    assert first[-1].type == "error" and not any(e.type == "message" for e in first)
    h.fault_hook = None
    final = list(h.conversation.resume(h.run.conversation_id, h.run.run_id))
    assert final[-1].type == "done", [(e.type, e.data) for e in final]
    assert (
        h.counts["knowledge_query"]
        == h.counts["retrieval"]
        == h.counts["knowledge_check"]
        == h.counts["knowledge_answer"]
        == 1
    )


def test_selector_switch_does_not_rebind_an_accepted_run(knowledge_factory, tmp_path):
    h = harness(knowledge_factory, tmp_path)
    h.conversation.release_selector_path.write_text(
        '{"enabled":false,"config_version":1}'
    )
    assert list(h.conversation.run(h.run.run_id))[-1].type == "done"
    assert (
        h.conversation.get_run(h.run.run_id).result_json["knowledge"]["release_id"]
        == "fixture-release"
    )


def test_v2_idempotency_survives_v3_acceptor(knowledge_factory, tmp_path):
    from tests.test_conversation_routing import service_for

    old_service = service_for(knowledge_factory)
    c = old_service.create()
    old, _ = old_service.accept(c.conversation_id, "same", "你好", "agent")
    h = harness(knowledge_factory, tmp_path)
    run, created = h.conversation.accept(c.conversation_id, "same", "你好", "agent")
    assert (
        not created
        and run.run_id == old.run_id
        and run.orchestration_version == "routed-v2"
    )


def test_raw_end_checkpoint_failure_keeps_reply_private_until_durable_end(knowledge_factory, tmp_path, monkeypatch):
    h = harness(knowledge_factory, tmp_path)
    saver = h.conversation.checkpoint_factory.saver
    original, fired = saver.put, False
    def put(config, checkpoint, metadata, versions):
        nonlocal fired
        if not fired and config["configurable"]["thread_id"] == h.run.conversation_id and checkpoint["channel_values"].get("phase") == "end":
            fired = True
            raise RuntimeError("injected END checkpoint failure")
        return original(config, checkpoint, metadata, versions)
    monkeypatch.setattr(saver, "put", put)
    first = list(h.conversation.run(h.run.run_id))
    assert fired and first[-1].type == "error" and not any(e.type == "message" for e in first)
    assert h.conversation.get_run(h.run.run_id).status == "finalizing"
    detail = h.conversation.detail(h.run.conversation_id)
    assert not any(m.role == "assistant" for m in detail.messages)
    assert next(r for r in detail.runs if r.run_id == h.run.run_id).result_json is None
    final = list(h.conversation.resume(h.run.conversation_id, h.run.run_id))
    assert final[-1].type == "done", [(e.type,e.data) for e in final]
    assert h.counts["knowledge_answer"] == 1


def test_v2_approval_recovers_under_v3_acceptor(knowledge_factory, tool_database):
    from tests.test_routing_graph import decision, tool
    from tests.test_routing_graph import harness as business_harness
    service, run, _model = business_harness(knowledge_factory, tool_database, "取消订单 MH001", "order", [
        decision(tool(id="read")), decision(tool("cancel_order", id="write")), decision(text="订单已发货，无法取消，未修改订单。")])
    events = list(service.run(run.run_id))
    approval = next(e.data for e in events if e.type == "approval_required")
    service.orchestration_version = "routed-v3"
    service.decide(approval["approval_id"], run.conversation_id, True, require_pause=True)
    assert list(service.resume(run.conversation_id, run.run_id))[-1].type == "done"
    assert service.get_run(run.run_id).orchestration_version == "routed-v2"
    assert service.get_run(run.run_id).result_json["schema_version"] == 1
