import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from resolv.models.conversation import ConversationAction
from resolv.models.tool_execution import ToolExecutionReceipt
from tests.branch_fixtures import branch_factory
from tests.test_routing_graph import decision, harness, tool
from tests.understanding_fixtures import fail_once

__all__ = ["branch_factory"]


@pytest.mark.parametrize(
    "window", ["after_understanding_terminal", "after_final_result"]
)
def test_fixed_reply_recovery_preserves_actions_and_inference(
    branch_factory, tool_database, window
):
    service, run, model = harness(
        branch_factory, tool_database, "我要投诉配送员", "complaint"
    )
    service.fault_hook = fail_once(window)
    first = list(service.run(run.run_id))
    assert not any(
        e.type == "done" and e.data.get("status") == "completed" for e in first
    )
    snapshot = service.understanding.record(
        run.conversation_id, run.understanding_id
    ).snapshot
    resumed = list(service.resume(run.conversation_id, run.run_id))
    saved = service.get_run(run.run_id)
    assert saved.status == "completed", resumed
    assert len(saved.result_json["suggested_actions"]) == 2 and model.calls == []
    assert len(service.understanding.resolver.model.client.calls) == 2
    assert (
        snapshot
        == service.understanding.record(
            run.conversation_id, run.understanding_id
        ).snapshot
    )


def test_sixth_decision_resume_does_not_reset_or_reinvoke(
    branch_factory, tool_database
):
    service, run, model = harness(
        branch_factory,
        tool_database,
        "查订单 MH001",
        "order",
        [decision(tool(id=f"t{i}")) for i in range(6)],
    )
    service.fault_hook = fail_once("after_sixth_model_step")
    first = list(service.run(run.run_id))
    assert any(e.type == "error" for e in first) and len(model.calls) == 6
    events = list(service.resume(run.conversation_id, run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert (
        len(model.calls) == 6
        and service.get_run(run.run_id).result_json["reply_kind"] == "step_limit"
    )


def test_sqlite_commit_then_parent_failure_reuses_one_receipt(
    branch_factory, tool_database
):
    service, run, model = harness(
        branch_factory,
        tool_database,
        "取消订单 MH002",
        "order",
        [decision(tool("cancel_order", "MH002")), AIMessage(content="取消结果已确认")],
    )
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "waiting_approval", events
    pause = next(e.data for e in events if e.type == "approval_required")
    service.decide(pause["approval_id"], run.conversation_id, True)
    service.fault_hook = fail_once("after_sqlite_commit")
    events = list(service.resume(run.conversation_id, run.run_id))
    assert any(e.type == "error" for e in events)
    with tool_database() as session:
        receipts = session.scalars(select(ToolExecutionReceipt)).all()
        assert len(receipts) == 1 and receipts[0].result["changed"] is True
    events = list(service.resume(run.conversation_id, run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    with branch_factory() as session:
        action = session.scalar(select(ConversationAction))
        assert action.result["changed"] is True
    with tool_database() as session:
        assert len(session.scalars(select(ToolExecutionReceipt)).all()) == 1
    assert len(model.calls) == 2


@pytest.mark.parametrize("committed", [False, True])
def test_permanent_decision_error_does_not_release_unresolved_write(
    branch_factory, tool_database, committed
):
    from resolv.schemas.conversation import ConversationError

    service, run, _ = harness(
        branch_factory,
        tool_database,
        "取消订单 MH002",
        "order",
        [decision(tool("cancel_order", "MH002")), AIMessage(content="取消结果已确认")],
    )
    events = list(service.run(run.run_id))
    pause = next(e.data for e in events if e.type == "approval_required")
    if committed:
        service.decide(pause["approval_id"], run.conversation_id, True)
        service.fault_hook = fail_once("after_sqlite_commit")
        list(service.resume(run.conversation_id, run.run_id))
    with service.execution(run.run_id) as execution:
        service._record_error(
            run.run_id, execution, ConversationError("business_scope_violation")
        )
    detail = service.detail(run.conversation_id, None, 50)
    assert detail.current_run is not None and detail.current_run.run_id == run.run_id
    assert service.get_run(run.run_id).status == "interrupted"
    with tool_database() as session:
        assert len(session.scalars(select(ToolExecutionReceipt)).all()) == int(
            committed
        )
    if committed:
        list(service.resume(run.conversation_id, run.run_id))
        assert service.get_run(run.run_id).status == "completed"
        with tool_database() as session:
            assert len(session.scalars(select(ToolExecutionReceipt)).all()) == 1


def test_tampered_action_cannot_execute_after_pause(branch_factory, tool_database):
    service, run, _ = harness(
        branch_factory,
        tool_database,
        "取消订单 MH002",
        "order",
        [decision(tool("cancel_order", "MH002"))],
    )
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "waiting_approval", events
    pause = next(e.data for e in events if e.type == "approval_required")
    service.decide(pause["approval_id"], run.conversation_id, True)
    with branch_factory.begin() as session:
        session.get(ConversationAction, pause["action_id"]).arguments = {
            "order_id": "MH001"
        }
    events = list(service.resume(run.conversation_id, run.run_id))
    assert any(e.type == "error" for e in events)
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None


@pytest.mark.parametrize(
    "field,value", [("branch", "refund_flow"), ("understanding_id", "another")]
)
def test_tampered_reply_projection_blocks_write_on_recovery(
    branch_factory, tool_database, field, value
):
    service, run, _ = harness(
        branch_factory,
        tool_database,
        "取消订单 MH002",
        "order",
        [decision(tool("cancel_order", "MH002")), AIMessage(content="结束")],
    )
    events = list(service.run(run.run_id))
    pause = next(e.data for e in events if e.type == "approval_required")
    service.decide(pause["approval_id"], run.conversation_id, True)
    with service.execution(run.run_id) as execution:
        graph = service._graph(execution)
        config = {"configurable": {"thread_id": run.conversation_id}}
        state = graph.get_state(config).values
        metadata = {**state["result_metadata"], field: value}
        graph.update_state(
            config, {"result_metadata": metadata}, as_node="stage_action"
        )
    events = list(service.resume(run.conversation_id, run.run_id))
    assert any(e.type == "error" for e in events)
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None


def test_populated_legacy_checkpoint_does_not_block_new_routed_run(
    branch_factory, tool_database
):
    from resolv.services.conversation import ConversationService
    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.test_conversation_graph import Model
    from tests.test_routing_graph import harness

    checkpoints = MemoryCheckpoints()
    routed, _, _ = harness(
        branch_factory, tool_database, "你好", "chitchat", checkpoints=checkpoints
    )
    legacy = ConversationService(
        branch_factory,
        checkpoints,
        model=Model(),
        context_builder=routed.context_builder,
    )
    c = legacy.create()
    old, _ = legacy.accept(c.conversation_id, "old", "旧问题", "chat")
    assert list(legacy.run(old.run_id))[-1].data["status"] == "completed"
    new, _ = routed.accept(c.conversation_id, "new", "你好", "agent")
    events = list(routed.run(new.run_id))
    assert routed.get_run(new.run_id).status == "completed", events


def test_raw_end_checkpoint_failure_never_emits_completed(
    branch_factory, tool_database, monkeypatch
):
    service, run, model = harness(branch_factory, tool_database, "投诉", "complaint")
    saver = service.checkpoint_factory.saver
    original = saver.put
    fired = False

    def put(config, checkpoint, metadata, versions):
        nonlocal fired
        if (
            not fired
            and config["configurable"]["thread_id"] == run.conversation_id
            and checkpoint["channel_values"].get("phase") == "end"
        ):
            fired = True
            raise RuntimeError("parent END failed")
        return original(config, checkpoint, metadata, versions)

    monkeypatch.setattr(saver, "put", put)
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "finalizing"
    assert events[-1].type == "error" and not any(e.type == "done" for e in events)
    saved = service.get_run(run.run_id).result_json
    events = list(service.resume(run.conversation_id, run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert service.get_run(run.run_id).result_json == saved and model.calls == []


def test_restored_read_action_is_rechecked_against_current_whitelist(
    branch_factory, tool_database
):
    service, run, _ = harness(
        branch_factory,
        tool_database,
        "查订单 MH001",
        "order",
        [decision(tool()), AIMessage(content="结束")],
    )
    with service.execution(run.run_id) as execution:
        graph = service._graph(execution)
        config = {"configurable": {"thread_id": run.conversation_id}}
        graph.invoke(
            service.initial_state(run, execution),
            config,
            durability="sync",
            interrupt_after=["agent_tools"],
        )
        state = graph.get_state(config).values
        state["messages"][1].tool_calls[0]["name"] = "query_refund_status"
        graph.update_state(
            config, {"messages": state["messages"]}, as_node="agent_tools"
        )
    with branch_factory.begin() as session:
        session.scalar(select(ConversationAction)).tool_name = "query_refund_status"
    events = list(service.resume(run.conversation_id, run.run_id))
    assert any(
        e.type == "error" and e.data["code"] == "business_scope_violation"
        for e in events
    ), events
