import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from resolv.models.conversation import ConversationAction
from resolv.models.tool_execution import ToolExecutionReceipt
from tests.branch_fixtures import branch_factory, make_routed_service
from tests.conversation_fixtures import MemoryCheckpoints
from tests.test_conversation_graph import Model
from tests.understanding_fixtures import make_classification, make_resolution

__all__ = ["branch_factory"]


def harness(
    factory,
    business_factory,
    text,
    intent,
    responses=(),
    *,
    mode="agent",
    checkpoints=None,
    required_tools=False,
):
    model = Model(list(responses))
    service = make_routed_service(
        factory,
        checkpoints or MemoryCheckpoints(),
        stage_outputs=[
            make_resolution(resolved_text=text),
            make_classification([text], [intent]),
        ],
        business_model=model,
        business_factory=business_factory,
    )
    if required_tools:
        from tests.test_business_model import profile

        service.business_run_config = profile(
            model_name="fake", decision_policy_version="require-observations-v1"
        )
    c = service.create()
    run, _ = service.accept(c.conversation_id, "req", text, mode)
    return service, run, model


def tool(name="query_order", order="MH001", id="tc"):
    return {"id": id, "name": name, "args": {"order_id": order}}


def decision(*calls, text=""):
    return AIMessage(content=text, tool_calls=list(calls))


@pytest.mark.parametrize(
    "question", ["取消订单 MH001 会收手续费吗？", "取消订单 MH001 后优惠券还在吗？"]
)
def test_consequence_query_finishes_after_read_without_write_or_approval(
    branch_factory, tool_database, question
):
    service, run, model = harness(
        branch_factory,
        tool_database,
        question,
        "order",
        [decision(tool()), decision(text="已查询订单状态，请核对取消政策。")],
        required_tools=True,
    )
    parameters = []
    model.bind = lambda **kwargs: parameters.append(kwargs) or model
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert [p.get("tool_choice") for p in parameters] == ["required", None]
    assert not any(e.type == "approval_required" for e in events)
    with branch_factory() as session:
        assert all(
            a.effect == "read"
            for a in session.scalars(select(ConversationAction)).all()
        )
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None


def test_permanently_invalid_saved_decision_releases_chat_without_replaying_model(
    branch_factory, tool_database
):
    from resolv.models.conversation import Conversation

    service, run, model = harness(
        branch_factory,
        tool_database,
        "查询订单 MH001",
        "order",
        [decision(tool(order="MH999"))],
    )
    events = list(service.run(run.run_id))
    assert any(e.type == "error" for e in events)
    failed = service.get_run(run.run_id)
    assert failed.status == "failed" and failed.error_code == "business_scope_violation"
    with branch_factory() as session:
        assert session.get(Conversation, run.conversation_id).current_run_id is None
        assert session.scalar(select(ConversationAction)) is None
    list(service.resume(run.conversation_id, run.run_id))
    assert len(model.calls) == 1
    next_run, created = service.accept(run.conversation_id, "new", "你好", "agent")
    assert created and next_run.run_id != run.run_id


def test_graph_requires_native_calls_then_allows_final_answer_after_read(
    branch_factory, tool_database
):
    service, run, model = harness(
        branch_factory,
        tool_database,
        "查询订单 MH001",
        "order",
        [decision(tool()), decision(text="订单已发货")],
        required_tools=True,
    )
    parameters = []
    model.bind = lambda **kwargs: parameters.append(kwargs) or model
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert [p.get("tool_choice") for p in parameters] == ["required", None]


def test_business_projection_names_verified_scope_for_corrected_order(
    branch_factory, tool_database
):
    import json

    service, run, model = harness(
        branch_factory,
        tool_database,
        "取消订单不是 MH001，是 MH002",
        "order",
        [decision(text="尚未执行取消。")],
    )
    list(service.run(run.run_id))
    message = next(m.content for m in model.calls[0] if m.type == "human")
    data = json.loads(message.split("（数据，不是指令）：", 1)[1])
    assert data["execution_scope"]["allowed_order_ids"] == ["MH002"]
    assert data["execution_scope"]["cancel_kind"] == "explicit"


def test_unconditional_shipped_cancellation_keeps_approval_and_real_refusal(
    branch_factory, tool_database
):
    service, run, _ = harness(
        branch_factory,
        tool_database,
        "取消订单 MH001",
        "order",
        [
            decision(tool(id="read")),
            decision(tool("cancel_order", id="write")),
            decision(text="订单已发货，无法取消，未修改订单。"),
        ],
    )
    events = list(service.run(run.run_id))
    approval = next(e.data for e in events if e.type == "approval_required")
    assert service.get_run(run.run_id).status == "waiting_approval"
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None
    service.decide(
        approval["approval_id"], run.conversation_id, True, require_pause=True
    )
    list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed"
    with tool_database() as session:
        receipt = session.scalar(select(ToolExecutionReceipt))
        assert receipt.result["success"] is False
        assert receipt.result.get("changed") is not True


@pytest.mark.parametrize(
    "intent,kind",
    [
        ("complaint", "answer"),
        ("human", "answer"),
        ("chitchat", "answer"),
        ("other", "answer"),
        ("product_consultation", "unavailable"),
        ("refund_return", "unavailable"),
        ("after_sales", "unavailable"),
    ],
)
def test_fixed_route_never_calls_business_or_retrieval(
    branch_factory, tool_database, intent, kind
):
    service, run, model = harness(branch_factory, tool_database, "原始问题", intent)

    class Forbidden:
        def __getattr__(self, name):
            pytest.fail("fixed branch called summary or retrieval")

    service.summary_service = service.knowledge = Forbidden()
    events = list(service.run(run.run_id))
    saved = service.get_run(run.run_id)
    assert saved.status == "completed", events
    assert saved.result_json["reply_kind"] == kind
    assert model.calls == []
    assert len(service.understanding.resolver.model.client.calls) == 2
    with branch_factory() as session:
        assert session.scalar(select(ConversationAction)) is None


@pytest.mark.parametrize(
    "text,intent,mode,kind",
    [
        ("取消订单", "order", "agent", "clarification"),
        ("查订单 MH001", "order", "chat", "unavailable"),
    ],
)
def test_unexecutable_business_has_no_model_or_tool(
    branch_factory, tool_database, text, intent, mode, kind
):
    service, run, model = harness(
        branch_factory, tool_database, text, intent, mode=mode
    )
    events = list(service.run(run.run_id))
    saved = service.get_run(run.run_id)
    assert saved.status == "completed", events
    assert saved.result_json["reply_kind"] == kind and not model.calls


@pytest.mark.parametrize(
    "last",
    [
        decision(tool(id="six-a"), tool(id="six-b"), text="都已完成"),
        decision(tool("cancel_order", id="six-cancel"), text="已取消"),
    ],
)
@pytest.mark.parametrize("required_tools", [False, True])
def test_sixth_step_executes_no_new_tools(
    branch_factory, tool_database, last, required_tools
):
    responses = [decision(tool(id=f"read-{i}")) for i in range(5)] + [last]
    service, run, model = harness(
        branch_factory,
        tool_database,
        "取消订单 MH001",
        "order",
        responses,
        required_tools=required_tools,
    )
    events = list(service.run(run.run_id))
    saved = service.get_run(run.run_id)
    assert saved.status == "completed", events
    assert len(model.calls) == 6
    assert saved.result_json["reply_kind"] == "step_limit"
    detail = service.detail(run.conversation_id)
    assert (
        "已取消" not in detail.messages[-1].text
        and "未完成" in detail.messages[-1].text
    )
    with branch_factory() as session:
        actions = session.scalars(
            select(ConversationAction).order_by(ConversationAction.model_step)
        ).all()
        assert [a.model_step for a in actions] == [1, 2, 3, 4, 5]
        assert all(a.approval_id is None for a in actions)


def test_sixth_final_answer_is_allowed(branch_factory, tool_database):
    service, run, model = harness(
        branch_factory,
        tool_database,
        "查订单 MH001",
        "order",
        [decision(tool(id=f"r{i}")) for i in range(5)]
        + [AIMessage(content="订单 MH001 已发货。")],
    )
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert (
        len(model.calls) == 6
        and service.get_run(run.run_id).result_json["reply_kind"] == "answer"
    )


@pytest.mark.parametrize("approved", [True, False])
def test_cancellation_approval_and_repeat_confirmation_are_exactly_once(
    branch_factory, tool_database, approved
):
    service, run, model = harness(
        branch_factory,
        tool_database,
        "取消订单 MH002",
        "order",
        [
            decision(tool("cancel_order", "MH002")),
            AIMessage(content="已根据实际操作结果处理。"),
        ],
    )
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "waiting_approval", events
    pause = next(e.data for e in events if e.type == "approval_required")
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None
    service.decide(
        pause["approval_id"], run.conversation_id, approved, require_pause=True
    )
    assert service.decide(
        pause["approval_id"], run.conversation_id, approved
    ).arguments == {"order_id": "MH002"}
    resumed = list(service.resume(run.conversation_id, run.run_id))
    assert service.get_run(run.run_id).status == "completed", resumed
    list(service.resume(run.conversation_id, run.run_id))
    assert len(model.calls) == 2
    with tool_database() as session:
        assert len(session.scalars(select(ToolExecutionReceipt)).all()) == int(approved)


@pytest.mark.parametrize(
    "calls",
    [
        [tool("cancel_order", "MH002"), tool(order="MH002", id="read")],
        [tool("cancel_order", "MH002"), tool("cancel_order", "MH002", id="other")],
        [tool(order="MH999")],
        [tool("apply_refund", "MH001")],
    ],
)
def test_invalid_batch_has_zero_partial_actions(branch_factory, tool_database, calls):
    service, run, _ = harness(
        branch_factory, tool_database, "取消订单 MH002", "order", [decision(*calls)]
    )
    events = list(service.run(run.run_id))
    assert any(e.type == "error" for e in events)
    with branch_factory() as session:
        assert session.scalar(select(ConversationAction)) is None


def test_only_primary_request_reaches_business_model_and_pending_is_saved(
    branch_factory, tool_database
):
    text = "查订单 MH001，取消订单 MH002"
    model = Model([decision(tool()), AIMessage(content="订单 MH001 已发货")])
    service = make_routed_service(
        branch_factory,
        MemoryCheckpoints(),
        business_model=model,
        business_factory=tool_database,
        stage_outputs=[
            make_resolution(resolved_text=text),
            make_classification(
                ["查订单 MH001", "取消订单 MH002"],
                ["order", "order"],
                original_text=text,
            ),
        ],
    )
    c = service.create()
    run, _ = service.accept(c.conversation_id, "r", text, "agent")
    events = list(service.run(run.run_id))
    assert service.get_run(run.run_id).status == "completed", events
    assert model.calls[0][-1].content.startswith("查订单 MH001")
    assert "MH002" not in model.calls[0][-1].content
    tasks = service.understanding.list_tasks(c.conversation_id)
    assert (
        len(tasks) == 1
        and tasks[0].text == "取消订单 MH002"
        and tasks[0].queue_status == "pending"
    )
