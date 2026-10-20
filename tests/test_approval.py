import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select

from resolv import agent_loop
from resolv.approval import InvalidApprovalState
from resolv.models import Order
from resolv.tools.cancel import cancel_order
from resolv.tools.executor import execute_approved_tool_call
from resolv.tools.logistics import query_logistics
from resolv.tools.order import query_order
from resolv.tools.refund import apply_refund, query_refund_status
from resolv.tools.registry import ToolEffect
from tests.stream_fakes import as_chunk

pytestmark = pytest.mark.usefixtures("tool_database")


def call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id}


def fake_model(monkeypatch, responses):
    class FakeModel:
        def __init__(self):
            self.seen: list[list[object]] = []

        def bind_tools(self, tools):
            assert tools == [
                query_order,
                query_logistics,
                query_refund_status,
                cancel_order,
                apply_refund,
            ]
            return self

        def stream(self, messages):
            self.seen.append(list(messages))
            yield as_chunk(responses[len(self.seen) - 1])

    model = FakeModel()
    monkeypatch.setattr(agent_loop, "get_chat_model", lambda: model)
    return model


def status(factory, order_no: str) -> str:
    with factory() as session:
        order = session.scalar(select(Order).where(Order.order_no == order_no))
        assert order is not None
        return order.status


def test_approve_executes_exact_pending_call_then_resumes(monkeypatch, tool_database):
    first = AIMessage(
        content="",
        tool_calls=[call("cancel_order", {"order_id": "MH002"}, "call-original")],
    )
    model = fake_model(monkeypatch, [first, AIMessage(content="订单已取消。")])

    paused = agent_loop.run_agent("取消 MH002")
    assert isinstance(paused, agent_loop.AgentApprovalRequired)
    assert paused.request.pending_call.tool_call_id == "call-original"
    assert paused.request.pending_call.name == "cancel_order"
    assert paused.request.pending_call.args == {"order_id": "MH002"}
    assert paused.request.pending_call.effect is ToolEffect.WRITE
    assert paused.state.remaining_iterations == 7
    assert len(model.seen) == 1
    assert status(tool_database, "MH002") == "processing"

    result = agent_loop.resume_agent(paused.state, approved=True)
    assert result == agent_loop.AgentCompleted("订单已取消。")
    assert status(tool_database, "MH002") == "cancelled"
    assert len(model.seen) == 2
    assert [type(message) for message in model.seen[1]] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    tool_message = model.seen[1][-1]
    assert tool_message.tool_call_id == "call-original"
    assert json.loads(tool_message.content)["changed"] is True


def test_reject_sends_matching_observation_without_mutation(monkeypatch, tool_database):
    first = AIMessage(
        content="",
        tool_calls=[call("cancel_order", {"order_id": "MH002"}, "call-rejected")],
    )
    model = fake_model(monkeypatch, [first, AIMessage(content="好的，没有取消。")])
    paused = agent_loop.run_agent("取消 MH002")

    result = agent_loop.resume_agent(paused.state, approved=False)

    assert result == agent_loop.AgentCompleted("好的，没有取消。")
    assert status(tool_database, "MH002") == "processing"
    observation = model.seen[1][-1]
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == "call-rejected"
    assert json.loads(observation.content) == {
        "executed": False,
        "reason": "user_rejected",
    }


def test_display_args_cannot_change_approved_action(monkeypatch, tool_database):
    first = AIMessage(
        content="",
        tool_calls=[call("cancel_order", {"order_id": "MH002"}, "call-1")],
    )
    fake_model(monkeypatch, [first, AIMessage(content="完成")])
    paused = agent_loop.run_agent("取消 MH002")

    paused.request.pending_call.args["order_id"] = "MH001"
    paused.state.messages[-1].tool_calls[0]["args"]["order_id"] = "MH001"
    object.__setattr__(
        paused.request.pending_call, "_args_json", '{"order_id":"MH001"}'
    )
    assert paused.request.pending_call.args == {"order_id": "MH001"}

    agent_loop.resume_agent(paused.state, approved=True)
    assert status(tool_database, "MH002") == "cancelled"
    assert status(tool_database, "MH001") == "shipped"


def test_non_boolean_approval_cannot_consume_state(monkeypatch, tool_database):
    first = AIMessage(
        content="",
        tool_calls=[call("cancel_order", {"order_id": "MH002"}, "call-1")],
    )
    fake_model(monkeypatch, [first, AIMessage(content="完成")])
    paused = agent_loop.run_agent("取消 MH002")
    with pytest.raises(TypeError, match="approved must be a bool"):
        agent_loop.resume_agent(paused.state, approved="yes")
    assert status(tool_database, "MH002") == "processing"
    assert agent_loop.resume_agent(
        paused.state, approved=False
    ) == agent_loop.AgentCompleted("完成")


def test_forged_or_reused_state_cannot_approve(monkeypatch, tool_database):
    first = AIMessage(
        content="",
        tool_calls=[call("cancel_order", {"order_id": "MH002"}, "call-1")],
    )
    fake_model(monkeypatch, [first, AIMessage(content="完成")])
    paused = agent_loop.run_agent("取消 MH002")
    forged = replace(paused.state)

    with pytest.raises(InvalidApprovalState):
        agent_loop.resume_agent(forged, approved=True)
    assert status(tool_database, "MH002") == "processing"

    agent_loop.resume_agent(paused.state, approved=True)
    with pytest.raises(InvalidApprovalState):
        agent_loop.resume_agent(paused.state, approved=True)


def test_arbitrary_write_cannot_use_approved_executor():
    pending = agent_loop.PendingToolCall.from_tool_call(
        call("cancel_order", {"order_id": "MH002"}, "forged"), ToolEffect.WRITE
    )
    with pytest.raises(InvalidApprovalState):
        execute_approved_tool_call(pending)


def test_approved_shipped_order_still_obeys_business_rule(monkeypatch, tool_database):
    first = AIMessage(
        content="",
        tool_calls=[call("cancel_order", {"order_id": "MH001"}, "call-shipped")],
    )
    model = fake_model(monkeypatch, [first, AIMessage(content="该订单无法取消。")])
    paused = agent_loop.run_agent("取消 MH001")

    result = agent_loop.resume_agent(paused.state, approved=True)

    assert result == agent_loop.AgentCompleted("该订单无法取消。")
    assert status(tool_database, "MH001") == "shipped"
    assert json.loads(model.seen[1][-1].content)["reason"] == "order_not_cancellable"


@pytest.mark.parametrize(
    "calls",
    [
        [
            call("cancel_order", {"order_id": "MH002"}, "write-1"),
            call("cancel_order", {"order_id": "MH001"}, "write-2"),
        ],
        [
            call("query_order", {"order_id": "MH002"}, "read-1"),
            call("cancel_order", {"order_id": "MH002"}, "write-1"),
        ],
    ],
)
def test_write_batches_fail_closed(monkeypatch, tool_database, calls):
    from resolv.services.order import OrderService

    reads: list[str] = []
    original = OrderService.get_order

    def observe_read(self, order_no):
        reads.append(order_no)
        return original(self, order_no)

    monkeypatch.setattr(OrderService, "get_order", observe_read)
    first = AIMessage(content="", tool_calls=calls)
    model = fake_model(monkeypatch, [first])
    with pytest.raises(agent_loop.UnsupportedApprovalBatchError):
        agent_loop.run_agent("批量操作")
    assert len(model.seen) == 1
    assert reads == []
    assert status(tool_database, "MH002") == "processing"
