import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select

import resolv.agent_loop as agent_loop_module
from resolv.agent_events import (
    AgentDoneEvent,
    AgentMessageEvent,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from resolv.models import Order
from resolv.tools.cancel import cancel_order
from resolv.tools.logistics import query_logistics
from resolv.tools.order import query_order
from resolv.tools.refund import apply_refund, query_refund_status
from resolv.tools.registry import (
    UnknownToolError,
    get_tool,
)
from tests.stream_fakes import as_chunk

pytestmark = pytest.mark.usefixtures("tool_database")


def call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id}


def fake_model(monkeypatch, responses):
    class FakeModel:
        def __init__(self):
            self.seen_messages = []

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
            self.seen_messages.append(list(messages))
            yield as_chunk(responses[len(self.seen_messages) - 1])

    model = FakeModel()
    monkeypatch.setattr(agent_loop_module, "get_chat_model", lambda: model)
    return model


def test_no_tool_call_returns_after_one_model_decision(monkeypatch) -> None:
    model = fake_model(monkeypatch, [AIMessage(content="你好，我是 Resolv。")])

    assert agent_loop_module.run_agent("你好") == agent_loop_module.AgentCompleted(
        "你好，我是 Resolv。"
    )
    assert len(model.seen_messages) == 1
    assert [type(message) for message in model.seen_messages[0]] == [
        SystemMessage,
        HumanMessage,
    ]


def test_agent_places_completed_history_before_current_question(monkeypatch) -> None:
    model = fake_model(monkeypatch, [AIMessage(content="已收到。")])
    history = [
        HumanMessage(content="帮我查 MH001"),
        AIMessage(content="订单 MH001 已发货，物流单号 SF001。"),
    ]

    agent_loop_module.run_agent("那它现在到哪了？", history=history)

    messages = model.seen_messages[0]
    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [message.content for message in messages[1:]] == [
        "帮我查 MH001",
        "订单 MH001 已发货，物流单号 SF001。",
        "那它现在到哪了？",
    ]
    assert history == [
        HumanMessage(content="帮我查 MH001"),
        AIMessage(content="订单 MH001 已发货，物流单号 SF001。"),
    ]


def test_single_tool_call_then_final_answer(monkeypatch) -> None:
    first = AIMessage(
        content="", tool_calls=[call("query_order", {"order_id": "MH001"}, "call-1")]
    )
    model = fake_model(monkeypatch, [first, AIMessage(content="订单已发货。")])

    assert agent_loop_module.run_agent("查 MH001") == agent_loop_module.AgentCompleted(
        "订单已发货。"
    )
    second_messages = model.seen_messages[1]
    assert [type(message) for message in second_messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    assert second_messages[2].tool_calls == first.tool_calls
    assert second_messages[3].tool_call_id == "call-1"
    assert json.loads(second_messages[3].content)["tracking_no"] == "SF001"


def test_runtime_events_are_step_objects_before_http_encoding(monkeypatch) -> None:
    fake_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[call("query_order", {"order_id": "MH001"}, "call-1")],
            ),
            AIMessage(content="订单已发货。"),
        ],
    )

    events = list(agent_loop_module.stream_agent_events("查 MH001"))

    assert [type(event) for event in events] == [
        AgentToolCallEvent,
        AgentToolResultEvent,
        AgentTextDeltaEvent,
        AgentMessageEvent,
        AgentDoneEvent,
    ]
    assert events[0].tool_call_id == events[1].tool_call_id == "call-1"
    assert events[1].result["tracking_no"] == "SF001"
    assert events[2].text == events[3].text == "订单已发货。"


def test_two_consecutive_tool_rounds_keep_message_order(monkeypatch) -> None:
    first = AIMessage(
        content="", tool_calls=[call("query_order", {"order_id": "MH001"}, "call-1")]
    )
    second = AIMessage(
        content="",
        tool_calls=[call("query_logistics", {"tracking_no": "SF001"}, "call-2")],
    )
    model = fake_model(monkeypatch, [first, second, AIMessage(content="已到达上海。")])

    assert agent_loop_module.run_agent(
        "查订单和物流"
    ) == agent_loop_module.AgentCompleted("已到达上海。")
    final_messages = model.seen_messages[2]
    assert [type(message) for message in final_messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
    ]
    assert final_messages[3].tool_call_id == "call-1"
    assert final_messages[5].tool_call_id == "call-2"
    logistics = json.loads(final_messages[5].content)
    assert logistics["status"] == "in_transit"
    assert logistics["location"] == "上海转运中心"


def test_multiple_tool_calls_in_one_response_are_executed_in_order(monkeypatch) -> None:
    first = AIMessage(
        content="",
        tool_calls=[
            call("query_order", {"order_id": "MH001"}, "call-1"),
            call("query_logistics", {"tracking_no": "SF001"}, "call-2"),
        ],
    )
    model = fake_model(monkeypatch, [first, AIMessage(content="订单与物流已查。")])

    assert agent_loop_module.run_agent(
        "查订单和物流"
    ) == agent_loop_module.AgentCompleted("订单与物流已查。")
    messages = model.seen_messages[1]
    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        ToolMessage,
    ]
    assert [message.tool_call_id for message in messages[3:]] == ["call-1", "call-2"]


def test_max_iterations_stops_repeated_tool_requests(monkeypatch) -> None:
    repeated = AIMessage(
        content="", tool_calls=[call("query_order", {"order_id": "MH001"}, "call-1")]
    )
    model = fake_model(monkeypatch, [repeated] * 3)

    with pytest.raises(agent_loop_module.MaxIterationsExceeded, match="3"):
        agent_loop_module.run_agent("反复查订单", max_iterations=3)

    assert len(model.seen_messages) == 3


def test_unknown_tool_error_propagates_without_retry(monkeypatch) -> None:
    first = AIMessage(content="", tool_calls=[call("unknown_tool", {}, "call-1")])
    model = fake_model(monkeypatch, [first])

    with pytest.raises(UnknownToolError, match="unknown_tool"):
        agent_loop_module.run_agent("调用未知工具")

    assert len(model.seen_messages) == 1


def test_agent_write_request_stops_before_database_change(
    monkeypatch, tool_database
) -> None:
    first = AIMessage(
        content="",
        tool_calls=[call("cancel_order", {"order_id": "MH002"}, "call-write")],
    )
    model = fake_model(monkeypatch, [first])

    result = agent_loop_module.run_agent("取消 MH002")

    assert isinstance(result, agent_loop_module.AgentApprovalRequired)
    assert result.request.pending_call.name == "cancel_order"
    assert len(model.seen_messages) == 1
    with tool_database() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH002"))
        assert order is not None
        assert order.status == "processing"


def test_registry_contains_logistics_tool() -> None:
    assert get_tool("query_logistics") is query_logistics
    assert query_logistics.invoke({"tracking_no": "SF001"}) == {
        "tracking_no": "SF001",
        "found": True,
        "status": "in_transit",
        "location": "上海转运中心",
    }
