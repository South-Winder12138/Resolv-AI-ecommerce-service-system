import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import resolv.tool_calling as tool_calling_module
from resolv.tools.executor import ToolExecutionResult
from resolv.tools.order import query_order


def test_tool_result_becomes_json_tool_message_with_matching_id() -> None:
    result = ToolExecutionResult(
        tool_call_id="call-123",
        name="query_order",
        output={"order_id": "MH001", "found": True, "status": "shipped"},
    )

    message = tool_calling_module.tool_result_to_message(result)

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-123"
    assert message.content == '{"found":true,"order_id":"MH001","status":"shipped"}'
    assert json.loads(message.content) == result.output


def test_no_tool_call_returns_first_answer_without_second_invoke(monkeypatch) -> None:
    first = AIMessage(content="你好，我是 Resolv。")

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            assert tools == [query_order]
            return self

        def invoke(self, messages):
            self.calls += 1
            assert len(messages) == 2
            return first

    model = FakeModel()
    monkeypatch.setattr(tool_calling_module, "get_chat_model", lambda: model)
    monkeypatch.setattr(
        tool_calling_module,
        "execute_tool_call",
        lambda call: pytest.fail("Executor must not run for a greeting"),
    )

    assert tool_calling_module.answer_with_tool("你好，请介绍一下自己。") == first.text
    assert model.calls == 1


def test_single_tool_roundtrip_sends_complete_message_sequence(monkeypatch) -> None:
    first = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "MH001"}, "id": "call-123"}
        ],
    )
    final = AIMessage(content="订单 MH001 已发货，物流单号为 SF001。")
    execution = ToolExecutionResult(
        tool_call_id="call-123",
        name="query_order",
        output={"order_id": "MH001", "found": True, "status": "shipped"},
    )
    executed_calls = []

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            assert tools == [query_order]
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                assert [type(message) for message in messages] == [
                    SystemMessage,
                    HumanMessage,
                ]
                return first
            assert self.calls == 2
            assert [type(message) for message in messages] == [
                SystemMessage,
                HumanMessage,
                AIMessage,
                ToolMessage,
            ]
            assert messages[2] is first
            assert messages[3].tool_call_id == first.tool_calls[0]["id"]
            assert json.loads(messages[3].content) == execution.output
            return final

    model = FakeModel()
    monkeypatch.setattr(tool_calling_module, "get_chat_model", lambda: model)

    def fake_execute(call):
        executed_calls.append(call)
        return execution

    monkeypatch.setattr(tool_calling_module, "execute_tool_call", fake_execute)

    answer = tool_calling_module.answer_with_tool("帮我查订单 MH001 到哪里了。")

    assert answer == final.text
    assert executed_calls == [first.tool_calls[0]]
    assert model.calls == 2


def test_multiple_first_tool_calls_are_rejected(monkeypatch) -> None:
    first = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "MH001"}, "id": "call-1"},
            {"name": "query_order", "args": {"order_id": "MH002"}, "id": "call-2"},
        ],
    )

    class FakeModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return first

    monkeypatch.setattr(tool_calling_module, "get_chat_model", FakeModel)
    monkeypatch.setattr(
        tool_calling_module,
        "execute_tool_call",
        lambda call: pytest.fail("No tool should execute"),
    )

    with pytest.raises(
        tool_calling_module.UnsupportedToolCallError, match="at most one"
    ):
        tool_calling_module.answer_with_tool("查两个订单")


def test_second_tool_call_is_rejected_without_loop(monkeypatch) -> None:
    tool_call = {
        "name": "query_order",
        "args": {"order_id": "MH001"},
        "id": "call-1",
    }
    first = AIMessage(content="", tool_calls=[tool_call])
    second = AIMessage(content="", tool_calls=[tool_call])

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            return first if self.calls == 1 else second

    model = FakeModel()
    monkeypatch.setattr(tool_calling_module, "get_chat_model", lambda: model)
    monkeypatch.setattr(
        tool_calling_module,
        "execute_tool_call",
        lambda call: ToolExecutionResult("call-1", "query_order", {"found": True}),
    )

    with pytest.raises(
        tool_calling_module.UnsupportedToolCallError, match="second tool call"
    ):
        tool_calling_module.answer_with_tool("查订单")

    assert model.calls == 2
