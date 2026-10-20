from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

import resolv.tool_calling as tool_calling_module
from resolv.tools.order import query_order


def test_query_order_has_name_description_and_order_id_schema() -> None:
    schema = convert_to_openai_tool(query_order)

    assert schema["type"] == "function"
    definition = schema["function"]
    assert definition["name"] == "query_order"
    assert "订单号" in definition["description"]
    assert definition["parameters"]["properties"]["order_id"]["type"] == "string"
    assert (
        "用户提供" in definition["parameters"]["properties"]["order_id"]["description"]
    )
    assert "order_id" in definition["parameters"]["required"]


def test_request_tool_call_binds_tool_and_invokes_messages(monkeypatch) -> None:
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "MH001"}, "id": "call-test"}
        ],
    )

    class FakeBoundModel:
        def invoke(self, messages):
            assert len(messages) == 2
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "帮我查订单 MH001 到哪里了。"
            return response

    class FakeModel:
        def bind_tools(self, tools):
            assert tools == [query_order]
            return FakeBoundModel()

    monkeypatch.setattr(tool_calling_module, "get_chat_model", lambda: FakeModel())

    result = tool_calling_module.request_tool_call("帮我查订单 MH001 到哪里了。")

    assert result is response
    assert result.tool_calls[0]["args"] == {"order_id": "MH001"}
