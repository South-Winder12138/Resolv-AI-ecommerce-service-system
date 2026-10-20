from langchain_core.messages import AIMessageChunk, ToolCallChunk

from tests.test_api_agent import order_status
from tests.test_api_agent_stream import events
from tests.test_api_conversations import ask, create


def scripted_model(api_runtime, rounds):
    class Model:
        def __init__(self):
            self.seen = []

        def bind_tools(self, tools):
            return self

        def bind(self, **kwargs):
            return self

        def stream(self, messages):
            self.seen.append(list(messages))
            for item in rounds[len(self.seen) - 1]:
                if isinstance(item, Exception):
                    raise item
                yield item

    model = Model()
    api_runtime.model = model
    return model


def split_tool_call(name="query_order", call_id="call-1"):
    return [
        AIMessageChunk(
            content="我来查一下。",
            tool_call_chunks=[
                ToolCallChunk(name=name, args='{"order', id=call_id, index=0)
            ],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                ToolCallChunk(name=None, args='_id":"MH001"}', id=None, index=0)
            ],
        ),
    ]


def test_final_chunks_become_deltas_and_one_canonical_turn(client, api_runtime):
    scripted_model(
        api_runtime,
        [[AIMessageChunk(content="订单"), AIMessageChunk(content="已发货")]],
    )
    session = create(client)
    result = events(ask(client, session, question="查订单"))
    assert [data["text"] for kind, data in result if kind == "text_delta"] == [
        "订单",
        "已发货",
    ]
    assert [m.text for m in api_runtime.detail(session, None, 50).messages] == [
        "查订单",
        "订单已发货",
    ]


def test_json_agent_still_returns_full_answer_from_same_runner(client, api_runtime):
    scripted_model(
        api_runtime,
        [[AIMessageChunk(content="订单"), AIMessageChunk(content="已发货")]],
    )
    assert (
        ask(client, create(client), path="/agent/chat").json()["answer"] == "订单已发货"
    )


def test_split_tool_args_execute_only_after_complete_aggregation(client, api_runtime):
    model = scripted_model(
        api_runtime,
        [
            split_tool_call(),
            [AIMessageChunk(content="订单"), AIMessageChunk(content="已发货")],
        ],
    )
    result = events(ask(client, create(client)))
    tools = [(kind, data) for kind, data in result if kind.startswith("tool_")]
    assert tools[0][1]["arguments"] == {"order_id": "MH001"}
    assert tools[1][1]["result"]["tracking_no"] == "SF001"
    assert [data["text"] for kind, data in result if kind == "text_delta"] == [
        "订单",
        "已发货",
    ]
    assert model.seen[-1][-1].tool_call_id == "call-1"


def test_partial_tool_args_are_not_executed(client, api_runtime, monkeypatch):
    import resolv.conversation_graph as graph

    monkeypatch.setattr(
        graph,
        "execute_tool_call",
        lambda call: (_ for _ in ()).throw(AssertionError("partial args executed")),
    )
    scripted_model(api_runtime, [[split_tool_call()[0], RuntimeError("stream failed")]])
    session = create(client)
    result = events(ask(client, session))
    assert result[-1][0] == "error" and not any(
        kind.startswith("tool_") for kind, _ in result
    )
    assert [m.role for m in api_runtime.detail(session, None, 50).messages] == ["user"]


def test_incomplete_tool_call_is_rejected_without_execution(client, api_runtime):
    scripted_model(api_runtime, [[split_tool_call()[0]]])
    result = events(ask(client, create(client)))
    assert result[-1][0] == "error" and not any(
        kind.startswith("tool_") for kind, _ in result
    )


def test_split_write_call_suspends_without_database_change(
    client, api_runtime, tool_database
):
    chunks = split_tool_call("cancel_order", "write-1")
    chunks[1].tool_call_chunks[0]["args"] = '_id":"MH002"}'
    scripted_model(api_runtime, [chunks])
    result = events(ask(client, create(client)))
    approval = next(data for kind, data in result if kind == "approval_required")
    assert approval["arguments"] == {"order_id": "MH002"}
    assert order_status(tool_database) == "processing"


def test_partial_final_text_then_provider_error_does_not_commit(client, api_runtime):
    scripted_model(
        api_runtime,
        [[AIMessageChunk(content="订单"), RuntimeError("secret gateway URL")]],
    )
    session = create(client)
    result = events(ask(client, session))
    assert result[-1][0] == "error" and not any(
        kind in {"text_delta", "message", "done"} for kind, _ in result
    )
    assert "secret" not in str(result)
    assert len(api_runtime.detail(session, None, 50).messages) == 1


def test_reasoning_content_is_not_sent_as_text_delta(client, api_runtime):
    scripted_model(
        api_runtime,
        [
            [
                AIMessageChunk(
                    content=[
                        {"type": "reasoning", "reasoning": "secret thinking"},
                        {"type": "text", "text": "你好"},
                    ]
                )
            ]
        ],
    )
    result = events(ask(client, create(client)))
    assert [data["text"] for kind, data in result if kind == "text_delta"] == ["你好"]
    assert "secret thinking" not in str(result)
