import json

from langchain_core.messages import AIMessage

from tests.test_api_agent import cancel_call, fake_model
from tests.test_api_conversations import ask, create
from tests.test_conversation_graph import read_decision


def events(response):
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        (
            frame.splitlines()[0].removeprefix("event: "),
            json.loads(frame.splitlines()[1][6:]),
        )
        for frame in response.text.strip().split("\n\n")
    ]


def test_completed_stream_emits_run_identifiers_and_done(client, api_runtime):
    fake_model(api_runtime, [AIMessage(content="你好。")])
    session = create(client)
    result = events(ask(client, session))
    assert [kind for kind, _ in result] == [
        "run_started",
        "knowledge_sources",
        "text_delta",
        "message",
        "done",
    ]
    ids = {data["run_id"] for _, data in result}
    assert len(ids) == 1
    assert result[-1][1]["status"] == "completed"


def test_read_tool_stream_preserves_call_result_order(client, api_runtime):
    fake_model(api_runtime, [read_decision(), AIMessage(content="done")])
    result = events(ask(client, create(client)))
    tools = [(kind, data) for kind, data in result if kind.startswith("tool_")]
    assert [kind for kind, _ in tools] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
    ]
    assert [data["tool_call_id"] for _, data in tools] == ["tc1", "tc1", "tc2", "tc2"]
    assert tools[0][1]["action_id"] == tools[1][1]["action_id"]
    assert tools[1][1]["result"]["tracking_no"] == "SF001"


def test_multistep_order_and_logistics_stream(client, api_runtime):
    fake_model(
        api_runtime,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "one", "name": "query_order", "args": {"order_id": "MH001"}}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "two",
                        "name": "query_logistics",
                        "args": {"tracking_no": "SF001"},
                    }
                ],
            ),
            AIMessage(content="done"),
        ],
    )
    result = events(ask(client, create(client)))
    assert [
        (kind, data["tool_call_id"])
        for kind, data in result
        if kind.startswith("tool_")
    ] == [
        ("tool_call", "one"),
        ("tool_result", "one"),
        ("tool_call", "two"),
        ("tool_result", "two"),
    ]


def test_write_stream_suspends_before_execution(client, api_runtime, tool_database):
    from tests.test_api_agent import order_status

    fake_model(api_runtime, [cancel_call()])
    session = create(client)
    result = events(ask(client, session))
    assert (
        result[-2][0] == "approval_required"
        and result[-1][1]["status"] == "waiting_approval"
    )
    assert order_status(tool_database) == "processing"
    assert ask(client, session, "new").status_code == 409


def test_stream_error_is_safe_and_keeps_original_question(client, api_runtime):
    fake_model(api_runtime, [RuntimeError("private key and URL")])
    session = create(client)
    result = events(ask(client, session))
    assert result[-1][0] == "error" and not any(kind == "done" for kind, _ in result)
    assert "private" not in str(result)
    assert len(api_runtime.detail(session, None, 50).messages) == 1


def test_tool_failure_has_no_fake_tool_result(client, api_runtime, monkeypatch):
    import resolv.conversation_graph as module

    fake_model(api_runtime, [read_decision()])
    monkeypatch.setattr(
        module,
        "execute_tool_call",
        lambda call: (_ for _ in ()).throw(RuntimeError("private DB")),
    )
    result = events(ask(client, create(client)))
    assert result[-1][0] == "error" and not any(
        kind == "tool_result" for kind, _ in result
    )


def test_stream_uses_same_session_completed_history(client, api_runtime):
    model = fake_model(
        api_runtime, [AIMessage(content="first"), AIMessage(content="second")]
    )
    session = create(client)
    ask(client, session, "one", "old", path="/agent/chat")
    ask(client, session, "two", "follow up")
    assert [m.content for m in model.calls[-1][1:]] == ["old", "first", "follow up"]


def test_invalid_or_pending_stream_fails_before_sse(client, api_runtime):
    session = create(client)
    assert ask(client, session, question="").status_code == 422
    fake_model(api_runtime, [cancel_call()])
    ask(client, session)
    response = ask(client, session, "new")
    assert response.status_code == 409 and response.headers["content-type"].startswith(
        "application/json"
    )
