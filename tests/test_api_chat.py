from langchain_core.messages import AIMessage

from tests.test_api_agent import fake_model
from tests.test_api_agent_stream import events
from tests.test_api_conversations import ask, create


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_requires_session_request_and_nonempty_question(client):
    assert client.post("/chat/stream", json={}).status_code == 422
    session = create(client)
    assert (
        client.post(
            "/chat/stream", json={"session_id": session, "question": "问题"}
        ).status_code
        == 422
    )
    assert ask(client, session, question="", path="/chat/stream").status_code == 422


def test_plain_stream_commits_full_answer(client, api_runtime):
    fake_model(api_runtime, [AIMessage(content="你好！")])
    session = create(client)
    result = events(ask(client, session, path="/chat/stream"))
    assert result[-2][1]["text"] == "你好！" and result[-1][1]["status"] == "completed"
    assert [m.text for m in api_runtime.detail(session, None, 50).messages] == [
        "问题",
        "你好！",
    ]


def test_plain_error_is_safe_and_has_no_final_answer(client, api_runtime):
    fake_model(api_runtime, [RuntimeError("private upstream")])
    session = create(client)
    result = events(ask(client, session, path="/chat/stream"))
    assert result[-1][0] == "error" and not any(kind == "done" for kind, _ in result)
    assert "private" not in str(result)
    assert len(api_runtime.detail(session, None, 50).messages) == 1


def test_second_turn_contains_question_once(client, api_runtime):
    model = fake_model(
        api_runtime, [AIMessage(content="你好，小明"), AIMessage(content="你叫小明")]
    )
    session = create(client)
    ask(client, session, "one", "我叫小明", path="/chat/stream")
    ask(client, session, "two", "我叫什么", path="/chat/stream")
    assert [m.content for m in model.calls[-1][1:]] == [
        "我叫小明",
        "你好，小明",
        "我叫什么",
    ]


def test_archive_preserves_audit_and_other_sessions(client, api_runtime):
    fake_model(api_runtime, [AIMessage(content="first"), AIMessage(content="second")])
    first, second = create(client), create(client)
    ask(client, first, "one", path="/chat/stream")
    ask(client, second, "two", path="/chat/stream")
    assert client.delete(f"/chat/sessions/{first}").json() == {"status": "archived"}
    assert len(client.get(f"/conversations/{first}").json()["messages"]) == 2
    ids = {
        row["conversation_id"]
        for row in client.get("/conversations").json()["conversations"]
    }
    assert ids == {second}
    assert ask(client, first, "new").status_code == 404
