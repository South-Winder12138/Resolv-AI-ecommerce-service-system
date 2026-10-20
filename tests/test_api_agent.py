import json

from langchain_core.messages import AIMessage
from sqlalchemy import select

from resolv.main import app
from resolv.models import Order
from tests.test_api_conversations import ask, create


def fake_model(api_runtime, responses):
    api_runtime.model.responses = list(responses)
    return api_runtime.model


def cancel_call(order_id="MH002", call_id="call-http"):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "cancel_order", "args": {"order_id": order_id}, "id": call_id}
        ],
    )


def order_status(factory, order_no="MH002"):
    with factory() as session:
        return session.scalar(select(Order).where(Order.order_no == order_no)).status


def test_http_approval_exact_call_and_idempotent_resume(
    client, api_runtime, tool_database
):
    model = fake_model(api_runtime, [cancel_call(), AIMessage(content="订单已取消。")])
    session = create(client)
    first = ask(client, session, question="取消 MH002", path="/agent/chat").json()
    assert first["status"] == "approval_required"
    assert first["action"] == {
        "tool": "cancel_order",
        "arguments": {"order_id": "MH002"},
    }
    assert order_status(tool_database) == "processing"
    assert len(api_runtime.detail(session, None, 50).messages) == 1
    assert ask(client, session, "other", path="/agent/chat").status_code == 409
    other = create(client)
    assert (
        client.post(
            f"/agent/approvals/{first['approval_id']}",
            json={"session_id": other, "approved": True},
        ).status_code
        == 409
    )
    path = f"/agent/approvals/{first['approval_id']}"
    approved = client.post(path, json={"session_id": session, "approved": True})
    assert approved.json()["answer"] == "订单已取消。"
    assert order_status(tool_database) == "cancelled"
    assert len(model.calls) == 2
    assert json.loads(model.calls[-1][-1].content)["changed"] is True
    duplicate = client.post(path, json={"session_id": session, "approved": True})
    assert duplicate.json() == approved.json()
    assert len(model.calls) == 2
    assert (
        client.post(path, json={"session_id": session, "approved": False}).status_code
        == 409
    )


def test_http_rejection_preserves_database_and_commits_final_turn(
    client, api_runtime, tool_database
):
    model = fake_model(api_runtime, [cancel_call(), AIMessage(content="没有取消。")])
    session = create(client)
    first = ask(client, session, path="/agent/chat").json()
    rejected = client.post(
        f"/agent/approvals/{first['approval_id']}",
        json={"session_id": session, "approved": False},
    )
    assert rejected.json()["answer"] == "没有取消。"
    assert order_status(tool_database) == "processing"
    assert json.loads(model.calls[-1][-1].content) == {
        "executed": False,
        "reason": "user_rejected",
    }
    assert len(api_runtime.detail(session, None, 50).messages) == 2


def test_http_no_tool_completes_without_pending(client, api_runtime):
    fake_model(api_runtime, [AIMessage(content="你好。")])
    session = create(client)
    response = ask(client, session, path="/agent/chat")
    assert (
        response.json()["status"] == "completed"
        and response.json()["answer"] == "你好。"
    )
    assert api_runtime.detail(session, None, 50).current_run is None


def test_http_agent_reuses_only_same_session_completed_turns(client, api_runtime):
    model = fake_model(
        api_runtime,
        [
            AIMessage(content="first"),
            AIMessage(content="second"),
            AIMessage(content="other"),
        ],
    )
    session = create(client)
    ask(client, session, "one", "old question", path="/agent/chat")
    ask(client, session, "two", "follow up", path="/agent/chat")
    ask(client, create(client), "other", "follow up", path="/agent/chat")
    assert [m.content for m in model.calls[1][1:]] == [
        "old question",
        "first",
        "follow up",
    ]
    assert [m.content for m in model.calls[2][1:]] == ["follow up"]


def test_http_approval_resumes_original_history(client, api_runtime):
    model = fake_model(
        api_runtime,
        [AIMessage(content="first"), cancel_call(), AIMessage(content="final")],
    )
    session = create(client)
    ask(client, session, "old", "old question", path="/agent/chat")
    approval = ask(client, session, "write", "取消 MH002", path="/agent/chat").json()
    assert len(api_runtime.detail(session, None, 50).messages) == 3
    client.post(
        f"/agent/approvals/{approval['approval_id']}",
        json={"session_id": session, "approved": True},
    )
    assert [m.content for m in model.calls[-1][1:4]] == [
        "old question",
        "first",
        "取消 MH002",
    ]
    assert len(api_runtime.detail(session, None, 50).messages) == 4


def test_http_input_validation_and_no_argument_override(client, api_runtime):
    fake_model(api_runtime, [cancel_call()])
    session = create(client)
    assert client.post("/agent/chat", json={}).status_code == 422
    first = ask(client, session, path="/agent/chat").json()
    assert (
        client.post(
            f"/agent/approvals/{first['approval_id']}",
            json={"session_id": session, "approved": "yes"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/agent/approvals/{first['approval_id']}",
            json={"session_id": session, "approved": True, "arguments": {}},
        ).status_code
        == 422
    )


def test_different_sessions_can_hold_distinct_pending_approvals(client, api_runtime):
    fake_model(api_runtime, [cancel_call(call_id="one"), cancel_call(call_id="two")])
    first = ask(client, create(client), path="/agent/chat").json()
    second = ask(client, create(client), path="/agent/chat").json()
    assert first["approval_id"] != second["approval_id"]


def test_second_write_request_gets_new_approval_and_keeps_original_question(
    client, api_runtime, tool_database
):
    fake_model(
        api_runtime,
        [
            cancel_call(call_id="one"),
            cancel_call(call_id="two"),
            AIMessage(content="final"),
        ],
    )
    session = create(client)
    first = ask(client, session, path="/agent/chat").json()
    second = client.post(
        f"/agent/approvals/{first['approval_id']}",
        json={"session_id": session, "approved": True},
    ).json()
    assert (
        second["status"] == "approval_required"
        and second["approval_id"] != first["approval_id"]
    )
    assert order_status(tool_database) == "cancelled"
    assert len(api_runtime.detail(session, None, 50).messages) == 1
    final = client.post(
        f"/agent/approvals/{second['approval_id']}",
        json={"session_id": session, "approved": False},
    ).json()
    assert (
        final["status"] == "completed"
        and len(api_runtime.detail(session, None, 50).messages) == 2
    )


def test_agent_responses_are_discriminated_by_status():
    schema = app.openapi()["paths"]["/agent/chat"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    assert schema["discriminator"]["propertyName"] == "status"
