from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select

from resolv.models.conversation import ConversationAction
from tests.branch_fixtures import branch_factory, make_routed_service
from tests.conversation_fixtures import MemoryCheckpoints
from tests.test_api_agent_stream import events
from tests.test_conversation_graph import Model
from tests.test_routing_graph import decision, tool
from tests.understanding_fixtures import make_classification, make_resolution

__all__ = ["branch_factory"]


@pytest.fixture
def routed_client(branch_factory, tool_database, monkeypatch):
    import resolv.main as module

    model = Model()
    service = make_routed_service(
        branch_factory,
        MemoryCheckpoints(),
        stage_outputs=[],
        business_model=model,
        business_factory=tool_database,
    )

    @contextmanager
    def owner(**kwargs):
        yield service

    @contextmanager
    def understanding(**kwargs):
        yield service.understanding

    monkeypatch.setattr(module, "conversation_runtime", owner, raising=False)
    monkeypatch.setattr(module, "understanding_runtime", understanding, raising=False)
    monkeypatch.setattr(module, "routed_conversation_runtime", owner, raising=False)

    def arrange(text, intent):
        service.understanding.resolver.model.client.replies.extend(
            [
                AIMessage(
                    content=make_resolution(resolved_text=text).model_dump_json()
                ),
                AIMessage(
                    content=make_classification([text], [intent]).model_dump_json()
                ),
            ]
        )

    with TestClient(module.app) as client:
        yield SimpleNamespace(
            client=client, service=service, model=model, arrange=arrange
        )


def create(h):
    response = h.client.post("/conversations")
    assert response.status_code == 200
    return response.json()["conversation_id"]


def ask(
    h,
    conversation,
    text="我要投诉配送员",
    intent="complaint",
    *,
    path="/agent/chat",
    request_id="r",
    **links,
):
    h.arrange(text, intent)
    return h.client.post(
        path,
        json={
            "session_id": conversation,
            "request_id": request_id,
            "question": text,
            **links,
        },
    )


def test_json_sse_replay_and_details_share_saved_metadata(routed_client):
    h = routed_client
    c = create(h)
    response = ask(h, c)
    assert response.status_code == 200
    reply = response.json()
    assert [a["type"] for a in reply["suggested_actions"]] == [
        "transfer_human",
        "create_ticket",
    ]
    detail = h.client.get(f"/conversations/{c}").json()
    metadata = detail["runs"][-1]["result_json"]
    assert {key: reply[key] for key in metadata} == metadata
    replay = h.client.post(
        "/agent/stream",
        json={"session_id": c, "request_id": "r", "question": "我要投诉配送员"},
    )
    message = next(data for kind, data in events(replay) if kind == "message")
    assert {key: message[key] for key in metadata} == metadata
    assert len(h.service.understanding.resolver.model.client.calls) == 2


@pytest.mark.parametrize(
    "intent,text",
    [
        ("order", "查订单 MH001"),
        ("order", "取消订单 MH002"),
        ("logistics", "查订单 MH001 物流"),
    ],
)
def test_chat_stream_has_zero_business_authority(routed_client, intent, text):
    h = routed_client
    c = create(h)
    response = ask(h, c, text, intent, path="/chat/stream")
    parsed = events(response)
    message = next(data for kind, data in parsed if kind == "message")
    assert message["text"] == "该入口不支持业务执行，请使用客服聊天入口"
    assert message["reply_kind"] == "unavailable" and message["suggested_actions"] == []
    assert h.model.calls == []
    with h.service.session_factory() as session:
        assert session.scalar(select(ConversationAction)) is None


@pytest.mark.parametrize(
    "patch",
    [
        {"branch": "business"},
        {"approved": True},
        {"continue_task_id": ""},
        {"continue_task_id": "t", "clarifies_understanding_id": "u"},
    ],
)
def test_online_input_cannot_inject_routes_or_approval(routed_client, patch):
    h = routed_client
    c = create(h)
    response = h.client.post(
        "/agent/chat",
        json={"session_id": c, "request_id": "r", "question": "你好", **patch},
    )
    assert response.status_code == 422


def test_links_are_in_request_fingerprint_and_cross_conversation_rejected(
    routed_client,
):
    h = routed_client
    c = create(h)
    reply = ask(h, c).json()
    response = h.client.post(
        "/agent/chat",
        json={
            "session_id": c,
            "request_id": "r",
            "question": "我要投诉配送员",
            "continue_task_id": "t",
        },
    )
    assert (
        response.status_code == 409
        and response.json()["detail"]["code"] == "request_conflict"
    )
    other = create(h)
    response = h.client.post(
        "/agent/chat",
        json={
            "session_id": other,
            "request_id": "new",
            "question": "是这单",
            "clarifies_understanding_id": reply["understanding_id"],
        },
    )
    assert response.status_code in {404, 409, 503}
    assert h.service.detail(other).messages == []


def test_approval_has_no_final_metadata_then_final_is_persisted(routed_client):
    h = routed_client
    c = create(h)
    h.model.responses = [
        decision(tool("cancel_order", "MH002")),
        AIMessage(content="已根据真实结果处理"),
    ]
    reply = ask(h, c, "取消订单 MH002", "order").json()
    assert reply["status"] == "approval_required" and "suggested_actions" not in reply
    detail = h.client.get(f"/conversations/{c}").json()
    assert detail["current_run"]["result_json"] is None
    final = h.client.post(
        f"/agent/approvals/{reply['approval_id']}",
        json={"session_id": c, "approved": False},
    ).json()
    assert final["reply_kind"] == "answer" and final["suggested_actions"] == []
    assert (
        h.client.get(f"/conversations/{c}").json()["runs"][-1]["result_json"][
            "understanding_id"
        ]
        == final["understanding_id"]
    )


def test_unexpected_accept_error_is_safe(routed_client, monkeypatch):
    h = routed_client
    c = create(h)

    def fail(*args, **kwargs):
        raise RuntimeError("password private-provider-error")

    monkeypatch.setattr(h.service.understanding.context, "freeze", fail)
    response = h.client.post(
        "/agent/chat", json={"session_id": c, "request_id": "r", "question": "你好"}
    )
    assert response.status_code == 503 and "password" not in response.text
