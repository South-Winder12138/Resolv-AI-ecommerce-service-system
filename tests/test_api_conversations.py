from langchain_core.messages import AIMessage

from tests.test_conversation_graph import write_decision


def create(client):
    response = client.post("/conversations")
    assert response.status_code == 200
    return response.json()["conversation_id"]


def ask(client, session, request_id="req", question="问题", path="/agent/stream"):
    return client.post(
        path,
        json={"session_id": session, "request_id": request_id, "question": question},
    )


def test_plain_cannot_bypass_pending_approval(client, api_runtime):
    api_runtime.model.responses = [write_decision()]
    session = create(client)
    assert ask(client, session).status_code == 200
    response = ask(client, session, "new", path="/chat/stream")
    assert (
        response.status_code == 409
        and response.json()["detail"]["code"] == "pending_approval_exists"
    )


def test_duplicate_request_before_first_sse_does_not_add_user(client, api_runtime):
    session = create(client)
    run, _ = api_runtime.accept(session, "req", "问题", "agent")
    response = ask(client, session)
    assert response.status_code == 200 and run.run_id in response.text
    assert not api_runtime.model.calls
    assert len(client.get(f"/conversations/{session}").json()["messages"]) == 1


def test_resume_adds_no_question(client, api_runtime):
    session = create(client)
    run, _ = api_runtime.accept(session, "req", "问题", "agent")
    response = client.post(f"/conversations/{session}/runs/{run.run_id}/resume")
    assert response.status_code == 200 and "event: done" in response.text
    assert len(client.get(f"/conversations/{session}").json()["messages"]) == 2


def test_approval_payload_cannot_change_args(client, api_runtime):
    api_runtime.model.responses = [write_decision()]
    session = create(client)
    ask(client, session)
    action = client.get(f"/conversations/{session}").json()["actions"][0]
    response = client.post(
        f"/agent/approvals/{action['approval_id']}",
        json={
            "session_id": session,
            "approved": True,
            "arguments": {"order_id": "MH002"},
        },
    )
    assert response.status_code == 422


def test_no_auto_resume_at_startup(client, api_runtime):
    session = create(client)
    api_runtime.accept(session, "req", "问题", "agent")
    assert client.get(f"/conversations/{session}").status_code == 200
    assert not api_runtime.model.calls


def test_done_only_after_durable_completion(client, api_runtime):
    session = create(client)
    response = ask(client, session)
    assert "event: done" in response.text
    detail = client.get(f"/conversations/{session}").json()
    assert detail["current_run"] is None and len(detail["messages"]) == 2


def test_pytest_lifespan_needs_no_mysql_settings(client, monkeypatch):
    monkeypatch.delenv("CONVERSATION_DATABASE_URL", raising=False)
    assert client.get("/health").status_code == 200


def test_archive_blocks_active_run(client, api_runtime):
    session = create(client)
    api_runtime.accept(session, "req", "问题", "agent")
    assert client.delete(f"/chat/sessions/{session}").status_code == 409


def test_history_page_cursor_no_duplicate(client, api_runtime):
    session = create(client)
    api_runtime.model.responses = [
        AIMessage(content="first"),
        AIMessage(content="second"),
    ]
    ask(client, session, "first")
    ask(client, session, "second")
    first = client.get(f"/conversations/{session}?limit=2").json()
    second = client.get(
        f"/conversations/{session}?limit=2&before_seq={first['next_before_seq']}"
    ).json()
    assert [m["seq"] for m in second["messages"] + first["messages"]] == [1, 2, 3, 4]


def test_unknown_session_is_not_implicitly_created(client):
    assert ask(client, "unknown").status_code == 404


def test_completed_history_keeps_run_knowledge_snapshot(client, api_runtime):
    from tests.test_conversation_graph import Knowledge

    api_runtime.knowledge = Knowledge()
    session = create(client)
    ask(client, session)
    detail = client.get(f"/conversations/{session}").json()
    assert detail["runs"][0]["knowledge_snapshot"][0]["source"] == "fake.md"


def test_approval_requires_confirmed_pause_checkpoint(client, api_runtime):
    from resolv.repositories.conversation import ConversationRepository

    session = create(client)
    run, _ = api_runtime.accept(session, "req", "退款", "agent")
    with (
        api_runtime.execution(run.run_id) as execution,
        api_runtime.session_factory.begin() as orm,
    ):
        action = ConversationRepository(orm).stage_action(
            run.run_id,
            execution.epoch,
            1,
            "tc",
            "apply_refund",
            "write",
            {"order_id": "MH001", "reason": "退款"},
        )
    response = client.post(
        f"/agent/approvals/{action.approval_id}",
        json={"session_id": session, "approved": True},
    )
    assert (
        response.status_code == 409
        and response.json()["detail"]["code"] == "run_resume_required"
    )
