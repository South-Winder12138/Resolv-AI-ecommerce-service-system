import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from resolv.models.conversation import Conversation, ConversationMessage
from tests.test_understanding_service import accept, require
from tests.understanding_fixtures import fail_once


def client_for(service):
    try:
        module = importlib.import_module("resolv.api.understandings")
    except ModuleNotFoundError:
        pytest.fail("understanding router missing")
    app = FastAPI()
    app.state.understanding_runtime = service
    app.include_router(module.router)
    return TestClient(app)


def test_complete_json_api_and_idempotency_do_not_become_chat_turn(
    understanding_service,
):
    s = require(understanding_service)
    with client_for(s) as client:
        payload = {"request_id": "req", "text": "查订单"}
        first = client.post("/conversations/c/understandings", json=payload)
        assert first.status_code == 200
        value = first.json()
        assert (
            value["branch"] == "business"
            and value["route_ready"]
            and value["status"] == "ready"
        )
        id = value["understanding_id"]
        assert client.get(f"/conversations/c/understandings/{id}").json() == value
        assert (
            client.post("/conversations/c/understandings", json=payload).json() == value
        )
        assert (
            client.post(f"/conversations/c/understandings/{id}/resume").json() == value
        )
        assert client.get("/conversations/c/understanding-tasks").json() == {
            "requests": [],
            "next_cursor": None,
        }
        assert s.resolver.model.calls_by_stage == {"resolution": 1, "intents": 1}
    with s.session_factory() as session:
        assert session.query(ConversationMessage).count() == 2
        assert session.get(Conversation, "c").current_run_id is None


@pytest.mark.parametrize(
    "patch",
    [
        {"execute": True},
        {"text": 12},
        {"request_id": ""},
        {"continue_task_id": "t", "clarifies_understanding_id": "p"},
    ],
)
def test_request_fields_are_strict(understanding_service, patch):
    with client_for(require(understanding_service)) as client:
        reply = client.post(
            "/conversations/c/understandings",
            json={"request_id": "req", "text": "查订单"} | patch,
        )
        assert reply.status_code == 422


def test_safe_404_409_and_503(understanding_service):
    with client_for(require(understanding_service)) as client:
        assert (
            client.post(
                "/conversations/foreign/understandings",
                json={"request_id": "r", "text": "查订单"},
            ).status_code
            == 404
        )
        assert client.get("/conversations/c/understandings/foreign").status_code == 404
        client.post(
            "/conversations/c/understandings",
            json={"request_id": "r", "text": "查订单"},
        )
        assert (
            client.post(
                "/conversations/c/understandings",
                json={"request_id": "r", "text": "退款"},
            ).status_code
            == 409
        )
        assert (
            client.get("/conversations/c/understanding-tasks?limit=101").status_code
            == 422
        )
    with client_for(None) as client:
        assert (
            client.post(
                "/conversations/c/understandings",
                json={"request_id": "r", "text": "查订单"},
            ).status_code
            == 503
        )


def test_duplicate_interrupted_request_does_not_automatically_retry(
    understanding_service,
):
    s = require(understanding_service)
    s.fault_hook = fail_once("after_resolution_saved")
    with client_for(s) as client:
        payload = {"request_id": "r", "text": "查订单"}
        reply = client.post("/conversations/c/understandings", json=payload)
        assert reply.status_code == 503
        id = reply.json()["detail"]["understanding_id"]
        duplicate = client.post("/conversations/c/understandings", json=payload)
        assert (
            duplicate.status_code == 200 and duplicate.json()["status"] == "interrupted"
        )
        assert s.resolver.model.calls_by_stage == {"resolution": 1}
        assert (
            client.post(f"/conversations/c/understandings/{id}/resume").json()["status"]
            == "ready"
        )


def test_archived_http_is_readable_but_not_executable(understanding_service):
    s = require(understanding_service)
    row = accept(s)
    s.run("c", row.understanding_id)
    with s.session_factory.begin() as session:
        session.get(Conversation, "c").status = "archived"
    with client_for(s) as client:
        assert (
            client.get(
                f"/conversations/c/understandings/{row.understanding_id}"
            ).status_code
            == 200
        )
        assert (
            client.post(
                f"/conversations/c/understandings/{row.understanding_id}/resume"
            ).status_code
            == 409
        )


def test_driver_errors_are_redacted(understanding_service, monkeypatch):
    from sqlalchemy.exc import OperationalError

    s = require(understanding_service)

    def fail(*_):
        raise OperationalError("password=secret", {}, Exception("secret"))

    monkeypatch.setattr(s, "accept", fail)
    with client_for(s) as client:
        reply = client.post(
            "/conversations/c/understandings",
            json={"request_id": "r", "text": "查订单"},
        )
        assert reply.status_code == 503 and "secret" not in reply.text


def test_validation_uses_safe_independent_error_contract(understanding_service):
    with client_for(require(understanding_service)) as client:
        reply = client.post(
            "/conversations/c/understandings",
            json={"request_id": "r", "text": "查订单", "execute": True},
        )
        assert reply.status_code == 422
        assert reply.json()["detail"] == {
            "code": "invalid_request",
            "retryable": False,
            "understanding_id": None,
            "related_id": None,
        }


def test_unavailable_read_still_identifies_requested_record():
    with client_for(None) as client:
        reply = client.get("/conversations/c/understandings/known")
        assert (
            reply.status_code == 503
            and reply.json()["detail"]["understanding_id"] == "known"
        )


def test_incompatible_stored_schema_identifies_record(understanding_service):
    from resolv.models.understanding import ConversationUnderstanding

    s = require(understanding_service)
    row = accept(s)
    with s.session_factory.begin() as session:
        stored = session.get(ConversationUnderstanding, row.understanding_id)
        stored.snapshot = stored.snapshot | {
            "config": stored.snapshot["config"] | {"state_schema_version": 2}
        }
    with client_for(s) as client:
        reply = client.get(f"/conversations/c/understandings/{row.understanding_id}")
        assert (
            reply.status_code == 409
            and reply.json()["detail"]["understanding_id"] == row.understanding_id
        )
