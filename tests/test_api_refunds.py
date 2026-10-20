"""HTTP boundary tests use only owned temporary stores and explicit adapters."""

from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.refund_submission_fixtures import flow_harness, submission_harness
from tests.test_api_agent_stream import events
from tests.test_refund_submission import counts

__all__ = ["flow_harness", "submission_harness"]


@pytest.fixture
def refund_http(submission_harness, monkeypatch):
    import resolv.main as module

    h = submission_harness
    h.service.refund_runtime = SimpleNamespace(submissions=h.submitter)

    @contextmanager
    def runtime():
        yield h.service

    monkeypatch.setattr(module, "routed_conversation_runtime", runtime)
    with TestClient(module.app) as client:
        module.app.state.refund_wait_seconds = 5
        yield SimpleNamespace(
            **(
                vars(h)
                | {
                    "client": client,
                    "app": module.app,
                    "url": f"/conversations/{h.offer.conversation_id}/refund-offers/{h.offer.offer_id}",
                }
            )
        )


@pytest.mark.parametrize(
    "body",
    [
        {"request_id": "r", "confirm": 1},
        {"request_id": "r", "confirm": "true"},
        {"request_id": "r", "confirm": False},
        {"request_id": " ", "confirm": True},
        {"request_id": "r"},
        {"request_id": "r", "confirm": True, "order_id": "OTHER"},
        {"request_id": "r", "confirm": True, "owner_id": "claim"},
        {"request_id": "r", "confirm": True, "money": 1},
    ],
)
def test_refund_http_ownership_and_strict_confirm(refund_http, body):
    h = refund_http
    response = h.client.post(h.url + "/submit", json=body)
    assert response.status_code == 422 and counts(h) == (0, 0)


def post(h, **kwargs):
    return h.client.post(
        h.url + "/submit", json={"request_id": "request", "confirm": True}, **kwargs
    )


def test_http_pending_replay_and_public_whitelist(refund_http):
    h = refund_http
    first = post(h)
    assert first.status_code == 200, first.text
    data = first.json()
    assert data["receipt"]["status"] == "pending" and data["receipt"]["application_id"]
    assert post(h).json() == data and counts(h) == (1, 1)
    detail = h.client.get(h.url)
    assert detail.status_code == 200 and detail.json()["validity"] == "submitted"
    for secret in (
        "subject_ref",
        "execution_epoch",
        "lease_expires_at",
        "critical_facts",
        "model_requests",
        "profile",
        "worker_id",
    ):
        assert secret not in first.text + detail.text


def test_owned_paths_never_reveal_another_order(refund_http):
    h = refund_http
    assert (
        h.client.get(h.url.replace(h.offer.conversation_id, "other", 1)).status_code
        == 404
    )
    assert h.client.get(h.url + "unknown").status_code == 404
    h.submitter.owner_id = "another-owner"
    response = h.client.get(h.url)
    assert response.status_code == 404 and "A001" not in response.text
    assert counts(h) == (0, 0)


@pytest.mark.parametrize(
    "change,status",
    [("expire", 409), ("identity", 503), ("transport", 503), ("archive", 409)],
)
def test_http_denial_codes_keep_command_association(refund_http, change, status):
    h = refund_http
    if change == "expire":
        h.clock.now = h.offer.expires_at
    elif change == "identity":
        from resolv.services.refund_adapters import UnconnectedRefundIdentity

        h.submitter.identity = UnconnectedRefundIdentity()
    elif change == "transport":
        h.facts.fail = TimeoutError("private-driver-secret")
    else:
        from resolv.models.conversation import Conversation

        with h.factory.begin() as s:
            s.get(Conversation, h.offer.conversation_id).status = "archived"
    response = post(h, headers={"X-Customer-ID": "fixture-subject"})
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert {"code", "retryable", "run_id", "submission_id"} <= detail.keys()
    if change in {"expire", "transport"}:
        assert detail["submission_id"]
    assert "private-driver-secret" not in response.text and counts(h) == (0, 0)


def test_submission_get_has_no_side_effects_and_timeout_keeps_original_job(refund_http):
    h = refund_http
    entered, release = Event(), Event()
    original = h.facts.query_order

    def block(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    h.facts.query_order = block
    h.app.state.refund_wait_seconds = 0.01
    try:
        response = post(h)
        assert response.status_code == 202, response.text
        cid = response.json()["submission_id"]
        assert entered.wait(2)
        before = h.counts.copy()
        url = f"/conversations/{h.offer.conversation_id}/refund-submissions/{cid}"
        for _ in range(3):
            read = h.client.get(url)
            assert read.status_code == 200 and read.json()["status"] == "running"
        assert h.counts == before and counts(h) == (0, 0)
        assert h.client.post(url + "/resume", json={"money": 1}).status_code == 422
    finally:
        release.set()
    h.app.state.refund_wait_seconds = 5
    finished = h.client.post(url + "/resume")
    assert finished.status_code == 200, finished.text
    assert finished.json()["submission_id"] == cid and counts(h) == (1, 1)


def test_archived_submitted_receipt_survives_expiry_and_get(refund_http):
    h = refund_http
    original = post(h).json()
    h.clock.now = h.offer.expires_at
    assert (
        h.client.delete(f"/chat/sessions/{h.offer.conversation_id}").status_code == 200
    )
    assert h.client.get(h.url).json()["submission"]["receipt"] == original["receipt"]
    assert post(h).json()["receipt"] == original["receipt"]


def test_old_submission_does_not_change_new_chat(refund_http):
    h = refund_http
    later, _ = h.service.accept(h.offer.conversation_id, "later", "继续", "chat")
    before = h.client.get(f"/conversations/{h.offer.conversation_id}").json()
    assert post(h).status_code == 200
    after = h.client.get(f"/conversations/{h.offer.conversation_id}").json()
    assert after["current_run"]["run_id"] == later.run_id
    assert after["messages"] == before["messages"] and after["runs"] == before["runs"]


def test_v3_sync_sse_detail_equivalence(tmp_path, monkeypatch):
    import resolv.main as module
    from tests.test_refund_routing_graph import harness

    with harness(tmp_path) as h:

        @contextmanager
        def runtime():
            yield h.service

        monkeypatch.setattr(module, "routed_conversation_runtime", runtime)
        with TestClient(module.app) as client:
            assert list(h.service.run(h.run.run_id))[-1].type == "done"
            body = {
                "session_id": h.run.conversation_id,
                "request_id": h.run.request_id,
                "question": h.run.question,
            }
            sync = client.post("/agent/chat", json=body)
            assert sync.status_code == 200, sync.text
            data = sync.json()
            assert data["schema_version"] == 3 and data["refund"]["offer_id"]
            stream = client.post("/agent/stream", json=body)
            message = next(d for e, d in events(stream) if e == "message")
            detail = client.get(f"/conversations/{h.run.conversation_id}").json()
            metadata = detail["runs"][0]["result_json"]
            assert (
                {k: data[k] for k in metadata}
                == {k: message[k] for k in metadata}
                == metadata
            )


def test_chat_request_preserves_reassessment_link():
    from resolv.api.chat import ChatRequest

    value = ChatRequest(
        session_id="c",
        request_id="r",
        question="新原因",
        reassesses_refund_offer_id="offer",
    )
    assert value.routing().reassesses_refund_offer_id == "offer"


def test_offer_is_not_public_or_submittable_before_chat_completed(refund_http):
    from resolv.models.conversation import ConversationRun

    h = refund_http
    with h.factory.begin() as s:
        s.get(ConversationRun, h.offer.run_id).status = "finalizing"
    assert h.client.get(h.url).status_code == 404
    assert post(h).status_code == 404 and counts(h) == (0, 0)


def test_duplicate_raw_confirmation_keys_rejected(refund_http):
    h = refund_http
    response = h.client.post(
        h.url + "/submit",
        content='{"request_id":"r","confirm":false,"confirm":true}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422 and counts(h) == (0, 0)


def test_receipt_read_refreshes_stale_command_snapshot(refund_http):
    from resolv.repositories.refund_submission import RefundSubmissionRepository

    h = refund_http
    accepted = h.submitter.accept_submission(
        h.offer.conversation_id, h.offer.offer_id, h.request, {}
    )
    with h.factory() as s:
        stale = RefundSubmissionRepository(s, "local").get(accepted.submission_id)
    result = h.submitter.execute(accepted.submission_id, {})
    assert result.receipt
    assert h.submitter._business_receipt(stale, h.offer) == result.receipt


def test_explicit_revoked_authority_is_not_a_transport_failure(refund_http):
    from tests.refund_fixtures import observation

    h = refund_http
    h.facts.authorize = lambda _, bound: observation(
        bound=bound, fields={"authorized": False}
    )
    result = post(h)
    assert result.status_code == 409, result.text
    assert result.json()["detail"]["code"] == "refund_needs_reassessment"
    assert counts(h) == (0, 0)


def test_resume_timeout_returns_pending_instead_of_previous_attempt_error(refund_http):
    h = refund_http

    def fail(point):
        if point == "refund_before_business_begin":
            raise OSError("temporary")

    h.submitter.fault_hook = fail
    first = post(h)
    assert first.status_code == 503
    cid = first.json()["detail"]["submission_id"]
    h.submitter.fault_hook = None
    entered, release = Event(), Event()
    original = h.facts.query_order

    def block(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    h.facts.query_order = block
    h.app.state.refund_wait_seconds = 0.01
    try:
        result = h.client.post(
            f"/conversations/{h.offer.conversation_id}/refund-submissions/{cid}/resume"
        )
        assert result.status_code == 202, result.text
        assert (
            result.json()["submission_id"] == cid
            and result.json()["error_code"] is None
        )
    finally:
        release.set()


def test_receipt_absence_read_is_one_consistent_sqlite_snapshot(
    refund_http, monkeypatch
):
    from resolv.repositories.refund_application import RefundApplicationRepository
    from resolv.repositories.refund_submission import RefundSubmissionRepository

    h = refund_http
    with h.business.kw["bind"].connect() as c:
        c.exec_driver_sql("PRAGMA journal_mode=WAL")
    accepted = h.submitter.accept_submission(
        h.offer.conversation_id, h.offer.offer_id, h.request, {}
    )
    with h.factory() as s:
        stale = RefundSubmissionRepository(s, "local").get(accepted.submission_id)
    original, fired = RefundApplicationRepository.receipt, False

    def read(repo, cid):
        nonlocal fired
        value = original(repo, cid)
        if not fired:
            fired = True
            assert value is None
            assert h.submitter.execute(cid, {}).receipt is not None
        return value

    monkeypatch.setattr(RefundApplicationRepository, "receipt", read)
    assert h.submitter._business_receipt(stale, h.offer) is None
    assert h.submitter._business_receipt(stale, h.offer) is not None
