"""Real short SQLite transactions around a controlled remote boundary."""

import json
from contextlib import contextmanager

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import SQLAlchemyError

from resolv.models.conversation import ConversationAction, ConversationRun
from resolv.models.logistics import ConversationLogisticsAttempt as Attempt
from resolv.repositories.conversation import ConversationRepository
from resolv.schemas.conversation import ConversationError
from resolv.schemas.logistics import LogisticsFailure, ShipmentEnvelope
from tests.conversation_fixtures import conversation_factory
from tests.logistics_fixtures import envelope_payload
from tests.test_logistics_contracts import api
from tests.test_logistics_mcp_server import profile
from tests.test_logistics_repository import repo_class, staged

__all__ = ["conversation_factory", "staged"]


class Crash(BaseException):
    pass


class Client:
    def __init__(self, inspect_send=None):
        self.profile, self.calls, self.inspect_send = profile(), [], inspect_send
        self.error = None

    def query(self, tracking_no, *, request_id):
        self.calls.append((tracking_no, request_id))
        if self.inspect_send:
            self.inspect_send(tracking_no, request_id)
        if self.error:
            raise self.error
        return ShipmentEnvelope.model_validate_json(json.dumps(envelope_payload(tracking_no=tracking_no, request_id=request_id)))


@pytest.fixture
def ready(staged):
    service, run, _, _ = staged
    repo_class(service)
    with service.session_factory.begin() as session:
        session.get(ConversationRun, run.run_id).business_config = {
            "config_version": 2, "logistics_backend": profile().model_dump(mode="json"),
        }
    return staged


def reader(ready, client=None, **kwargs):
    cls = api("resolv.services.logistics_execution").LogisticsReadService
    return cls(ready[0].session_factory, client or Client(), profile(), **kwargs)


def test_prepare_committed_before_network_and_no_transaction_held(ready):
    service, _, execution, action = ready
    observed = []

    def inspect_send(tracking, request_id):
        assert service.session_factory.kw["bind"].pool.checkedout() == 0
        with service.session_factory() as session:
            row = session.get(Attempt, request_id)
            assert row.state == "prepared" and row.tracking_no == tracking
            assert session.get(ConversationAction, action.action_id).execution_status == "pending"
            observed.append(row.attempt_id)

    client = Client(inspect_send)
    value = reader(ready, client).read(action.action_id, execution)
    assert observed == [client.calls[0][1]]
    assert value.output["found"] is True and value.output["location"] == "上海转运中心"
    assert value.output["provenance"]["simulated"] is True
    assert value.output["provenance"]["attempt_id"] == observed[0]


def test_same_run_same_tracking_reuses_observation(ready):
    service, run, execution, action = ready
    client = Client()
    reads = reader(ready, client)
    first = reads.read(action.action_id, execution)
    with service.session_factory.begin() as session:
        alias = ConversationRepository(session).stage_action(run.run_id, execution.epoch, 2, "tc2", "query_logistics", "read", {"tracking_no": "MOCKSF001"})
    second = reads.read(alias.action_id, execution)
    assert second.tool_call_id == "tc2" and second.output == first.output
    assert len(client.calls) == 1
    with service.session_factory() as session:
        assert len(session.scalars(select(Attempt)).all()) == 1
        assert session.get(ConversationAction, alias.action_id).execution_status == "completed"


@pytest.mark.parametrize("crash_at, prior_calls", [("after_attempt_prepared", 0), ("after_remote_result", 1)])
def test_unknown_requires_explicit_resume(ready, crash_at, prior_calls):
    service, _, execution, action = ready
    interrupted = False

    def hook(stage, attempt):
        nonlocal interrupted
        assert attempt.attempt_id
        if stage == crash_at and not interrupted:
            interrupted = True
            raise Crash()

    client = Client()
    reads = reader(ready, client, fault_hook=hook)
    with pytest.raises(Crash):
        reads.read(action.action_id, execution)
    assert len(client.calls) == prior_calls
    with pytest.raises(ConversationError, match="logistics_result_unknown"):
        reads.read(action.action_id, execution)
    assert len(client.calls) == prior_calls
    value = reads.read(action.action_id, execution, explicit_resume=True)
    assert value.output["found"] is True and len(client.calls) == prior_calls + 1
    with service.session_factory() as session:
        rows = session.scalars(select(Attempt).order_by(Attempt.attempt_no)).all()
        assert [(r.attempt_no, r.state) for r in rows] == [(1, "unknown"), (2, "completed")]


def test_commit_ack_failure_reconciles_without_second_read(ready):
    service, _, execution, action = ready
    lose_ack = False
    lost = []

    def after_flush(session, context):
        nonlocal lose_ack
        if any(isinstance(row, Attempt) and row.state == "completed" for row in session.dirty):
            session.info["completed_logistics"] = True
            lose_ack = True

    def after_commit(session):
        nonlocal lose_ack
        if lose_ack and session.info.get("completed_logistics"):
            lose_ack = False
            lost.append("after_commit")
            raise SQLAlchemyError("private commit acknowledgement lost")

    client = Client()
    event.listen(service.session_factory.class_, "after_flush", after_flush)
    event.listen(service.session_factory.class_, "after_commit", after_commit)
    try:
        value = reader(ready, client).read(action.action_id, execution)
    finally:
        event.remove(service.session_factory.class_, "after_commit", after_commit)
        event.remove(service.session_factory.class_, "after_flush", after_flush)
    assert value.output["found"] is True and len(client.calls) == 1
    assert lost == ["after_commit"]
    with service.session_factory() as session:
        assert session.get(ConversationAction, action.action_id).execution_status == "completed"


def test_completed_action_replays_with_server_offline(ready):
    _, _, execution, action = ready
    client = Client()
    first = reader(ready, client).read(action.action_id, execution)
    client.error = LogisticsFailure("logistics_transport_failed", True)
    second = reader(ready, client).read(action.action_id, execution)
    assert second == first and len(client.calls) == 1


def test_stale_epoch_result_is_discarded(ready):
    service, run, execution, action = ready

    def revoke(tracking, request_id):
        with service.session_factory.begin() as session:
            session.get(ConversationRun, run.run_id).execution_epoch += 1

    client = Client(revoke)
    with pytest.raises(ConversationError, match="conversation_busy"):
        reader(ready, client).read(action.action_id, execution)
    with service.session_factory() as session:
        assert session.get(ConversationAction, action.action_id).execution_status == "pending"
        assert session.scalar(select(Attempt)).state == "prepared"


def test_explicit_resume_sends_at_most_once_on_repeated_failure(ready):
    _, _, execution, action = ready
    client = Client()
    client.error = LogisticsFailure("logistics_timeout", True)
    reads = reader(ready, client)
    for explicit, expected_calls in [(False, 1), (False, 1), (True, 2)]:
        with pytest.raises(ConversationError, match="logistics_timeout"):
            reads.read(action.action_id, execution, explicit_resume=explicit)
        assert len(client.calls) == expected_calls


def test_profile_drift_is_rejected_before_network(ready):
    service, run, execution, action = ready
    with service.session_factory.begin() as session:
        row = session.get(ConversationRun, run.run_id)
        row.business_config = {**row.business_config, "logistics_backend": {**row.business_config["logistics_backend"], "dataset_version": "other"}}
    client = Client()
    with pytest.raises(ConversationError, match="logistics_configuration_conflict"):
        reader(ready, client).read(action.action_id, execution)
    assert client.calls == []


def test_entire_frozen_config_change_during_network_prevents_result_commit(ready):
    service, run, execution, action = ready

    def change_config(tracking, request_id):
        with service.session_factory.begin() as session:
            row = session.get(ConversationRun, run.run_id)
            row.business_config = {**row.business_config, "prompt_version": "changed"}

    client = Client(change_config)
    with pytest.raises(ConversationError, match="logistics_configuration_conflict"):
        reader(ready, client).read(action.action_id, execution)
    with service.session_factory() as session:
        assert session.get(ConversationAction, action.action_id).execution_status == "pending"
        assert session.scalar(select(Attempt)).state == "prepared"


def test_observation_cache_does_not_cross_runs(ready):
    service, _, execution, action = ready
    client = Client()
    reads = reader(ready, client)
    reads.read(action.action_id, execution)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "new-run", "查询物流", "agent")
    with service.execution(run.run_id) as other:
        with service.session_factory.begin() as session:
            session.get(ConversationRun, run.run_id).business_config = {"config_version": 2, "logistics_backend": profile().model_dump(mode="json")}
            action2 = ConversationRepository(session).stage_action(run.run_id, other.epoch, 1, "other-tc", "query_logistics", "read", {"tracking_no": "MOCKSF001"})
        reads.read(action2.action_id, other)
    assert len(client.calls) == 2 and client.calls[0][1] != client.calls[1][1]


def test_store_failure_before_prepare_never_dispatches(ready):
    _, _, execution, action = ready
    client = Client()
    reads = reader(ready, client)

    class FailingFactory:
        @contextmanager
        def begin(self):
            raise SQLAlchemyError("private database unavailable")
            yield  # pragma: no cover

    reads.session_factory = FailingFactory()
    with pytest.raises(ConversationError, match="logistics_store_failed"):
        reads.read(action.action_id, execution)
    assert client.calls == []
