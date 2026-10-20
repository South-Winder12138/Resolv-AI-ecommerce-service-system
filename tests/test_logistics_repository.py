from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from resolv.db.conversation_base import utc_now
from resolv.models.conversation import ConversationAction, ConversationRun
from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.schemas.conversation import ConversationError
from resolv.schemas.logistics import LogisticsFailure, ShipmentEnvelope
from resolv.services.conversation import ConversationService
from tests.conversation_fixtures import MemoryCheckpoints, conversation_factory
from tests.logistics_fixtures import envelope_payload
from tests.test_logistics_contracts import api

__all__ = ["conversation_factory"]


@pytest.fixture
def staged(conversation_factory):
    service = ConversationService(conversation_factory, MemoryCheckpoints())
    conversation = service.create()
    run, _ = service.accept(conversation.conversation_id, "logistics", "查询运单 MOCKSF001", "agent")
    with service.execution(run.run_id) as execution:
        with conversation_factory.begin() as session:
            row = session.get(ConversationRun, run.run_id)
            row.business_config = {"profile": "frozen-test"}
            action = ConversationRepository(session).stage_action(
                run.run_id, execution.epoch, 1, "tc1", "query_logistics", "read", {"tracking_no": "MOCKSF001"},
            )
        yield service, run, execution, action


def repo_class(service):
    cls = api("resolv.repositories.logistics").LogisticsAttemptRepository
    model = api("resolv.models.logistics")
    model.ConversationLogisticsAttempt.__table__.create(service.session_factory.kw["bind"], checkfirst=True)
    return cls


def prepare(staged):
    service, _, execution, action = staged
    cls = repo_class(service)
    with service.session_factory.begin() as session:
        return cls(session, "local").prepare(
            action.action_id, execution.epoch,
            config_hash=fingerprint({"profile": "frozen-test"}), tracking_no="MOCKSF001",
        )


def result(attempt):
    return ShipmentEnvelope.model_validate_json(__import__("json").dumps(envelope_payload(request_id=attempt.attempt_id)))


def test_prepare_is_owned_and_monotonic(staged):
    service, _, execution, action = staged
    first = prepare(staged)
    assert first.state == "prepared" and first.attempt_no == 1
    assert first.owner_id == "local" and first.execution_epoch == execution.epoch
    cls = api("resolv.repositories.logistics").LogisticsAttemptRepository
    with pytest.raises(ConversationError), service.session_factory.begin() as session:
        cls(session, "other-owner").prepare(action.action_id, execution.epoch, config_hash=first.config_hash, tracking_no="MOCKSF001")
    with pytest.raises(LogisticsFailure, match="logistics_result_unknown"), service.session_factory.begin() as session:
        cls(session, "local").prepare(action.action_id, execution.epoch, config_hash=first.config_hash, tracking_no="MOCKSF001")
    with service.session_factory.begin() as session:
        cls(session, "local").mark_unknown(first.attempt_id, execution.epoch)
    second = prepare(staged)
    assert second.attempt_no == 2 and second.attempt_id != first.attempt_id
    model = api("resolv.models.logistics").ConversationLogisticsAttempt
    with pytest.raises(IntegrityError), service.session_factory.begin() as session:
        row = session.get(model, second.attempt_id)
        session.execute(model.__table__.insert().values(**{
            c.name: ("f" * 32 if c.name == "attempt_id" else getattr(row, c.name))
            for c in model.__table__.columns
        }))


def test_complete_and_action_rollback_together(staged):
    service, run, execution, action = staged
    attempt = prepare(staged)
    cls = api("resolv.repositories.logistics").LogisticsAttemptRepository
    with pytest.raises(RuntimeError, match="rollback"), service.session_factory.begin() as session:
        cls(session, "local").complete(attempt.attempt_id, execution.epoch, result(attempt), {"found": True})
        session.flush()
        raise RuntimeError("rollback")
    with service.session_factory.begin() as session:
        repo = cls(session, "local")
        assert repo.find_completed(run.run_id, attempt.config_hash, "MOCKSF001") is None
        assert session.get(ConversationAction, action.action_id).execution_status == "pending"
        repo.complete(attempt.attempt_id, execution.epoch, result(attempt), {"found": True})
    with service.session_factory() as session:
        saved = cls(session, "local").find_completed(run.run_id, attempt.config_hash, "MOCKSF001")
        assert saved.state == "completed" and saved.result_hash and saved.envelope["simulated"] is True
        assert session.get(ConversationAction, action.action_id).result == {"found": True}


def test_completed_result_cannot_be_overwritten(staged):
    service, _, execution, _ = staged
    attempt = prepare(staged)
    cls = api("resolv.repositories.logistics").LogisticsAttemptRepository
    with service.session_factory.begin() as session:
        cls(session, "local").complete(attempt.attempt_id, execution.epoch, result(attempt), {"found": True})
    with pytest.raises(LogisticsFailure, match="logistics_store_failed"), service.session_factory.begin() as session:
        cls(session, "local").fail(attempt.attempt_id, execution.epoch, LogisticsFailure("logistics_timeout", True))


def test_cached_prepared_row_cannot_overwrite_completed_result(staged):
    service, _, execution, action = staged
    attempt = prepare(staged)
    cls = api("resolv.repositories.logistics").LogisticsAttemptRepository
    model = api("resolv.models.logistics").ConversationLogisticsAttempt
    with service.session_factory() as stale:
        cached_attempt = stale.get(model, attempt.attempt_id)
        cached_action = stale.get(ConversationAction, action.action_id)
        with service.session_factory.begin() as fresh:
            cls(fresh, "local").complete(attempt.attempt_id, execution.epoch, result(attempt), {"found": True})
        assert cached_attempt.state == "prepared" and cached_action.execution_status == "pending"
        with pytest.raises(LogisticsFailure, match="logistics_store_failed"):
            cls(stale, "local").fail(attempt.attempt_id, execution.epoch, LogisticsFailure("logistics_timeout", True))


def test_late_epoch_cannot_complete(staged):
    service, run, execution, _ = staged
    attempt = prepare(staged)
    cls = api("resolv.repositories.logistics").LogisticsAttemptRepository
    with service.session_factory.begin() as session:
        session.get(ConversationRun, run.run_id).execution_epoch += 1
    with pytest.raises(ConversationError, match="conversation_busy"), service.session_factory.begin() as session:
        cls(session, "local").complete(attempt.attempt_id, execution.epoch, result(attempt), {"found": True})
    with service.session_factory() as session:
        model = api("resolv.models.logistics").ConversationLogisticsAttempt
        assert session.scalar(select(model)).state == "prepared"


def test_expired_lease_cannot_prepare(staged):
    service, run, _, _ = staged
    with service.session_factory.begin() as session:
        session.get(ConversationRun, run.run_id).lease_expires_at = utc_now() - timedelta(seconds=1)
    with pytest.raises(ConversationError, match="conversation_busy"):
        prepare(staged)


def test_wrong_tracking_or_config_cannot_prepare(staged):
    service, _, execution, action = staged
    cls = api("resolv.repositories.logistics").LogisticsAttemptRepository
    for config, tracking in [("0" * 64, "MOCKSF001"), (fingerprint({"profile": "frozen-test"}), "OTHER")]:
        with pytest.raises(LogisticsFailure), service.session_factory.begin() as session:
            cls(session, "local").prepare(action.action_id, execution.epoch, config_hash=config, tracking_no=tracking)
