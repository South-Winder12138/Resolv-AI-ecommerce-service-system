import importlib
from datetime import timedelta

import pytest

from resolv.db.conversation_base import utc_now
from resolv.models.conversation import ConversationRun
from resolv.schemas.conversation import ConversationError
from tests.conversation_fixtures import MemoryCheckpoints, conversation_factory

__all__ = ["conversation_factory"]


@pytest.fixture
def service(conversation_factory):
    module = importlib.import_module("resolv.services.conversation")
    return module.ConversationService(conversation_factory, MemoryCheckpoints())


def test_same_request_returns_same_run_and_one_user(service):
    conversation = service.create()
    first, created = service.accept(
        conversation.conversation_id, "req", "查 MH001", "agent"
    )
    second, duplicate = service.accept(
        conversation.conversation_id, "req", "查 MH001", "agent"
    )
    assert created and not duplicate
    assert first.run_id == second.run_id
    assert [
        m.text for m in service.detail(conversation.conversation_id, None, 50).messages
    ] == ["查 MH001"]


@pytest.mark.parametrize("question,mode", [("不同内容", "agent"), ("查 MH001", "chat")])
def test_same_request_different_question_conflicts(service, question, mode):
    c = service.create()
    service.accept(c.conversation_id, "req", "查 MH001", "agent")
    with pytest.raises(ConversationError, match="request_conflict"):
        service.accept(c.conversation_id, "req", question, mode)


@pytest.mark.parametrize(
    "status,code",
    [
        ("running", "conversation_busy"),
        ("waiting_approval", "pending_approval_exists"),
        ("interrupted", "run_resume_required"),
        ("finalizing", "run_resume_required"),
    ],
)
@pytest.mark.parametrize("mode", ["agent", "chat"])
def test_busy_pending_interrupted_block_every_new_question(
    service, conversation_factory, status, code, mode
):
    c = service.create()
    run, _ = service.accept(c.conversation_id, "req", "查 MH001", "agent")
    with conversation_factory.begin() as session:
        session.get(ConversationRun, run.run_id).status = status
    with pytest.raises(ConversationError, match=code):
        service.accept(c.conversation_id, "new", "新的问题", mode)


def test_expired_lease_with_live_lock_cannot_resume(service, conversation_factory):
    c = service.create()
    run, _ = service.accept(c.conversation_id, "req", "问题", "agent")
    with service.execution(run.run_id) as execution:
        with conversation_factory.begin() as session:
            session.get(ConversationRun, run.run_id).lease_expires_at = (
                utc_now() - timedelta(seconds=1)
            )
        with (
            pytest.raises(ConversationError, match="conversation_busy"),
            service.execution(run.run_id),
        ):
            pytest.fail("second executor entered")
        execution.assert_owned()


def test_opposite_decision_conflicts(service):
    from resolv.repositories.conversation import ConversationRepository

    c = service.create()
    run, _ = service.accept(c.conversation_id, "req", "取消 MH001", "agent")
    with (
        service.execution(run.run_id) as execution,
        service.session_factory.begin() as session,
    ):
        action = ConversationRepository(session).stage_action(
            run.run_id,
            execution.epoch,
            1,
            "tc",
            "cancel_order",
            "write",
            {"order_id": "MH001"},
        )
    first = service.decide(action.approval_id, c.conversation_id, True)
    assert first.approval_status == "approved"
    assert service.decide(action.approval_id, c.conversation_id, True) == first
    with pytest.raises(ConversationError, match="approval_conflict"):
        service.decide(action.approval_id, c.conversation_id, False)


def test_archive_keeps_audit(service, conversation_factory):
    c = service.create()
    run, _ = service.accept(c.conversation_id, "req", "问题", "agent")
    with pytest.raises(ConversationError):
        service.archive(c.conversation_id)
    with conversation_factory.begin() as session:
        row = session.get(ConversationRun, run.run_id)
        row.status = "failed"
        from resolv.models.conversation import Conversation

        session.get(Conversation, c.conversation_id).current_run_id = None
    service.archive(c.conversation_id)
    assert len(service.detail(c.conversation_id, None, 50).messages) == 1
    assert service.list(None, 50) == []
    with pytest.raises(ConversationError):
        service.accept(c.conversation_id, "new", "问题", "agent")


def test_restart_detects_missing_executor_before_lease_expires(service,conversation_factory):
    c=service.create()
    run,_=service.accept(c.conversation_id,"req","问题","agent")
    with conversation_factory.begin() as session:
        row=session.get(ConversationRun,run.run_id)
        row.execution_owner_id="previous-process"
        row.execution_epoch=1
        row.lease_expires_at=utc_now()+timedelta(seconds=120)
    assert service.detail(c.conversation_id,None,50).current_run.status=="interrupted"


def test_live_named_lock_remains_running_after_lease_expires(service,conversation_factory):
    c=service.create()
    run,_=service.accept(c.conversation_id,"req","问题","agent")
    with service.execution(run.run_id):
        with conversation_factory.begin() as session:
            session.get(ConversationRun,run.run_id).lease_expires_at=utc_now()-timedelta(seconds=1)
        assert service.detail(c.conversation_id,None,50).current_run.status=="running"
