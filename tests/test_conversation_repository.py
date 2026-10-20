import importlib

import pytest

from resolv.schemas.conversation import ConversationError
from tests.conversation_fixtures import MemoryCheckpoints, conversation_factory

__all__ = ["conversation_factory"]


@pytest.fixture
def owned(conversation_factory):
    module = importlib.import_module("resolv.services.conversation")
    service = module.ConversationService(conversation_factory, MemoryCheckpoints())
    c = service.create()
    run, _ = service.accept(c.conversation_id, "req", "取消 MH001", "agent")
    with service.execution(run.run_id) as execution:
        yield service, run, execution


def test_stage_action_is_frozen(owned):
    from resolv.repositories.conversation import ConversationRepository

    service, run, execution = owned
    with service.session_factory.begin() as session:
        repository = ConversationRepository(session)
        first = repository.stage_action(
            run.run_id,
            execution.epoch,
            1,
            "tc",
            "cancel_order",
            "write",
            {"order_id": "MH001"},
        )
        second = repository.stage_action(
            run.run_id,
            execution.epoch,
            1,
            "tc",
            "cancel_order",
            "write",
            {"order_id": "MH001"},
        )
        assert first.action_id == second.action_id
    with (
        pytest.raises(ConversationError, match="action_conflict"),
        service.session_factory.begin() as session,
    ):
        ConversationRepository(session).stage_action(
            run.run_id,
            execution.epoch,
            1,
            "tc",
            "cancel_order",
            "write",
            {"order_id": "MH002"},
        )


def test_old_epoch_cannot_publish(owned):
    from resolv.repositories.conversation import ConversationRepository

    service, run, execution = owned
    with (
        pytest.raises(ConversationError, match="conversation_busy"),
        service.session_factory.begin() as session,
    ):
        ConversationRepository(session).save_knowledge(
            run.run_id, execution.epoch - 1, []
        )


def test_malformed_args_against_frozen_action_remain_conflict(owned):
    from resolv.repositories.conversation import ConversationRepository

    service, run, execution = owned
    with service.session_factory.begin() as session:
        ConversationRepository(session).stage_action(
            run.run_id,
            execution.epoch,
            1,
            "tc",
            "cancel_order",
            "write",
            {"order_id": "MH001"},
        )
    with (
        pytest.raises(ConversationError, match="action_conflict"),
        service.session_factory.begin() as session,
    ):
        ConversationRepository(session).stage_action(
            run.run_id,
            execution.epoch,
            1,
            "tc",
            "cancel_order",
            "write",
            {"order_id": "MH001", "extra": True},
        )


@pytest.mark.parametrize(
    "args",
    [
        {"order_id": "MH001", "approved": True},
        {"order_id": float("nan")},
        {"order_id": object()},
    ],
)
def test_invalid_action_cannot_be_staged(owned, args):
    from resolv.repositories.conversation import ConversationRepository

    service, run, execution = owned
    with pytest.raises(ConversationError), service.session_factory.begin() as session:
        ConversationRepository(session).stage_action(
            run.run_id, execution.epoch, 1, "tc", "cancel_order", "write", args
        )
