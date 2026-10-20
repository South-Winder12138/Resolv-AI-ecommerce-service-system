from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from resolv.db.checkpoints import CheckpointError
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.models.conversation import Conversation
from resolv.models.understanding import ConversationUnderstanding  # noqa: F401
from resolv.schemas.conversation import (
    ActionRecord,
    CompletedTurn,
    MessageRecord,
    RunRecord,
    SummaryRecord,
)


@pytest.fixture
def conversation_factory(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'conversations.db'}")
    Conversation.metadata.create_all(engine)
    yield create_session_factory(engine)
    engine.dispose()


class MemoryCheckpoints:
    def __init__(self):
        self.saver = InMemorySaver()
        self.locks = {}

    @contextmanager
    def __call__(self, *, conversation_id):
        lock = self.locks.setdefault(conversation_id, Lock())
        if not lock.acquire(blocking=False):
            raise CheckpointError("conversation_busy")
        active = True

        def assert_owned():
            if not active:
                raise CheckpointError()

        try:
            yield SimpleNamespace(saver=self.saver, assert_owned=assert_owned)
        finally:
            active = False
            lock.release()


def make_run(**overrides):
    now = datetime.now(UTC)
    return RunRecord(
        **(
            {
                "run_id": "r",
                "conversation_id": "c",
                "request_id": "req",
                "request_fingerprint": "f" * 64,
                "mode": "agent",
                "status": "running",
                "attempt": 1,
                "execution_epoch": 1,
                "question_message_id": "q",
                "answer_message_id": "a",
                "question": "查 MH001",
                "model_name": "fake",
                "prompt_version": "v1",
                "system_prompt_snapshot": "客服",
                "tool_schema_fingerprint": "",
                "state_schema_version": 1,
                "created_at": now,
                "updated_at": now,
            }
            | overrides
        )
    )


def make_action(**overrides):
    return ActionRecord(
        **(
            {
                "action_id": "action",
                "run_id": "r",
                "model_step": 1,
                "tool_call_id": "tc",
                "tool_name": "cancel_order",
                "effect": "write",
                "arguments": {"order_id": "MH001"},
                "arguments_fingerprint": "f" * 64,
                "approval_id": "approval",
                "approval_status": "approved",
                "execution_status": "pending",
            }
            | overrides
        )
    )


def make_turn(seq=1, question="问题", answer="回答", **overrides):
    run_id = uuid4().hex

    def message(role, text, number):
        return MessageRecord(
            message_id=uuid4().hex,
            conversation_id="c",
            run_id=run_id,
            seq=number,
            role=role,
            text=text,
            created_at=datetime.now(UTC),
        )

    return CompletedTurn(
        **(
            {
                "user": message("user", question, seq),
                "assistant": message("assistant", answer, seq + 1),
            }
            | overrides
        )
    )


def make_summary(**overrides):
    return SummaryRecord(
        **(
            {
                "conversation_id": "c",
                "version": 1,
                "content": {
                    "user_goal": "",
                    "confirmed_facts": [],
                    "completed_actions": [],
                    "open_questions": [],
                },
                "covered_until_seq": 2,
                "model_name": "fake",
                "created_at": datetime.now(UTC),
                "source_run_id": "old",
            }
            | overrides
        )
    )
