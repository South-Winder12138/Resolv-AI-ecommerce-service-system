from datetime import UTC, datetime, timedelta, timezone
from importlib import import_module

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError


def test_status_and_summary_shape_are_strict():
    schemas = import_module("resolv.schemas.conversation")
    assert schemas.RunStatus.COMPLETED.value == "completed"
    with pytest.raises(ValidationError):
        schemas.ConversationSummary(
            user_goal="",
            confirmed_facts=[],
            completed_actions=[],
            open_questions=[],
            approved=True,
        )
    with pytest.raises(ValidationError):
        schemas.ConversationSummary(
            user_goal="", confirmed_facts=[42], completed_actions=[], open_questions=[]
        )


def test_utc_round_trip(tmp_path):
    from resolv.db.engine import create_db_engine, create_session_factory

    models = import_module("resolv.models.conversation")
    engine = create_db_engine(f"sqlite:///{tmp_path / 'conversation.db'}")
    models.Conversation.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        session.add(
            models.Conversation(
                conversation_id="c",
                thread_id="c",
                title="你好",
                created_at=datetime(
                    2026, 10, 12, 10, tzinfo=timezone(timedelta(hours=8))
                ),
            )
        )
    with factory() as session:
        assert session.get(models.Conversation, "c").created_at == datetime(
            2026, 10, 12, 2, tzinfo=UTC
        )
    engine.dispose()


@pytest.mark.parametrize(
    "table, rows",
    [
        (
            "ConversationRun",
            [
                {"run_id": "r", "request_id": "req"},
                {"run_id": "other", "request_id": "req"},
            ],
        ),
        (
            "ConversationMessage",
            [
                {"message_id": "m", "seq": 1, "role": "user"},
                {"message_id": "n", "seq": 1, "role": "assistant"},
            ],
        ),
        (
            "ConversationMessage",
            [
                {"message_id": "m", "seq": 1, "role": "user"},
                {"message_id": "n", "seq": 2, "role": "user"},
            ],
        ),
        (
            "ConversationAction",
            [
                {"action_id": "a", "tool_call_id": "tc"},
                {"action_id": "b", "tool_call_id": "tc"},
            ],
        ),
    ],
)
def test_unique_request_message_sequence_action_constraints(tmp_path, table, rows):
    from resolv.db.engine import create_db_engine, create_session_factory

    models = import_module("resolv.models.conversation")
    engine = create_db_engine(f"sqlite:///{tmp_path / 'unique.db'}")
    models.Conversation.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        session.add(models.Conversation(conversation_id="c", thread_id="c", title=""))
    if table != "ConversationRun":
        with factory.begin() as session:
            session.add(
                models.ConversationRun(
                    run_id="r",
                    conversation_id="c",
                    request_id="req",
                    request_fingerprint="f" * 64,
                    question_message_id="m",
                    answer_message_id="answer",
                )
            )
    with pytest.raises(IntegrityError), factory.begin() as session:
        for row in rows:
            defaults = {"conversation_id": "c"}
            if table == "ConversationRun":
                defaults.update(
                    request_fingerprint="f" * 64,
                    question_message_id="q",
                    answer_message_id="answer",
                )
            elif table == "ConversationMessage":
                defaults.update(run_id="r", text="hello")
            else:
                defaults = {
                    "run_id": "r",
                    "model_step": 1,
                    "tool_name": "get_order",
                    "effect": "read",
                    "arguments": {},
                    "arguments_fingerprint": "f" * 64,
                }
            session.add(getattr(models, table)(**(defaults | row)))
    engine.dispose()
