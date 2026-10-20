import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from resolv.core import config
from resolv.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
    ConversationSummaryRecord,
)
from resolv.repositories.conversation import ConversationRepository
from resolv.schemas.conversation import ConversationError, ConversationSummary
from resolv.services.conversation import ConversationService
from tests.conversation_fixtures import (
    MemoryCheckpoints,
    conversation_factory,
    make_action,
    make_summary,
    make_turn,
)
from tests.test_context import Counter

__all__ = ["conversation_factory"]


class Model:
    calls = 0
    fail_on_call = None
    response = '{"user_goal":"MH001", "confirmed_facts":[], "completed_actions":[], "open_questions":[]}'

    def invoke(self, messages):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("fake summary unavailable")
        return AIMessage(content=self.response)


@pytest.fixture
def summary_harness():
    module = importlib.import_module("resolv.services.summarization")
    model, counter = Model(), Counter()
    settings = config.SummaryModelSettings(
        _env_file=None,
        model_name="fake-summary",
        api_key="test-key",
        base_url="https://fake.invalid/v1",
        model_context_window_tokens=16000,
        reserved_output_tokens=1000,
        input_max_tokens=12000,
        summary_max_tokens=800,
    )
    return SimpleNamespace(
        module=module,
        model=model,
        counter=counter,
        settings=settings,
        service=module.SummaryService(model, counter, settings),
    )


@pytest.mark.parametrize("length,success", [(800, True), (801, False)])
def test_summary_800_vs_801(summary_harness, length, success):
    h = summary_harness
    h.counter.count_text = lambda text: length
    if success:
        assert (
            h.service.create_candidate(None, [make_turn()]).content.user_goal == "MH001"
        )
    else:
        with pytest.raises(h.module.SummaryFailure):
            h.service.create_candidate(None, [make_turn()])


def test_watermark_is_program_owned(summary_harness):
    h = summary_harness
    candidate = h.service.create_candidate(make_summary(), [make_turn(seq=3)])
    assert candidate.covered_until_seq == 4 and candidate.previous_version == 1
    h.model.response = '{"user_goal":"", "confirmed_facts":[], "completed_actions":[], "open_questions":[], "covered_until_seq":999}'
    with pytest.raises(h.module.SummaryFailure):
        h.service.create_candidate(None, [make_turn()])


def test_summary_overhead_rejects_input_before_model_call(summary_harness):
    h = summary_harness
    h.settings.provider_input_overhead_tokens = 5000
    h.counter.total = 7001
    with pytest.raises(h.module.SummaryFailure):
        h.service.create_candidate(None, [make_turn()])
    assert h.model.calls == 0


@pytest.mark.parametrize(
    "missing",
    [None, "user_goal", "confirmed_facts", "completed_actions", "open_questions"],
)
def test_model_summary_requires_all_four_fields(summary_harness, missing):
    h = summary_harness
    value = json.loads(h.model.response)
    if missing is None:
        value = {}
    else:
        value.pop(missing)
    h.model.response = json.dumps(value)
    with pytest.raises(h.module.SummaryFailure):
        h.service.create_candidate(make_summary(), [make_turn(seq=3)])


def test_missing_summary_fields_never_publish_or_advance_watermark(memory_service):
    service, run, _counter, h = memory_service
    with service.session_factory.begin() as session:
        old_user = session.scalar(
            select(ConversationMessage).where(
                ConversationMessage.run_id != run.run_id,
                ConversationMessage.role == "user",
            )
        )
        old_user.text = "x" * 4800
    h.model.response = "{}"
    with service.execution(run.run_id) as execution:
        prepared = service.prepare_memory(run.run_id, execution)
    assert h.model.calls == 1
    assert prepared.summary_version == 0
    with service.session_factory() as session:
        assert session.scalar(select(ConversationSummaryRecord)) is None


def test_summary_batches_carry_only_their_associated_action_facts(summary_harness):
    h = summary_harness
    h.settings.input_max_tokens = 2000
    turns = []
    for index in range(3):
        action = make_action(
            action_id=f"action-{index}",
            effect="read",
            approval_status="not_required",
            execution_status="completed",
            result={"payload": "x" * 800},
        )
        turns.append(make_turn(seq=2 * index + 1, actions=[action]))
    candidate = h.service.create_candidate(None, turns)
    assert candidate.covered_until_seq == 6
    assert h.model.calls == 3


def test_summary_cannot_invent_completed_actions(summary_harness):
    h = summary_harness
    h.model.response = '{"user_goal":"MH001", "confirmed_facts":[], "completed_actions":["已退款"], "open_questions":[]}'
    with pytest.raises(h.module.SummaryFailure):
        h.service.create_candidate(None, [make_turn()])


@pytest.fixture
def memory_service(conversation_factory, summary_harness):
    from resolv.services.context import ContextBuilder

    counter = Counter()
    settings = config.ContextSettings(
        _env_file=None,
        model_context_window_tokens=20000,
        tokenizer_profile="cl100k_base",
    )
    service = ConversationService(
        conversation_factory,
        MemoryCheckpoints(),
        context_builder=ContextBuilder(settings, counter),
        summary_service=summary_harness.service,
    )
    c = service.create()
    old, _ = service.accept(c.conversation_id, "old", "old q", "chat")
    with conversation_factory.begin() as session:
        row = session.get(ConversationRun, old.run_id)
        row.status = "completed"
        session.add(
            ConversationMessage(
                message_id=old.answer_message_id,
                conversation_id=c.conversation_id,
                run_id=old.run_id,
                seq=2,
                role="assistant",
                text="old a",
            )
        )
        session.get(Conversation, c.conversation_id).next_message_seq = 3
        session.get(Conversation, c.conversation_id).current_run_id = None
    run, _ = service.accept(
        c.conversation_id, "current", "current pending question", "chat"
    )
    return service, run, counter, summary_harness


@pytest.mark.parametrize("tokens,calls", [(4799, 0), (4800, 1)])
def test_soft_trigger_4799_vs_4800(memory_service, tokens, calls):
    service, run, counter, h = memory_service
    counter.count_text = lambda text: (
        tokens // 2 + (tokens % 2 if text == "old q" else 0)
        if text in {"old q", "old a"}
        else len(text)
    )
    with service.execution(run.run_id) as execution:
        saved = service.prepare_memory(run.run_id, execution)
    assert h.model.calls == calls
    assert saved.summary_version == calls


def test_current_pending_turn_excluded(memory_service):
    service, run, _, h = memory_service
    with service.execution(run.run_id) as execution:
        saved = service.prepare_memory(run.run_id, execution)
    assert saved.history_cutoff_seq == 2
    assert h.model.calls == 0


def test_failure_fallback_uses_all_uncovered_history(memory_service):
    service, run, counter, h = memory_service
    counter.count_text = lambda text: 2500 if text in {"old q", "old a"} else len(text)
    h.model.fail_on_call = 1
    with service.execution(run.run_id) as execution:
        result = service.prepare_memory(run.run_id, execution)
    assert result.memory_prepared and result.summary_version == 0
    with service.session_factory() as session:
        assert (
            len(
                ConversationRepository(session).completed_turns(
                    run.conversation_id,
                    after_seq=0,
                    cutoff_seq=result.history_cutoff_seq,
                )
            )
            == 1
        )
        assert session.scalar(select(ConversationSummaryRecord)) is None


def test_later_batch_failure_publishes_nothing(memory_service):
    service, run, counter, h = memory_service
    with service.session_factory.begin() as session:
        old_user = session.scalar(
            select(ConversationMessage).where(
                ConversationMessage.role == "user",
                ConversationMessage.run_id != run.run_id,
            )
        )
        old_user.text = "x" * 7000
    # Summary budget allows exactly one complete old turn per batch.
    h.settings.input_max_tokens = 9000
    h.model.fail_on_call = 2
    original = service.summary_service.create_candidate
    service.summary_service.create_candidate = lambda previous, turns: original(
        previous, [*turns, make_turn(seq=3, question="y" * 7000)]
    )
    counter.total = 13000
    with (
        service.execution(run.run_id) as execution,
        pytest.raises(ConversationError, match="context_budget_exceeded"),
    ):
        service.prepare_memory(run.run_id, execution)
    assert h.model.calls == 2
    with service.session_factory() as session:
        assert session.scalar(select(ConversationSummaryRecord)) is None
        assert session.get(ConversationRun, run.run_id).memory_prepared is False


def test_db_failure_cannot_continue_without_pinned_version(memory_service, monkeypatch):
    service, run, _, _ = memory_service

    def fail(*args, **kwargs):
        raise RuntimeError("publish unavailable")

    monkeypatch.setattr(ConversationRepository, "publish_memory", fail)
    with (
        service.execution(run.run_id) as execution,
        pytest.raises(ConversationError, match="conversation_store_failed"),
    ):
        service.prepare_memory(run.run_id, execution)


def test_summary_and_run_reference_commit_together(memory_service, monkeypatch):
    service, run, counter, h = memory_service
    counter.count_text = lambda text: 2500 if text in {"old q", "old a"} else len(text)
    with service.execution(run.run_id) as execution:
        first = service.prepare_memory(run.run_id, execution)
        second = service.prepare_memory(run.run_id, execution)
    assert first.summary_version == second.summary_version == 1
    assert h.model.calls == 1
    with service.session_factory() as session:
        row = session.get(ConversationSummaryRecord, (run.conversation_id, 1))
        assert row.covered_until_seq == 2


def test_existing_summary_is_pinned_without_republishing(memory_service):
    service, run, _, h = memory_service
    with service.session_factory.begin() as session:
        prior = session.scalar(
            select(ConversationRun).where(ConversationRun.run_id != run.run_id)
        )
        session.add(
            ConversationSummaryRecord(
                conversation_id=run.conversation_id,
                version=1,
                content=ConversationSummary(user_goal="MH001").model_dump(),
                covered_until_seq=2,
                model_name="fake",
                source_run_id=prior.run_id,
            )
        )
    with service.execution(run.run_id) as execution:
        result = service.prepare_memory(run.run_id, execution)
    assert result.summary_version == 1
    assert h.model.calls == 0


def test_publish_failure_rolls_back_summary_and_reference(memory_service, monkeypatch):
    service, run, counter, _ = memory_service
    counter.count_text = lambda text: 2500 if text in {"old q", "old a"} else len(text)
    original = ConversationRepository.publish_memory

    def fail_after_flush(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("before commit failure")

    monkeypatch.setattr(ConversationRepository, "publish_memory", fail_after_flush)
    with (
        service.execution(run.run_id) as execution,
        pytest.raises(ConversationError, match="conversation_store_failed"),
    ):
        service.prepare_memory(run.run_id, execution)
    with service.session_factory() as session:
        assert session.scalar(select(ConversationSummaryRecord)) is None
        assert not session.get(ConversationRun, run.run_id).memory_prepared
