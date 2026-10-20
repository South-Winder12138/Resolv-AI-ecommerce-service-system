from threading import Event, Thread

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.dml import Update

from resolv.models.conversation import ConversationAction, ConversationMessage
from resolv.models.tool_execution import ToolExecutionReceipt
from resolv.repositories.conversation import ConversationRepository
from resolv.schemas.conversation import ConversationError
from tests.conversation_fixtures import conversation_factory
from tests.test_conversation_graph import graph_harness, write_decision

__all__ = ["conversation_factory", "graph_harness"]


@pytest.fixture
def recovery_harness(graph_harness):
    return graph_harness


def fail_checkpoint(h, phase, monkeypatch):
    saver = h.service.checkpoint_factory.saver
    original = saver.put
    failed = False

    def put(config, checkpoint, metadata, new_versions):
        nonlocal failed
        if not failed and checkpoint["channel_values"].get("phase") == phase:
            failed = True
            raise RuntimeError("injected checkpoint failure")
        return original(config, checkpoint, metadata, new_versions)

    monkeypatch.setattr(saver, "put", put)


def test_user_saved_initial_checkpoint_missing_reuses_question(recovery_harness):
    h = recovery_harness
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].type == "done"
    assert len(h.service.detail(h.run.conversation_id, None, 50).messages) == 2


@pytest.mark.parametrize("phase", ["retrieve_knowledge", "call_model"])
def test_prepared_memory_and_knowledge_reused_after_checkpoint_failure(
    recovery_harness, monkeypatch, phase
):
    h = recovery_harness
    fail_checkpoint(h, phase, monkeypatch)
    events = list(h.service.run(h.run.run_id))
    assert events[-1].type == "error" and not any(e.type == "done" for e in events)
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].type == "done" and h.knowledge.calls == 1


def test_model_complete_pending_writes_reused_after_checkpoint_failure(
    recovery_harness, monkeypatch
):
    h = recovery_harness
    h.model.responses = [write_decision(), AIMessage(content="final")]
    fail_checkpoint(h, "stage_action", monkeypatch)
    assert list(h.service.run(h.run.run_id))[-1].type == "error"
    with h.service.session_factory() as session:
        assert session.scalar(select(ConversationAction)) is None
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].type == "done" and len(h.model.calls) == 1


def test_model_partial_response_retried_without_partial_tool(recovery_harness):
    h = recovery_harness
    h.model.responses = [
        RuntimeError("partial response disconnected"),
        write_decision(),
        AIMessage(content="final"),
    ]
    assert list(h.service.run(h.run.run_id))[-1].type == "error"
    with h.service.session_factory() as session:
        assert session.scalar(select(ConversationAction)) is None
    assert (
        list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type == "done"
    )
    assert len(h.model.calls) == 2


def test_action_saved_pause_checkpoint_missing_is_reconciled(
    recovery_harness, monkeypatch
):
    h = recovery_harness
    h.model.responses = [write_decision(), AIMessage(content="final")]
    fail_checkpoint(h, "await_approval", monkeypatch)
    events = list(h.service.run(h.run.run_id))
    assert events[-1].type == "error" and not any(
        e.type == "approval_required" for e in events
    )
    with h.service.session_factory() as session:
        original = session.scalar(select(ConversationAction)).action_id
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].type == "done" and any(
        e.type == "approval_required" for e in events
    )
    assert (
        h.service.detail(h.run.conversation_id, None, 50).actions[0].action_id
        == original
    )
    assert len(h.model.calls) == 1


def test_receipt_committed_mysql_failed_returns_original(
    recovery_harness, monkeypatch, tool_database
):
    h = recovery_harness
    h.model.responses = [write_decision(), AIMessage(content="final")]
    assert list(h.service.run(h.run.run_id))[-1].type == "done"
    action = h.service.detail(h.run.conversation_id, None, 50).actions[0]
    h.service.decide(action.approval_id, h.run.conversation_id, True)
    original = ConversationRepository.save_action_result

    def fail(*args, **kwargs):
        raise RuntimeError("mysql write failed after SQLite commit")

    monkeypatch.setattr(ConversationRepository, "save_action_result", fail)
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].type == "error" and not any(e.type == "done" for e in events)
    with tool_database() as session:
        receipt = session.get(ToolExecutionReceipt, action.action_id)
        assert receipt.result["changed"] is True
        result = receipt.result
    monkeypatch.setattr(ConversationRepository, "save_action_result", original)
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].type == "done"
    assert h.service.detail(h.run.conversation_id, None, 50).actions[0].result == result
    with tool_database() as session:
        assert len(session.scalars(select(ToolExecutionReceipt)).all()) == 1


def test_sqlite_before_commit_failure_has_no_receipt_and_retries(
    recovery_harness, monkeypatch, tool_database
):
    from resolv.repositories.tool_execution import ToolExecutionRepository

    h = recovery_harness
    h.model.responses = [write_decision(), AIMessage(content="final")]
    list(h.service.run(h.run.run_id))
    action = h.service.detail(h.run.conversation_id, None, 50).actions[0]
    h.service.decide(action.approval_id, h.run.conversation_id, True)
    original = ToolExecutionRepository.add
    monkeypatch.setattr(
        ToolExecutionRepository,
        "add",
        lambda *args: (_ for _ in ()).throw(RuntimeError("before commit")),
    )
    assert (
        list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type == "error"
    )
    with tool_database() as session:
        assert session.get(ToolExecutionReceipt, action.action_id) is None
    monkeypatch.setattr(ToolExecutionRepository, "add", original)
    assert (
        list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type == "done"
    )


def test_answer_saved_checkpoint_failed_no_model_recall(recovery_harness, monkeypatch):
    h = recovery_harness
    fail_checkpoint(h, "end", monkeypatch)
    assert list(h.service.run(h.run.run_id))[-1].type == "error"
    assert h.service.get_run(h.run.run_id).status == "finalizing"
    assert (
        list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type == "done"
    )
    assert len(h.model.calls) == 1
    with h.service.session_factory() as session:
        assert len(session.scalars(select(ConversationMessage)).all()) == 2


def test_current_run_released_only_after_end(recovery_harness, monkeypatch):
    h = recovery_harness
    original = h.service.mark_completed

    def checked(run_id, epoch):
        detail = h.service.detail(h.run.conversation_id, None, 50)
        assert detail.current_run.status == "finalizing"
        snapshot = h.service.checkpoint_factory.saver.get_tuple(
            {"configurable": {"thread_id": h.run.conversation_id}}
        )
        assert snapshot.checkpoint["channel_values"]["phase"] == "end"
        return original(run_id, epoch)

    monkeypatch.setattr(h.service, "mark_completed", checked)
    assert list(h.service.run(h.run.run_id))[-1].type == "done"
    assert h.service.detail(h.run.conversation_id, None, 50).current_run is None


def test_completed_done_lost_returns_saved_answer(recovery_harness):
    h = recovery_harness
    list(h.service.run(h.run.run_id))
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].type == "done" and len(h.model.calls) == 1


def test_prior_terminal_checkpoint_allows_new_run(recovery_harness):
    h = recovery_harness
    list(h.service.run(h.run.run_id))
    newer, _ = h.service.accept(h.run.conversation_id, "new", "second question", "chat")
    h.model.responses = [AIMessage(content="second answer")]
    assert list(h.service.run(newer.run_id))[-1].type == "done"
    assert len(h.service.detail(h.run.conversation_id, None, 50).messages) == 4
    assert [m.content for m in h.model.calls[-1] if m.type == "human"][
        -1
    ] == "second question"


def test_incompatible_checkpoint_is_not_replaced(recovery_harness, monkeypatch):
    h = recovery_harness
    h.invoke(interrupt_after=["retrieve_knowledge"])
    saver = h.service.checkpoint_factory.saver
    original = saver.get_tuple

    def incompatible(config):
        saved = original(config)
        if saved:
            checkpoint = saved.checkpoint.copy()
            checkpoint["channel_values"] = checkpoint["channel_values"] | {
                "state_schema_version": 999
            }
            return saved._replace(checkpoint=checkpoint)
        return saved

    monkeypatch.setattr(saver, "get_tuple", incompatible)
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].data["code"] == "run_state_incompatible"
    assert not h.model.calls


def test_stream_disconnect_not_second_executor(recovery_harness):
    h = recovery_harness
    started, release = Event(), Event()
    original = h.model.stream

    def slow(messages):
        started.set()
        assert release.wait(3)
        yield from original(messages)

    h.model.stream = slow
    worker = Thread(target=lambda: list(h.service.run(h.run.run_id)))
    worker.start()
    assert started.wait(2)
    with (
        pytest.raises(ConversationError, match="conversation_busy"),
        h.service.execution(h.run.run_id),
    ):
        pytest.fail("second executor entered")
    release.set()
    worker.join(3)
    assert not worker.is_alive() and len(h.model.calls) == 1


def test_recovery_smoke_requires_owned_isolated_business_db(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("scripts.smoke_conversation_recovery")
    monkeypatch.setenv(
        "RECOVERY_TEST_URL",
        "mysql+pymysql://app:private_pw@host/resolv_conversations_smoke_test?charset=utf8mb4",
    )
    existing = tmp_path / "existing.db"
    existing.write_bytes(b"must not change")
    with pytest.raises(SystemExit) as raised:
        module.main(
            ["--database-env", "RECOVERY_TEST_URL", "--business-db", str(existing)]
        )
    assert "private_pw" not in str(raised.value)
    assert existing.read_bytes() == b"must not change"


def test_permanent_invalid_batch_releases_only_clean_run(recovery_harness):
    h = recovery_harness
    h.model.responses = [
        AIMessage(
            content="",
            tool_calls=[*write_decision().tool_calls, *write_decision().tool_calls],
        )
    ]
    events = list(h.service.run(h.run.run_id))
    assert events[-1].type == "error"
    assert h.service.get_run(h.run.run_id).status == "failed"
    assert h.service.detail(h.run.conversation_id, None, 50).current_run is None


def test_saved_tool_schema_mismatch_stops_without_replacing_checkpoint(
    recovery_harness,
):
    from resolv.models.conversation import ConversationRun

    h = recovery_harness
    h.invoke(interrupt_after=["retrieve_knowledge"])
    with h.service.session_factory.begin() as session:
        session.get(
            ConversationRun, h.run.run_id
        ).tool_schema_fingerprint = "wrong-schema"
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert (
        events[-1].type == "error"
        and events[-1].data["code"] == "run_state_incompatible"
    )
    assert not h.model.calls


def test_fixed_knowledge_conflict_stops_before_model(recovery_harness):
    from resolv.models.conversation import ConversationRun

    h = recovery_harness
    h.invoke(interrupt_after=["retrieve_knowledge"])
    with h.service.session_factory.begin() as session:
        session.get(ConversationRun, h.run.run_id).knowledge_snapshot = []
    events = list(h.service.resume(h.run.conversation_id, h.run.run_id))
    assert events[-1].data["code"] == "run_state_conflict"
    assert not h.model.calls


def test_resume_uses_pinned_budget_after_runtime_setting_change(recovery_harness):
    h = recovery_harness
    h.invoke(interrupt_after=["retrieve_knowledge"])
    h.service.context_builder.settings.app_max_input_tokens = 1
    assert (
        list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type == "done"
    )


def test_late_old_error_cannot_clobber_resumed_completion(recovery_harness):
    from resolv.models.conversation import ConversationRun

    h = recovery_harness
    with h.service.execution(h.run.run_id) as old:
        pass
    factory = h.service.session_factory
    injected = False

    class InterleavingSession(Session):
        def get(self, entity, ident, **kwargs):
            nonlocal injected
            row = super().get(entity, ident, **kwargs)
            if entity is ConversationRun and not injected:
                injected = True
                h.service.session_factory = factory
                assert (
                    list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type
                    == "done"
                )
            return row

    h.service.session_factory = sessionmaker(
        bind=factory.kw["bind"], class_=InterleavingSession
    )
    try:
        h.service._record_error(
            h.run.run_id, old, ConversationError("unsupported_approval_batch")
        )
    finally:
        h.service.session_factory = factory
    assert injected
    assert h.service.get_run(h.run.run_id).status == "completed"
    assert h.service.detail(h.run.conversation_id, None, 50).current_run is None
    assert len(h.service.detail(h.run.conversation_id, None, 50).messages) == 2


def test_lost_lock_cleanup_cannot_clear_new_executor(recovery_harness):
    from resolv.models.conversation import ConversationRun

    h = recovery_harness
    factory = h.service.session_factory
    injected = False

    def takeover():
        nonlocal injected
        injected = True
        with factory.begin() as session:
            row = session.get(ConversationRun, h.run.run_id)
            row.execution_epoch += 1
            row.execution_owner_id = "new-executor-after-connection-loss"
            row.status = "running"

    class InterleavingSession(Session):
        def get(self, entity, ident, **kwargs):
            row = super().get(entity, ident, **kwargs)
            if entity is ConversationRun and not injected:
                takeover()
            return row

        def execute(self, statement, *args, **kwargs):
            if (
                isinstance(statement, Update)
                and statement.table.name == "conversation_runs"
                and not injected
            ):
                takeover()
            return super().execute(statement, *args, **kwargs)

    with h.service.execution(h.run.run_id) as old:
        # Simulate the DB interleaving after raw lock loss. Memory locks do not
        # model socket lifetime; real KILL CONNECTION is tested by the gate.
        old._failed.set()
        h.service.session_factory = sessionmaker(
            bind=factory.kw["bind"], class_=InterleavingSession
        )
    h.service.session_factory = factory
    assert injected
    newer = h.service.get_run(h.run.run_id)
    assert newer.execution_epoch == old.epoch + 1
    assert newer.execution_owner_id == "new-executor-after-connection-loss"
    assert newer.status == "running"


def test_extra_first_model_argument_finishes_invalid_run_without_action(
    recovery_harness,
):
    h = recovery_harness
    h.model.responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "invalid-first",
                    "name": "query_order",
                    "args": {"order_id": "MH001", "extra": True},
                }
            ],
        )
    ]
    events = list(h.service.run(h.run.run_id))
    assert events[-1].type == "error" and events[-1].data["code"] == "invalid_request"
    assert h.service.get_run(h.run.run_id).status == "failed"
    assert h.service.detail(h.run.conversation_id, None, 50).current_run is None
    assert h.service.detail(h.run.conversation_id, None, 50).actions == []
    newer, _ = h.service.accept(
        h.run.conversation_id, "after-invalid", "a valid question", "chat"
    )
    assert newer.run_id != h.run.run_id


def test_resumed_model_uses_frozen_output_limit_without_shared_mutation(
    recovery_harness,
):
    h = recovery_harness
    invoked_limits = []
    h.model.max_tokens = 2000

    def bind(**kwargs):
        invoked_limits.append(kwargs["max_completion_tokens"])
        return h.model

    h.model.bind = bind
    h.invoke(interrupt_after=["retrieve_knowledge"])
    h.service.context_builder.settings.reserved_output_tokens = 9000
    h.model.max_tokens = 9000
    assert (
        list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type == "done"
    )
    assert invoked_limits == [2000]
    assert h.model.max_tokens == 9000


def test_resumed_output_limit_reaches_real_responses_api_payload(
    recovery_harness, monkeypatch
):
    import asyncio

    from langchain_core.messages import AIMessageChunk
    from langchain_openai import ChatOpenAI

    h = recovery_harness
    h.invoke(interrupt_after=["retrieve_knowledge"])
    model = ChatOpenAI(
        model="fake",
        api_key="offline-placeholder",
        base_url="https://fake.invalid/v1",
        max_tokens=9000,
        use_responses_api=True,
    )
    payload_limits = []

    def transport_stub(self, messages, config=None, **kwargs):
        payload_limits.append(
            self._get_request_payload(messages, **kwargs)["max_output_tokens"]
        )
        yield AIMessageChunk(content="offline complete answer")

    monkeypatch.setattr(ChatOpenAI, "stream", transport_stub)
    h.service.model = model
    try:
        assert (
            list(h.service.resume(h.run.conversation_id, h.run.run_id))[-1].type
            == "done"
        )
        assert payload_limits == [2000]
        assert model.max_tokens == 9000
    finally:
        model.root_client.close()
        asyncio.run(model.root_async_client.close())
