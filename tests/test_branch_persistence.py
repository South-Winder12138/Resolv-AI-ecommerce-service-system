import importlib

import pytest
from sqlalchemy import event, select

from resolv.models.conversation import ConversationMessage, ConversationRun
from resolv.repositories.conversation import ConversationRepository
from resolv.schemas.conversation import ConversationError
from resolv.services.branch_replies import render_reply
from resolv.services.conversation import ConversationService
from tests.branch_fixtures import branch_factory, make_view
from tests.conversation_fixtures import MemoryCheckpoints

__all__ = ["branch_factory"]


@pytest.fixture
def owned(branch_factory):
    service = ConversationService(branch_factory, MemoryCheckpoints())
    c = service.create()
    run, _ = service.accept(c.conversation_id, "request", "投诉", "agent")
    with service.execution(run.run_id) as execution:
        yield service, run, execution


def test_legacy_run_projection_has_no_new_execution_authority(owned):
    _, run, _ = owned
    assert run.orchestration_version == "legacy-v1"
    assert run.understanding_id is None and run.result_json is None
    assert run.routing_input == {} and run.model_audits == []
    assert run.business_config is None


def routed_result(service, run):
    meta = render_reply(
        run.run_id, make_view("投诉", "complaint"), mode="agent"
    ).metadata
    with service.session_factory.begin() as session:
        row = session.get(ConversationRun, run.run_id)
        assert hasattr(row, "orchestration_version"), "routed fields missing"
        row.orchestration_version = "routed-v2"
    return meta


def test_routed_answer_requires_metadata_and_compares_it_on_retry(owned):
    service, run, execution = owned
    meta = routed_result(service, run)
    with (
        pytest.raises(ConversationError, match="run_state_conflict"),
        service.session_factory.begin() as session,
    ):
        ConversationRepository(session).commit_answer(
            run.run_id, execution.epoch, "回复"
        )
    with service.session_factory.begin() as session:
        first = ConversationRepository(session).commit_answer(
            run.run_id, execution.epoch, "回复", metadata=meta
        )
    with service.session_factory.begin() as session:
        again = ConversationRepository(session).commit_answer(
            run.run_id, execution.epoch, "回复", metadata=meta
        )
        assert first == again
    different = meta.model_copy(update={"reply_kind": "step_limit"})
    with (
        pytest.raises(ConversationError, match="run_state_conflict"),
        service.session_factory.begin() as session,
    ):
        ConversationRepository(session).commit_answer(
            run.run_id, execution.epoch, "回复", metadata=different
        )
    assert service.get_run(run.run_id).result_json == meta.model_dump(mode="json")


def test_answer_and_metadata_failure_rolls_back_together(owned):
    service, run, execution = owned
    meta = routed_result(service, run)
    before = service.detail(run.conversation_id).conversation.next_message_seq

    def reject_metadata(mapper, connection, target):
        if target.result_json:
            raise RuntimeError("injected result persistence failure")

    event.listen(ConversationRun, "before_update", reject_metadata)
    try:
        with (
            pytest.raises(RuntimeError, match="injected"),
            service.session_factory.begin() as session,
        ):
            ConversationRepository(session).commit_answer(
                run.run_id, execution.epoch, "回复", metadata=meta
            )
    finally:
        event.remove(ConversationRun, "before_update", reject_metadata)
    with service.session_factory() as session:
        assert session.get(ConversationMessage, run.answer_message_id) is None
    assert service.get_run(run.run_id).result_json is None
    assert service.detail(run.conversation_id).conversation.next_message_seq == before


def test_audit_is_owned_and_idempotent(owned):
    from resolv.schemas.branch_execution import BusinessAudit

    service, run, execution = owned
    audit = BusinessAudit(
        invocation_id="inv-1",
        model_step=1,
        attempt=1,
        model_name="fake",
        profile_fingerprint="f" * 64,
        estimate=7000,
        actual_input=100,
        actual_output=20,
    )
    with (
        pytest.raises(ConversationError, match="conversation_busy"),
        service.session_factory.begin() as session,
    ):
        ConversationRepository(session).record_model_audit(
            run.run_id, execution.epoch - 1, audit
        )
    for _ in range(2):
        with service.session_factory.begin() as session:
            ConversationRepository(session).record_model_audit(
                run.run_id, execution.epoch, audit
            )
    assert service.get_run(run.run_id).model_audits == [audit.model_dump(mode="json")]


@pytest.mark.parametrize(
    "status,blocks",
    [
        ("running", True),
        ("waiting_approval", True),
        ("interrupted", True),
        ("finalizing", True),
        ("completed", False),
        ("failed", False),
    ],
)
def test_legacy_preflight_blocks_only_unfinished_owned_runs(owned, status, blocks):
    service, run, _ = owned
    try:
        check = importlib.import_module(
            "resolv.services.conversation_routing"
        ).assert_no_legacy_inflight
    except ModuleNotFoundError:
        pytest.fail("legacy preflight missing")
    with service.session_factory.begin() as session:
        session.get(ConversationRun, run.run_id).status = status
    with service.session_factory() as session:
        if blocks:
            with pytest.raises(
                ConversationError, match="legacy_run_requires_resolution"
            ):
                check(session, owner_id="local")
        else:
            check(session, owner_id="local")
        check(session, owner_id="another-owner")
        assert len(session.scalars(select(ConversationRun)).all()) == 1
