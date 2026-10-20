import importlib

import pytest
from sqlalchemy import func, select

from resolv.models.conversation import Conversation, ConversationMessage
from resolv.schemas.understanding import UnderstandingError, UnderstandingInput
from resolv.understanding_rules import select_route
from tests.understanding_fixtures import (
    make_audit,
    make_classification,
    make_resolution,
    make_snapshot,
)


def repository(session, owner="local"):
    try:
        module = importlib.import_module("resolv.repositories.understanding")
    except ModuleNotFoundError:
        pytest.fail("understanding repository missing")
    return module.UnderstandingRepository(session, owner)


def create(repo, request_id="req", text="查订单", **links):
    request = UnderstandingInput(request_id=request_id, text=text, **links)
    return repo.create("c", request, lambda _: make_snapshot(text))


def prepared(factory, text="查订单，退款"):
    with factory.begin() as session:
        repo = repository(session)
        record, _ = create(repo, text=text)
        claimed = repo.claim(record.understanding_id, "executor")
        resolution = make_resolution(resolved_text=text)
        classified = make_classification(["查订单", "退款"], ["order", "refund_return"])
        repo.save_stage(
            record.understanding_id,
            claimed.execution_epoch,
            "executor",
            "resolution",
            resolution,
            make_audit(),
        )
        repo.save_stage(
            record.understanding_id,
            claimed.execution_epoch,
            "executor",
            "intents",
            classified,
            make_audit("intents"),
        )
        selected = select_route(record.snapshot, resolution, classified)
    return claimed, selected


def test_create_is_idempotent(understanding_factory):
    with understanding_factory.begin() as session:
        repo = repository(session)
        first, created = create(repo)
        again, duplicate = repo.create(
            "c",
            UnderstandingInput(request_id="req", text="查订单"),
            lambda _: pytest.fail("snapshot reread"),
        )
    assert created and not duplicate
    assert first.understanding_id == again.understanding_id


def test_freeze_failure_rolls_back_acceptance(understanding_factory):
    with pytest.raises(ValueError), understanding_factory.begin() as session:
        repository(session).create(
            "c",
            UnderstandingInput(request_id="req", text="查订单"),
            lambda _: (_ for _ in ()).throw(ValueError("failed freeze")),
        )
    with understanding_factory.begin() as session:
        _, created = create(repository(session))
    assert created


@pytest.mark.parametrize(
    "links,text",
    [
        ({}, "退款"),
        ({"continue_task_id": "t"}, "查订单"),
        ({"clarifies_understanding_id": "p"}, "查订单"),
    ],
)
def test_request_fingerprint_includes_text_and_links(
    understanding_factory, links, text
):
    with understanding_factory.begin() as session:
        create(repository(session))
    with (
        pytest.raises(UnderstandingError, match="request_conflict"),
        understanding_factory.begin() as session,
    ):
        create(repository(session), text=text, **links)


def test_context_and_input_do_not_change_chat(understanding_factory):
    with understanding_factory.begin() as session:
        record, _ = create(repository(session))
        c = session.get(Conversation, "c")
        assert c.current_run_id is None and c.next_message_seq == 1
        assert (
            session.scalar(select(func.count()).select_from(ConversationMessage)) == 0
        )
        assert record.original_text == record.snapshot.original_text == "查订单"


def test_message_cannot_borrow_completed_run_from_foreign_conversation(
    understanding_factory,
):
    from resolv.models.conversation import ConversationRun

    with understanding_factory.begin() as session:
        session.add(Conversation(conversation_id="foreign", thread_id="foreign"))
        session.flush()
        session.add(
            ConversationRun(
                run_id="foreign-run",
                conversation_id="foreign",
                request_id="r",
                request_fingerprint="f" * 64,
                question_message_id="q",
                answer_message_id="a",
                status="completed",
            )
        )
        session.flush()
        session.add_all(
            [
                ConversationMessage(
                    message_id="q",
                    conversation_id="c",
                    run_id="foreign-run",
                    seq=1,
                    role="user",
                    text="订单 MH999",
                ),
                ConversationMessage(
                    message_id="a",
                    conversation_id="c",
                    run_id="foreign-run",
                    seq=2,
                    role="assistant",
                    text="跨会话源",
                ),
            ]
        )
    with (
        pytest.raises(UnderstandingError, match="snapshot_invalid"),
        understanding_factory.begin() as session,
    ):
        create(repository(session))


@pytest.mark.parametrize("epoch,owner", [(0, "executor"), (1, "wrong")])
def test_old_owner_cannot_save_stage_or_error_or_release(
    understanding_factory, epoch, owner
):
    with understanding_factory.begin() as session:
        repo = repository(session)
        record, _ = create(repo)
        repo.claim(record.understanding_id, "executor")
    for operation in ("stage", "error", "release"):
        with (
            pytest.raises(UnderstandingError, match="understanding_busy"),
            understanding_factory.begin() as session,
        ):
            repo = repository(session)
            args = (record.understanding_id, epoch, owner)
            if operation == "stage":
                repo.save_stage(*args, "resolution", make_resolution(), make_audit())
            elif operation == "error":
                repo.record_error(*args, UnderstandingError("bad", True))
            else:
                repo.release(*args)


def test_stage_is_immutable_after_first_save(understanding_factory):
    with understanding_factory.begin() as session:
        repo = repository(session)
        record, _ = create(repo)
        owned = repo.claim(record.understanding_id, "executor")
        args = (
            record.understanding_id,
            owned.execution_epoch,
            "executor",
            "resolution",
        )
        repo.save_stage(*args, make_resolution(), make_audit())
        again = repo.save_stage(
            *args, make_resolution(resolved_text="changed"), make_audit()
        )
    assert again.resolution.output.resolved_text == "查订单"


def test_finalizing_hides_queue_and_publishing_preserves_ids(understanding_factory):
    owned, selected = prepared(understanding_factory)
    args = (owned.understanding_id, owned.execution_epoch, "executor")
    with understanding_factory.begin() as session:
        repo = repository(session)
        first = repo.publish(*args, selected)
        again = repo.publish(*args, selected)
        assert first.final_json == again.final_json and first.status == "finalizing"
        assert repo.list_tasks("c", "pending") == []
        assert first.final_json["primary_request"]["queue_status"] == "not_queued"
        repo.finish(*args, "ready")
        pending = repo.list_tasks("c", "pending")
        assert len(pending) == 1 and pending[0].intent == "refund_return"


def test_finish_archive_failure_never_exposes_pending(understanding_factory):
    owned, selected = prepared(understanding_factory)
    args = (owned.understanding_id, owned.execution_epoch, "executor")
    with understanding_factory.begin() as session:
        repo = repository(session)
        repo.publish(*args, selected)
        session.get(Conversation, "c").status = "archived"
    with (
        pytest.raises(UnderstandingError, match="conversation_archived"),
        understanding_factory.begin() as session,
    ):
        repository(session).finish(*args, "ready")
    with understanding_factory.begin() as session:
        repo = repository(session)
        repo.record_error(*args, UnderstandingError("conversation_archived"))
        assert repo.get("c", owned.understanding_id).status == "failed"
        assert repo.list_tasks("c", "pending") == []


def test_only_one_direct_clarification_child(understanding_factory):
    owned, selected = prepared(understanding_factory)
    args = (owned.understanding_id, owned.execution_epoch, "executor")
    with understanding_factory.begin() as session:
        repo = repository(session)
        selected.route_ready = False
        selected.clarification_question = "哪笔订单？"
        repo.publish(*args, selected)
        repo.finish(*args, "needs_clarification")
        child, _ = create(
            repo, "child", "MH001", clarifies_understanding_id=owned.understanding_id
        )
    with (
        pytest.raises(UnderstandingError) as caught,
        understanding_factory.begin() as session,
    ):
        create(
            repository(session),
            "another",
            "MH002",
            clarifies_understanding_id=owned.understanding_id,
        )
    assert caught.value.related_id == child.understanding_id


def test_cross_owner_records_are_not_found(understanding_factory):
    with understanding_factory.begin() as session:
        row, _ = create(repository(session))
    with (
        pytest.raises(UnderstandingError, match="conversation_not_found"),
        understanding_factory.begin() as session,
    ):
        repository(session, "stranger").get("c", row.understanding_id)


def test_two_accepted_continuations_have_one_effective_selection(understanding_factory):
    from resolv.schemas.understanding import EvidenceRef

    parent, selected = prepared(understanding_factory)
    with understanding_factory.begin() as session:
        repo = repository(session)
        repo.publish(
            parent.understanding_id, parent.execution_epoch, "executor", selected
        )
        repo.finish(
            parent.understanding_id, parent.execution_epoch, "executor", "ready"
        )
        pending = repo.list_tasks("c", "pending")
        task_id = pending[0].task_id
        children = []
        for request_id in ("first", "second"):
            request = UnderstandingInput(
                request_id=request_id, text="继续退款", continue_task_id=task_id
            )
            child, _ = repo.create(
                "c",
                request,
                lambda sources: make_snapshot(
                    "继续退款", pending_requests=sources.pending_requests
                ),
            )
            owned = repo.claim(child.understanding_id, request_id)
            resolution = make_resolution(continuation_selection=task_id)
            classified = make_classification(["继续退款"], ["refund_return"])
            classified.requests[0].evidence_refs.append(
                EvidenceRef(
                    kind="task",
                    source_id=task_id,
                    span=None,
                    json_path=["text"],
                    quote="退款",
                )
            )
            repo.save_stage(
                child.understanding_id,
                owned.execution_epoch,
                request_id,
                "resolution",
                resolution,
                make_audit(),
            )
            repo.save_stage(
                child.understanding_id,
                owned.execution_epoch,
                request_id,
                "intents",
                classified,
                make_audit("intents"),
            )
            children.append(
                (
                    owned,
                    select_route(child.snapshot, resolution, classified),
                    request_id,
                )
            )
    first, first_selection, first_owner = children[0]
    with understanding_factory.begin() as session:
        repository(session).publish(
            first.understanding_id, first.execution_epoch, first_owner, first_selection
        )
    second, second_selection, second_owner = children[1]
    with (
        pytest.raises(UnderstandingError, match="task_conflict") as caught,
        understanding_factory.begin() as session,
    ):
        repository(session).publish(
            second.understanding_id,
            second.execution_epoch,
            second_owner,
            second_selection,
        )
    assert caught.value.related_id == first.understanding_id


def test_saved_terminal_cannot_be_overwritten_by_error(understanding_factory):
    owned, selection = prepared(understanding_factory)
    args = (owned.understanding_id, owned.execution_epoch, "executor")
    with understanding_factory.begin() as session:
        repo = repository(session)
        repo.publish(*args, selection)
        repo.finish(*args, "ready")
        repo.record_error(*args, UnderstandingError("late", True))
        assert repo.get("c", owned.understanding_id).status == "ready"
