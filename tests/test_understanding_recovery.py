import pytest
from langchain_core.messages import AIMessage

from resolv.models.conversation import Conversation
from resolv.models.understanding import ConversationUnderstanding
from resolv.schemas.understanding import UnderstandingError
from tests.test_understanding_service import accept, require
from tests.understanding_fixtures import fail_once, make_classification, make_resolution


@pytest.mark.parametrize(
    "point,expected",
    [
        ("after_resolution_saved", (1, 1)),
        ("after_intents_saved", (1, 1)),
        ("after_publish", (1, 1)),
        ("after_graph_end", (1, 1)),
    ],
)
def test_saved_stages_are_reused_after_interruption(
    understanding_service, point, expected
):
    s = require(understanding_service)
    row = accept(s)
    s.fault_hook = fail_once(point)
    with pytest.raises(UnderstandingError):
        s.run("c", row.understanding_id)
    result = s.resume("c", row.understanding_id)
    assert result.status == "ready"
    assert (
        s.resolver.model.calls_by_stage["resolution"],
        s.classifier.model.calls_by_stage["intents"],
    ) == expected


def test_finalizing_is_not_exposed_as_ready_and_preserves_ids(understanding_service):
    s = require(understanding_service)
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(resolved_text="查订单，退款").model_dump_json()
        ),
        AIMessage(
            content=make_classification(
                ["查订单", "退款"], ["order", "refund_return"]
            ).model_dump_json()
        ),
    ]
    row = accept(s, text="查订单，退款")
    s.fault_hook = fail_once("after_publish")
    with pytest.raises(UnderstandingError):
        s.run("c", row.understanding_id)
    initial = s.get("c", row.understanding_id)
    assert initial.status == "finalizing" and not initial.route_ready
    assert s.list_tasks("c", "pending") == []
    with s.session_factory() as session:
        saved_ids = [
            p["task_id"]
            for p in session.get(
                ConversationUnderstanding, row.understanding_id
            ).final_json["pending_requests"]
        ]
    view = s.resume("c", row.understanding_id)
    assert [p.task_id for p in view.pending_requests] == saved_ids
    assert s.get("c", row.understanding_id).model_dump() == view.model_dump()


def test_publish_then_archive_cannot_finish_or_expose_pending(understanding_service):
    s = require(understanding_service)
    row = accept(s)

    def archive(point):
        if point == "after_publish":
            with s.session_factory.begin() as session:
                session.get(Conversation, "c").status = "archived"

    s.fault_hook = archive
    with pytest.raises(UnderstandingError, match="conversation_archived"):
        s.run("c", row.understanding_id)
    assert s.get("c", row.understanding_id).status == "failed"
    assert s.list_tasks("c", "pending") == []


def test_resume_rejects_changed_hash_without_overwriting_it(understanding_service):
    s = require(understanding_service)
    row = accept(s)
    with s.session_factory.begin() as session:
        db = session.get(ConversationUnderstanding, row.understanding_id)
        db.snapshot = db.snapshot | {"original_text": "tampered"}
    with pytest.raises(UnderstandingError, match="understanding_incompatible"):
        s.resume("c", row.understanding_id)
    assert s.get("c", row.understanding_id).original_text == "查订单"
    assert not s.resolver.model.calls_by_stage


def test_resume_uses_frozen_budget_not_current_startup_value(understanding_service):
    s = require(understanding_service)
    row = accept(s)
    s.fault_hook = fail_once("after_resolution_saved")
    with pytest.raises(UnderstandingError):
        s.run("c", row.understanding_id)
    s.context.config.app_max_input_tokens = 1
    assert s.resume("c", row.understanding_id).status == "ready"


def test_resume_uses_frozen_thinking_mode_after_startup_changes(understanding_service):
    s = require(understanding_service)
    s.context.config.enable_thinking = False
    row = accept(s)
    s.fault_hook = fail_once("after_resolution_saved")
    with pytest.raises(UnderstandingError):
        s.run("c", row.understanding_id)
    s.context.config.enable_thinking = True
    assert s.resume("c", row.understanding_id).status == "ready"
    assert s.classifier.model.client.bound[-1]["extra_body"]["enable_thinking"] is False


def test_older_snapshot_omitted_optional_flags_preserve_hash_on_restore(understanding_service):
    from copy import deepcopy

    from resolv.repositories.conversation import fingerprint

    s = require(understanding_service)
    row = accept(s)
    with s.session_factory.begin() as session:
        db = session.get(ConversationUnderstanding, row.understanding_id)
        old = deepcopy(db.snapshot)
        for field in ("model_api_mode", "enable_thinking", "thinking_budget", "output_format", "schema_constraint_version"):
            old["config"].pop(field)
        old["snapshot_hash"] = fingerprint({k:v for k,v in old.items() if k != "snapshot_hash"})
        db.snapshot = old
    result = s.resume("c", row.understanding_id)
    assert result.status == "ready"
    assert s.record("c", row.understanding_id).snapshot.snapshot_hash == old["snapshot_hash"]


def test_active_named_lock_prevents_second_executor(understanding_service):
    s = require(understanding_service)
    row = accept(s)
    with (
        s.checkpoint_factory(conversation_id="understanding:" + row.understanding_id),
        pytest.raises(UnderstandingError, match="understanding_busy"),
    ):
        s.resume("c", row.understanding_id)
    assert not s.resolver.model.calls_by_stage


@pytest.mark.parametrize("phase", ["resolve_references", "classify_intents", "end"])
def test_actual_checkpoint_put_failure_reuses_saved_resolution(
    understanding_service, monkeypatch, phase
):
    s = require(understanding_service)
    saver = s.checkpoint_factory.saver
    original = saver.put
    failed = False

    def put(config, checkpoint, metadata, versions):
        nonlocal failed
        if checkpoint["channel_values"].get("phase") == phase and not failed:
            failed = True
            raise RuntimeError("checkpoint write failed")
        return original(config, checkpoint, metadata, versions)

    monkeypatch.setattr(saver, "put", put)
    row = accept(s)
    with pytest.raises(UnderstandingError):
        s.run("c", row.understanding_id)
    assert s.resume("c", row.understanding_id).status == "ready"
    assert s.resolver.model.calls_by_stage["resolution"] == 1


def test_corrupt_saved_stage_state_is_incompatible(understanding_service):
    from resolv.understanding_graph import (
        UnderstandingNodes,
        build_understanding_graph,
    )

    s = require(understanding_service)
    row = accept(s)
    s.fault_hook = fail_once("after_intents_saved")
    with pytest.raises(UnderstandingError):
        s.run("c", row.understanding_id)
    graph = build_understanding_graph(
        UnderstandingNodes(s, None), s.checkpoint_factory.saver
    )
    config = {"configurable": {"thread_id": "understanding:" + row.understanding_id}}
    graph.update_state(config, {"resolution": {"output": {"resolved_text": "corrupt"}}})
    with pytest.raises(UnderstandingError, match="understanding_incompatible"):
        s.resume("c", row.understanding_id)


@pytest.mark.parametrize(
    "mutation",
    [
        {"state_schema_version": True},
        {
            "selection": {
                "primary_index": 0,
                "secondary_indexes": [],
                "branch": "escalate",
                "route_ready": True,
                "clarification_question": None,
                "continuation_task_id": None,
            },
            "result_status": "ready",
        },
    ],
)
def test_checkpoint_version_and_route_are_reconciled_with_domain(
    understanding_service, mutation
):
    from resolv.understanding_graph import (
        UnderstandingNodes,
        build_understanding_graph,
    )

    s = require(understanding_service)
    row = accept(s)
    s.fault_hook = fail_once("after_intents_saved")
    with pytest.raises(UnderstandingError):
        s.run("c", row.understanding_id)
    graph = build_understanding_graph(
        UnderstandingNodes(s, None), s.checkpoint_factory.saver
    )
    graph.update_state(
        {"configurable": {"thread_id": "understanding:" + row.understanding_id}},
        mutation,
    )
    with pytest.raises(UnderstandingError, match="understanding_incompatible"):
        s.resume("c", row.understanding_id)


def test_classification_expansion_exceeds_frozen_budget_before_call(
    understanding_service,
):
    s = require(understanding_service)
    s.resolver.model.client.replies = [
        AIMessage(content=make_resolution(resolved_text="x" * 35000).model_dump_json())
    ]
    # The output itself would normally hit its reserve first; independently exercise
    # the persisted-stage boundary using an admitted result from an earlier attempt.
    row = accept(s)
    with s.session_factory.begin() as session:
        from resolv.repositories.understanding import UnderstandingRepository
        from tests.understanding_fixtures import make_audit

        repo = UnderstandingRepository(session)
        owned = repo.claim(row.understanding_id, "old")
        repo.save_stage(
            row.understanding_id,
            owned.execution_epoch,
            "old",
            "resolution",
            make_resolution(resolved_text="x" * 35000),
            make_audit(),
        )
        repo.release(row.understanding_id, owned.execution_epoch, "old")
    with pytest.raises(UnderstandingError, match="context_budget_exceeded"):
        s.resume("c", row.understanding_id)
    assert s.classifier.model.calls_by_stage["intents"] == 0


def test_continue_id_conflicting_with_original_is_not_consumed(understanding_service):
    s = require(understanding_service)
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(resolved_text="查订单，退款").model_dump_json()
        ),
        AIMessage(
            content=make_classification(
                ["查订单", "退款"], ["order", "refund_return"]
            ).model_dump_json()
        ),
    ]
    parent = accept(s, text="查订单，退款")
    s.run("c", parent.understanding_id)
    task = s.list_tasks("c", "pending")[0]
    s.resolver.model.client.replies = [
        AIMessage(
            content=make_resolution(
                resolved_text="别继续退款，查订单"
            ).model_dump_json()
        ),
        AIMessage(
            content=make_classification(
                ["查订单"], ["order"], original_text="别继续退款，查订单"
            ).model_dump_json()
        ),
    ]
    child = accept(
        s, request_id="child", text="别继续退款，查订单", continue_task_id=task.task_id
    )
    with pytest.raises(UnderstandingError, match="continuation_conflict"):
        s.run("c", child.understanding_id)
    assert s.list_tasks("c", "pending")[0].task_id == task.task_id
