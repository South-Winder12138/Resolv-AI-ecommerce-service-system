import importlib

import pytest
from langgraph.graph import END, START, StateGraph
from sqlalchemy import event, func, select

from resolv.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
)
from resolv.models.understanding import ConversationUnderstanding
from resolv.schemas.branch_execution import RoutingInput
from resolv.schemas.conversation import ConversationError
from tests.branch_fixtures import branch_factory, make_routed_service
from tests.conversation_fixtures import MemoryCheckpoints
from tests.understanding_fixtures import make_classification, make_resolution

__all__ = ["branch_factory"]


def service_for(factory, text="你好", intent="chitchat", checkpoints=None):
    try:
        return make_routed_service(
            factory,
            checkpoints or MemoryCheckpoints(),
            stage_outputs=[
                make_resolution(resolved_text=text),
                make_classification([text], [intent]),
            ],
        )
    except (TypeError, AttributeError) as exc:
        pytest.fail(f"routed service construction missing: {exc}")


def test_accept_retry_keeps_one_linked_understanding(branch_factory):
    service = service_for(branch_factory)
    c = service.create()
    first, created = service.accept(c.conversation_id, "r1", "你好", "agent")
    second, duplicate = service.accept(c.conversation_id, "r1", "你好", "agent")
    assert created is True and duplicate is False
    assert first.run_id == second.run_id
    assert first.understanding_id == second.understanding_id and first.understanding_id
    with branch_factory() as session:
        assert (
            session.scalar(select(func.count()).select_from(ConversationUnderstanding))
            == 1
        )
        assert (
            session.scalar(select(func.count()).select_from(ConversationMessage)) == 1
        )
        u = session.get(ConversationUnderstanding, first.understanding_id)
        assert u.request_id == "chat:" + first.run_id and u.original_text == "你好"
    with pytest.raises(ConversationError, match="request_conflict"):
        service.accept(
            c.conversation_id,
            "r1",
            "你好",
            "agent",
            routing_input=RoutingInput(continue_task_id="t"),
        )


@pytest.mark.parametrize("point", ["freeze", "understanding", "run", "message"])
def test_accept_rolls_back_all_records_without_inference(
    branch_factory, point, monkeypatch
):
    service = service_for(branch_factory)
    c = service.create()
    fake = service.understanding.resolver.model.client

    def fail(*args, **kwargs):
        raise RuntimeError("injected accept failure")

    if point == "freeze":
        monkeypatch.setattr(service.understanding.context, "freeze", fail)
        model = None
    else:
        model = {
            "understanding": ConversationUnderstanding,
            "run": ConversationRun,
            "message": ConversationMessage,
        }[point]
        event.listen(model, "before_insert", fail)
    try:
        with pytest.raises((RuntimeError, ConversationError)):
            service.accept(c.conversation_id, "r1", "你好", "agent")
    finally:
        if model:
            event.remove(model, "before_insert", fail)
    with branch_factory() as session:
        for table in (ConversationRun, ConversationMessage, ConversationUnderstanding):
            assert session.scalar(select(func.count()).select_from(table)) == 0
        assert session.get(Conversation, c.conversation_id).current_run_id is None
    assert fake.calls == []


def routing_api():
    module = importlib.import_module("resolv.services.conversation_routing")
    assert hasattr(module, "run_understanding"), "understanding coordinator missing"
    return module


def test_parent_checkpoint_configuration_cannot_reach_child(branch_factory):
    service = service_for(branch_factory)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "r1", "你好", "agent")
    checkpoints = service.checkpoint_factory
    results = []

    def parent_node(state):
        results.append(routing_api().run_understanding(run, service.understanding))
        return {"value": "parent-finished"}

    graph = StateGraph(dict)
    graph.add_node("understand", parent_node)
    graph.add_edge(START, "understand")
    graph.add_edge("understand", END)
    parent = graph.compile(checkpointer=checkpoints.saver)
    config = {"configurable": {"thread_id": c.conversation_id}}
    # A real persisted parent checkpoint and map are present when the child is called.
    parent.invoke({"value": "before"}, config, durability="sync")
    child_config = {
        "configurable": {
            "thread_id": "understanding:" + run.understanding_id,
            "checkpoint_ns": "",
        }
    }
    child = checkpoints.saver.get_tuple(child_config)
    assert child and child.checkpoint["channel_values"]["phase"] == "end"
    saved = parent.get_state(config)
    assert saved.values == {"value": "parent-finished"} and not saved.next
    assert results[0].status == "ready"
    fake = service.understanding.resolver.model.client
    assert len(fake.calls) == 2
    routing_api().run_understanding(run, service.understanding)
    assert len(fake.calls) == 2


def test_original_snapshot_is_reused_after_parent_failure(branch_factory):
    service = service_for(branch_factory)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "r1", "你好", "agent")
    original = service.understanding.record(c.conversation_id, run.understanding_id)
    first = routing_api().run_understanding(run, service.understanding)
    second = routing_api().run_understanding(run, service.understanding)
    assert first == second
    current = service.understanding.record(c.conversation_id, run.understanding_id)
    assert current.snapshot == original.snapshot
    assert len(service.understanding.resolver.model.client.calls) == 2
    primary = routing_api().load_primary(
        run, first, snapshot_hash=current.snapshot.snapshot_hash
    )
    assert primary.primary.text == "你好"
    with pytest.raises(ConversationError):
        routing_api().load_primary(
            run,
            first.model_copy(update={"conversation_id": "other"}),
            snapshot_hash=current.snapshot.snapshot_hash,
        )


def test_routed_service_requires_same_database_and_owner(branch_factory, tmp_path):
    from resolv.db.engine import create_db_engine, create_session_factory
    from resolv.services.conversation import ConversationService

    service = service_for(branch_factory)
    other = create_db_engine(f"sqlite:///{tmp_path / 'other.db'}")
    try:
        with pytest.raises(ValueError):
            ConversationService(
                create_session_factory(other),
                MemoryCheckpoints(),
                understanding=service.understanding,
                orchestration_version="routed-v2",
            )
        with pytest.raises(ValueError):
            ConversationService(
                branch_factory,
                MemoryCheckpoints(),
                owner_id="other",
                understanding=service.understanding,
                orchestration_version="routed-v2",
            )
        with pytest.raises(ValueError):
            ConversationService(
                branch_factory, MemoryCheckpoints(), orchestration_version="routed-v2"
            )
    finally:
        other.dispose()


def test_routed_accept_preserves_busy_archived_and_owner_boundaries(branch_factory):
    service = service_for(branch_factory)
    c = service.create()
    run, _ = service.accept(c.conversation_id, "r1", "你好", "agent")
    with branch_factory.begin() as session:
        session.get(ConversationRun, run.run_id).status = "waiting_approval"
    with pytest.raises(ConversationError, match="pending_approval_exists"):
        service.accept(c.conversation_id, "r2", "你好", "agent")
    other = service.create()
    service.archive(other.conversation_id)
    with pytest.raises(ConversationError, match="conversation_not_found"):
        service.accept(other.conversation_id, "r1", "你好", "agent")
    with pytest.raises(ConversationError):
        service.accept(
            service.create().conversation_id,
            "linked",
            "你好",
            "agent",
            routing_input=RoutingInput(clarifies_understanding_id=run.understanding_id),
        )
