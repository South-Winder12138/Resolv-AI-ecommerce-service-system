import pytest

from resolv.models.refund_flow import ConversationRefundRun
from resolv.schemas.branch_execution import RoutingInput
from resolv.schemas.conversation import ConversationError
from tests.knowledge_branch_fixtures import knowledge_factory
from tests.test_knowledge_routing_graph import harness as knowledge_harness
from tests.test_refund_routing_graph import events, harness

__all__ = ["knowledge_factory"]


@pytest.mark.parametrize(
    "point",
    [
        "refund_after_query_saved",
        "refund_after_retrieval_saved",
        "refund_after_model_request_saved",
        "refund_after_observation_saved",
        "refund_after_finalize_commit",
        "after_final_result",
    ],
)
def test_v4_saved_stages_and_end_recover_without_duplicate_offer(tmp_path, point):
    with harness(tmp_path) as h:
        fired = False

        def fail(actual):
            nonlocal fired
            if actual == point and not fired:
                fired = True
                raise OSError("injected stage boundary")

        h.service.fault_hook = fail
        first = events(h)
        assert fired, point
        h.service.fault_hook = None
        final = events(h, resume=True) if first[-1].type == "error" else first
        assert final[-1].type == "done", [(e.type, e.data) for e in final]
        assert h.counts["prepare"] == h.counts["retrieve"] == 1
        assert h.counts["decide"] == 2


def test_reassessment_link_does_not_inherit_authority(tmp_path):
    with harness(tmp_path) as h:
        assert events(h)[-1].type == "done"
        offer_id = h.service.get_run(h.run.run_id).result_json["refund"]["offer_id"]
        new, created = h.service.accept(
            h.run.conversation_id,
            "changed",
            "订单A001申请退款，因为新的原因",
            "agent",
            routing_input=RoutingInput(reassesses_refund_offer_id=offer_id),
        )
        assert (
            created
            and new.run_id != h.run.run_id
            and new.understanding_id != h.run.understanding_id
        )
        assert new.routing_input["reassesses_refund_offer_id"] == offer_id
        assert new.result_json is None
        with h.factory() as s:
            assert s.get(ConversationRefundRun, new.run_id) is None
        with pytest.raises(ValueError):
            RoutingInput(
                reassesses_refund_offer_id=offer_id, continue_task_id="pending"
            )
        other = h.service.create()
        with pytest.raises(ConversationError, match="refund_not_found"):
            h.service.accept(
                other.conversation_id,
                "cross",
                "退款",
                "agent",
                routing_input=RoutingInput(reassesses_refund_offer_id=offer_id),
            )


@pytest.mark.parametrize("version", ["routed-v2", "routed-v3"])
def test_old_request_identity_and_original_graph_recovery(
    knowledge_factory, tmp_path, version
):
    if version == "routed-v3":
        h = knowledge_harness(knowledge_factory, tmp_path)
        c, old = h.conversation, h.run
    else:
        from tests.test_conversation_routing import service_for

        c = service_for(knowledge_factory)
        conv = c.create()
        old, _ = c.accept(conv.conversation_id, "same", "你好", "agent")
    c.orchestration_version = "routed-v4"
    saved, created = c.accept(
        old.conversation_id, old.request_id, old.question, old.mode
    )
    assert (
        not created
        and saved.run_id == old.run_id
        and saved.orchestration_version == version
    )
    final = list(c.run(old.run_id))
    assert final[-1].type == "done", [(e.type, e.data) for e in final]
    assert c.get_run(old.run_id).result_json["schema_version"] == (
        2 if version == "routed-v3" else 1
    )


def test_v4_knowledge_keeps_all_old_gates_and_projects_v3(knowledge_factory, tmp_path):
    from resolv.models.conversation import ConversationRun

    h = knowledge_harness(knowledge_factory, tmp_path)
    with knowledge_factory.begin() as s:
        row = s.get(ConversationRun, h.run.run_id)
        row.orchestration_version, row.state_schema_version = "routed-v4", 4
    final = list(h.conversation.run(h.run.run_id))
    assert final[-1].type == "done", [(e.type, e.data) for e in final]
    saved = h.conversation.get_run(h.run.run_id)
    assert saved.result_json["schema_version"] == 3
    assert h.counts["knowledge_check"] == h.counts["knowledge_answer"] == 1
    assert h.conversation.reply_metadata(saved.run_id) == saved.result_json


def test_legacy_request_and_graph_recover_under_v4_acceptor(knowledge_factory):
    from resolv.core.config import ContextSettings
    from resolv.services.context import ContextBuilder
    from resolv.services.conversation import ConversationService
    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.test_context import Counter
    from tests.test_conversation_graph import Model
    from tests.test_conversation_routing import service_for

    checkpoints = MemoryCheckpoints()
    model = Model()
    builder = ContextBuilder(
        ContextSettings(
            _env_file=None,
            tokenizer_profile="cl100k_base",
            model_context_window_tokens=16000,
        ),
        Counter(),
    )
    old = ConversationService(
        knowledge_factory, checkpoints, model=model, context_builder=builder
    )
    conv = old.create()
    run, _ = old.accept(conv.conversation_id, "legacy", "你好", "agent")
    understanding = service_for(knowledge_factory).understanding
    new = ConversationService(
        knowledge_factory,
        checkpoints,
        model=model,
        context_builder=builder,
        understanding=understanding,
        orchestration_version="routed-v4",
    )
    saved, created = new.accept(
        conv.conversation_id, "legacy", "你好", "agent", routing_input=RoutingInput()
    )
    assert not created and saved.orchestration_version == "legacy-v1"
    final = list(new.run(run.run_id))
    assert final[-1].type == "done", [(e.type, e.data) for e in final]
    assert new.get_run(run.run_id).state_schema_version == 1


def test_v4_reply_and_offer_stay_private_until_durable_end(tmp_path, monkeypatch):
    with harness(tmp_path) as h:
        saver = h.service.checkpoint_factory.saver
        original, fired = saver.put, False

        def put(config, checkpoint, metadata, versions):
            nonlocal fired
            if (
                not fired
                and config["configurable"]["thread_id"] == h.run.conversation_id
                and checkpoint["channel_values"].get("phase") == "end"
            ):
                fired = True
                raise OSError("lost END")
            return original(config, checkpoint, metadata, versions)

        monkeypatch.setattr(saver, "put", put)
        first = events(h)
        assert (
            fired
            and first[-1].type == "error"
            and not any(e.type == "message" for e in first)
        )
        detail = h.service.detail(h.run.conversation_id)
        assert not any(m.role == "assistant" for m in detail.messages)
        assert (
            next(r for r in detail.runs if r.run_id == h.run.run_id).result_json is None
        )
        final = events(h, resume=True)
        assert final[-1].type == "done", [(e.type, e.data) for e in final]
        assert h.counts["decide"] == 2


def test_refund_profile_limit_recovery_before_routing(tmp_path):
    from resolv.models.conversation import ConversationRun

    with harness(tmp_path) as h:
        h.client.mode = "loop"
        profile = h.profile.model_copy(update={"max_agent_steps": 3})
        with h.factory.begin() as s:
            s.get(ConversationRun, h.run.run_id).refund_config = profile.model_dump(
                mode="json"
            )

        def fail(point):
            if point == "after_understanding_terminal":
                raise OSError("before routing checkpoint")

        h.service.fault_hook = fail
        assert events(h)[-1].type == "error"
        h.service.fault_hook = None
        final = events(h, resume=True)
        assert final[-1].type == "done", [(e.type, e.data) for e in final]
        assert h.counts["decide"] == 3 and h.counts["query_refund_order"] == 2


def test_refund_profile_limit_recovers_at_bind_checkpoint(tmp_path):
    from resolv.models.conversation import ConversationRun

    with harness(tmp_path) as h:
        h.client.mode = "loop"
        with h.factory.begin() as s:
            s.get(ConversationRun, h.run.run_id).refund_config = h.profile.model_copy(
                update={"max_agent_steps": 3}
            ).model_dump(mode="json")
        with h.service.execution(h.run.run_id) as execution:
            run = h.service.get_run(h.run.run_id)
            graph = h.service._graph(execution)
            graph.invoke(
                h.service.initial_state(run, execution),
                {"configurable": {"thread_id": run.conversation_id}},
                interrupt_before=["refund_bind"],
                durability="sync",
            )
        final = events(h, resume=True)
        assert final[-1].type == "done", [(e.type, e.data) for e in final]
        assert h.counts["decide"] == 3


def test_recovery_rejects_changed_refund_provider_before_model_call(tmp_path):
    with harness(tmp_path) as h:
        changed = h.profile.query_model.model_copy(
            update={"model_name": "different-provider-model"}
        )
        h.service.refund_profile = h.profile.model_copy(update={"query_model": changed})
        result = events(h)
        assert (
            result[-1].type == "error"
            and result[-1].data["code"] == "run_state_incompatible"
        )
        assert h.counts["prepare"] == h.counts["decide"] == 0
