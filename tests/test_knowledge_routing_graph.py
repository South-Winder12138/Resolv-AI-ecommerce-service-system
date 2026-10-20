"""Real LangGraph with offline services: SQL decisions, no knowledge business tools."""

import json

import pytest
from sqlalchemy import func, select

from resolv.knowledge_routing_graph import KnowledgeRoutedNodes
from resolv.models.conversation import ConversationAction
from resolv.schemas.conversation import ConversationError
from tests.conversation_fixtures import MemoryCheckpoints
from tests.knowledge_branch_fixtures import (
    hybrid_result,
    knowledge_factory,
    make_knowledge_service,
)

__all__ = ["knowledge_factory"]


def harness(factory, tmp_path, **kwargs):
    h = make_knowledge_service(factory, MemoryCheckpoints(), **kwargs)
    c = h.conversation
    c.orchestration_version = "routed-v3"
    c.knowledge_profile, c.knowledge_clients, c.knowledge_retrievers = (
        h.profile,
        h.service.clients,
        h.retrievers,
    )
    c.knowledge_branch_service = h.service
    c.release_selector_path = tmp_path / "selector.json"
    s = h.profile.selection
    c.release_selector_path.write_text(
        json.dumps(
            {
                "enabled": True,
                **s.model_dump(
                    mode="json", exclude={"status", "profile", "error_code"}
                ),
            }
        )
    )
    return h


@pytest.mark.parametrize("mode", ["agent", "chat"])
def test_knowledge_runs_two_gates_then_once_without_business_actions(
    knowledge_factory, tmp_path, mode
):
    from resolv.models.conversation import ConversationRun

    h = harness(knowledge_factory, tmp_path)
    with knowledge_factory.begin() as session:
        session.get(ConversationRun, h.run.run_id).mode = mode
    events = list(h.conversation.run(h.run.run_id))
    assert events[-1].type == "done" and events[-1].data["status"] == "completed", [
        (e.type, e.data) for e in events
    ]
    saved = h.conversation.get_run(h.run.run_id)
    assert (
        saved.orchestration_version == "routed-v3" and saved.state_schema_version == 3
    )
    assert h.counts["knowledge_answer"] == 1 and h.counts["retrieval"] == 1
    assert saved.result_json["knowledge"]["used_citation_numbers"] == [1]
    with knowledge_factory() as session:
        assert session.scalar(select(func.count()).select_from(ConversationAction)) == 0


def test_parent_state_flag_cannot_authorize_low_confidence_answer(
    knowledge_factory, tmp_path
):
    h = harness(knowledge_factory, tmp_path, hybrid_result=hybrid_result((-10.0,)))
    with h.conversation.execution(h.run.run_id) as execution:
        nodes = KnowledgeRoutedNodes(h.conversation, execution)
        state = nodes.initial_state(h.conversation.get_run(h.run.run_id))
        state.update(nodes.route_by_intent(state))
        h.service.prepare_query(h.run.run_id, execution)
        h.service.retrieve(h.run.run_id, execution)
        h.service.check(h.run.run_id, execution)
        state.update(evidence_strong=True, phase="main_agent")
        from resolv.schemas.knowledge_branch import KnowledgeBranchError

        with pytest.raises(
            KnowledgeBranchError, match="knowledge_evidence_not_sufficient"
        ):
            nodes.main_agent(state)
        state["branch"] = "business"
        with pytest.raises(ConversationError):
            nodes.main_agent(state)
    assert h.counts["knowledge_answer"] == 0


def test_new_accept_freezes_selector_without_retriever_or_model_calls(
    knowledge_factory, tmp_path
):
    h = harness(knowledge_factory, tmp_path)
    c = h.conversation.create()
    before = dict(h.counts)
    run, created = h.conversation.accept(c.conversation_id, "new", "退货", "agent")
    assert created and run.orchestration_version == "routed-v3"
    assert run.knowledge_config["selection"]["release_id"] == "fixture-release"
    assert dict(h.counts) == before


def test_disabled_selector_only_blocks_knowledge(knowledge_factory, tmp_path):
    from tests.understanding_fixtures import make_classification, make_resolution

    h = harness(knowledge_factory, tmp_path)
    h.conversation.release_selector_path.write_text(
        '{"enabled":false,"config_version":1}'
    )
    text = "你好"
    h.conversation.understanding.resolver.model.client.replies.extend(
        [
            __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(
                content=o.model_dump_json()
            )
            for o in (
                make_resolution(resolved_text=text),
                make_classification([text], ["chitchat"]),
            )
        ]
    )
    c = h.conversation.create()
    run, _ = h.conversation.accept(c.conversation_id, "hello", text, "agent")
    events = list(h.conversation.run(run.run_id))
    assert events[-1].type == "done", [(e.type, e.data) for e in events]
    assert h.conversation.get_run(run.run_id).result_json["knowledge"] is None


def test_v3_business_approval_still_executes_existing_policy(knowledge_factory, tool_database):
    from resolv.models.conversation import ConversationRun
    from tests.knowledge_branch_fixtures import branch_profile
    from tests.test_routing_graph import decision, tool
    from tests.test_routing_graph import harness as business_harness
    service, run, _ = business_harness(knowledge_factory, tool_database, "取消订单 MH001", "order", [
        decision(tool(id="read")), decision(tool("cancel_order", id="write")), decision(text="订单已发货，无法取消，未修改订单。")])
    with knowledge_factory.begin() as session:
        row = session.get(ConversationRun, run.run_id)
        row.orchestration_version, row.state_schema_version = "routed-v3", 3
        row.knowledge_config = branch_profile().model_dump(mode="json")
    events = list(service.run(run.run_id))
    approval = next(e.data for e in events if e.type == "approval_required")
    service.decide(approval["approval_id"], run.conversation_id, True, require_pause=True)
    final = list(service.resume(run.conversation_id, run.run_id))
    assert final[-1].type == "done", [(e.type,e.data) for e in final]
    assert service.get_run(run.run_id).result_json["schema_version"] == 2
    assert service.get_run(run.run_id).result_json["knowledge"] is None


@pytest.mark.parametrize("selector", [None, "{broken"])
def test_missing_or_bad_selector_fails_only_knowledge_without_external_calls(
    knowledge_factory, tmp_path, selector
):
    from langchain_core.messages import AIMessage

    from tests.understanding_fixtures import make_classification, make_resolution

    h = harness(knowledge_factory, tmp_path)
    if selector is None:
        h.conversation.release_selector_path.unlink()
    else:
        h.conversation.release_selector_path.write_text(selector)
    text = "退货运费由谁承担？"
    h.conversation.understanding.resolver.model.client.replies.extend(
        AIMessage(content=o.model_dump_json()) for o in (
            make_resolution(resolved_text=text),
            make_classification([text], ["product_consultation"]),
        )
    )
    c = h.conversation.create()
    before = dict(h.counts)
    run, _ = h.conversation.accept(c.conversation_id, "new", text, "agent")
    events = list(h.conversation.run(run.run_id))
    assert events[-1].type == "error", [(e.type, e.data) for e in events]
    assert h.conversation.get_run(run.run_id).status == "failed"
    assert dict(h.counts) == before


def test_only_primary_is_prepared_and_secondary_remains_pending(knowledge_factory, tmp_path):
    from langchain_core.messages import AIMessage

    from tests.understanding_fixtures import make_classification, make_resolution

    h = harness(knowledge_factory, tmp_path)
    primary, secondary = "退货运费由谁承担？", "我要退订单MH001"
    text = primary + "，" + secondary
    h.conversation.understanding.resolver.model.client.replies.extend(
        AIMessage(content=o.model_dump_json()) for o in (
            make_resolution(resolved_text=text),
            make_classification([primary, secondary], ["product_consultation", "refund_return"]),
        )
    )
    client = h.service.clients["knowledge_query"]
    invoke, inputs = client.invoke_json, []
    def capture(purpose, messages, *args, **kwargs):
        if purpose == "knowledge_query":
            inputs.append(messages[-1].content)
        return invoke(purpose, messages, *args, **kwargs)
    client.invoke_json = capture
    c = h.conversation.create()
    run, _ = h.conversation.accept(c.conversation_id, "multi", text, "agent")
    events = list(h.conversation.run(run.run_id))
    assert events[-1].type == "done", [(e.type, e.data) for e in events]
    assert len(inputs) == 1
    assert json.loads(inputs[0])["primary_text"] == primary
    assert secondary not in inputs[0] and "MH001" not in inputs[0]
    saved = h.conversation.get_run(run.run_id)
    view = h.conversation.understanding.view(h.conversation.understanding.record(c.conversation_id, saved.understanding_id))
    assert [(p.text, p.queue_status) for p in view.pending_requests] == [(secondary, "pending")]


def test_v3_order_refund_keeps_unavailable_route(knowledge_factory, tool_database):
    from resolv.models.conversation import ConversationRun
    from tests.knowledge_branch_fixtures import branch_profile
    from tests.test_routing_graph import harness as business_harness
    service, run, model = business_harness(knowledge_factory, tool_database, "退订单MH001", "refund_return")
    with knowledge_factory.begin() as session:
        row = session.get(ConversationRun, run.run_id)
        row.orchestration_version, row.state_schema_version = "routed-v3", 3
        row.knowledge_config = branch_profile().model_dump(mode="json")
    events = list(service.run(run.run_id))
    assert events[-1].type == "done", [(e.type,e.data) for e in events]
    reply = service.get_run(run.run_id).result_json
    assert reply["reply_kind"] == "unavailable" and reply["knowledge"] is None
    assert model.calls == []
