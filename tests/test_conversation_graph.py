import importlib
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from sqlalchemy import select

from resolv.core.config import ContextSettings
from resolv.models.conversation import ConversationAction
from resolv.repositories.conversation import ConversationRepository
from resolv.schemas.conversation import ConversationError
from resolv.services.context import ContextBuilder
from resolv.services.conversation import ConversationService
from tests.conversation_fixtures import MemoryCheckpoints, conversation_factory
from tests.stream_fakes import as_chunk
from tests.test_context import Counter, hit

__all__ = ["conversation_factory"]


class Model:
    def __init__(self, responses=None):
        self.responses = list(responses or [AIMessage(content="fake answer")])
        self.calls = []

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    def stream(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        yield response if isinstance(response, AIMessageChunk) else as_chunk(response)


class Knowledge:
    def __init__(self):
        self.calls = 0

    def retrieve(self, question):
        self.calls += 1
        return [hit()]


@dataclass
class Harness:
    service: object
    run: object
    module: object
    model: object
    knowledge: object

    def invoke(self, initial=True, command=None, **kwargs):
        with self.service.execution(self.run.run_id) as execution:
            nodes = self.module.ConversationNodes(self.service, execution)
            graph = self.module.build_conversation_graph(
                nodes, execution.checkpoint_session.saver
            )
            result = graph.invoke(
                command
                if command is not None
                else nodes.initial_state(self.run)
                if initial
                else None,
                {"configurable": {"thread_id": self.run.conversation_id}},
                durability="sync",
                **kwargs,
            )
            return result, graph.get_state(
                {"configurable": {"thread_id": self.run.conversation_id}}
            )


@pytest.fixture
def graph_harness(conversation_factory, tool_database):
    module = importlib.import_module("resolv.conversation_graph")
    model, knowledge = Model(), Knowledge()
    builder = ContextBuilder(
        ContextSettings(
            _env_file=None,
            model_context_window_tokens=16000,
            tokenizer_profile="cl100k_base",
        ),
        Counter(),
    )
    service = ConversationService(
        conversation_factory,
        MemoryCheckpoints(),
        context_builder=builder,
        model=model,
        knowledge=knowledge,
    )
    c = service.create()
    from resolv.services.tool_execution import WriteExecutionService

    service.write_service = WriteExecutionService(conversation_factory, tool_database)
    run, _ = service.accept(c.conversation_id, "req", "问题", "agent")
    return Harness(service, run, module, model, knowledge)


def test_plain_graph_final_answer(graph_harness):
    h = graph_harness
    with h.service.session_factory.begin() as session:
        from resolv.models.conversation import ConversationRun

        session.get(ConversationRun, h.run.run_id).mode = "chat"
    h.run = h.service.get_run(h.run.run_id)
    result, snapshot = h.invoke()
    assert result["final_answer"] == "fake answer" and snapshot.next == ()
    detail = h.service.detail(h.run.conversation_id, None, 50)
    assert [m.text for m in detail.messages] == ["问题", "fake answer"]
    assert detail.current_run.status == "finalizing"


def read_decision():
    return AIMessage(
        content="",
        tool_calls=[
            {"id": "tc1", "name": "query_order", "args": {"order_id": "MH001"}},
            {"id": "tc2", "name": "query_order", "args": {"order_id": "MH002"}},
        ],
    )


def test_read_node_per_call_and_saved_result_reused(graph_harness, monkeypatch):
    h = graph_harness
    h.model.responses = [read_decision(), AIMessage(content="done")]
    result, snapshot = h.invoke(interrupt_after=["read_tool"])
    assert result["tool_cursor"] == 1 and snapshot.next == ("read_tool",)
    assert len(h.model.calls) == 1
    result, snapshot = h.invoke(initial=False)
    assert [m.tool_call_id for m in result["messages"] if m.type == "tool"] == [
        "tc1",
        "tc2",
    ]
    assert snapshot.next == ()
    with h.service.session_factory() as session:
        rows = session.scalars(select(ConversationAction)).all()
        assert len(rows) == 2 and all(a.execution_status == "completed" for a in rows)


def test_each_model_call_rebuilds_context(graph_harness):
    h = graph_harness
    h.model.responses = [read_decision(), AIMessage(content="done")]
    h.invoke()
    assert len(h.model.calls) == 2
    assert sum(m.type == "tool" for m in h.model.calls[0]) == 0
    assert sum(m.type == "tool" for m in h.model.calls[1]) == 2


def test_partial_tool_json_never_executes(graph_harness):
    h = graph_harness
    h.model.responses = [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "query_order", "args": '{"order_id":', "id": "tc", "index": 0}
            ],
        )
    ]
    with pytest.raises(Exception, match="Incomplete streamed tool call"):
        h.invoke()
    with h.service.session_factory() as session:
        assert session.scalar(select(ConversationAction)) is None


def test_knowledge_snapshot_reused_after_resume(graph_harness):
    h = graph_harness
    _, snapshot = h.invoke(interrupt_after=["retrieve_knowledge"])
    assert snapshot.next == ("call_model",)
    h.invoke(initial=False)
    assert h.knowledge.calls == 1
    assert h.service.get_run(h.run.run_id).knowledge_snapshot == [hit().model_dump()]


def test_current_tool_result_over_budget_errors(graph_harness, monkeypatch):
    h = graph_harness
    h.model.responses = [read_decision(), AIMessage(content="must not call")]
    h.invoke(interrupt_after=["read_tool"])
    h.service.context_builder.counter.total = 12001
    with pytest.raises(ConversationError, match="context_budget_exceeded"):
        h.invoke(initial=False)
    assert len(h.model.calls) == 1


def test_new_run_resets_previous_working_state(graph_harness):
    h = graph_harness
    with h.service.execution(h.run.run_id) as execution:
        nodes = h.module.ConversationNodes(h.service, execution)
        initial = nodes.initial_state(h.run)
    assert len(initial["messages"]) == 1 and isinstance(
        initial["messages"][0], HumanMessage
    )
    assert (
        initial["tool_cursor"] == 0
        and initial["action_id"] is None
        and initial["remaining_iterations"] == 8
    )


def write_decision():
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "refund-call",
                "name": "apply_refund",
                "args": {"order_id": "MH001", "reason": "退款测试"},
            }
        ],
    )


def paused(h):
    h.model.responses = [write_decision(), AIMessage(content="fake write answer")]
    result, snapshot = h.invoke()
    with h.service.session_factory() as session:
        action = ConversationRepository(session).action(result["action_id"])
    return result, snapshot, action


def test_interrupt_before_write(graph_harness, tool_database):
    from resolv.models.tool_execution import ToolExecutionReceipt

    h = graph_harness
    _, snapshot, action = paused(h)
    assert action.approval_status == "pending" and snapshot.next == ("await_approval",)
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None


@pytest.mark.parametrize("approved", [True, False])
def test_rebuild_graph_uses_frozen_action(graph_harness, approved):
    from langgraph.types import Command

    h = graph_harness
    _, _, action = paused(h)
    h.service.decide(action.approval_id, h.run.conversation_id, approved)
    result, snapshot = h.invoke(
        initial=False, command=Command(resume={"action_id": action.action_id})
    )
    assert snapshot.next == () and len(h.model.calls) == 2
    with h.service.session_factory() as session:
        saved = ConversationRepository(session).action(action.action_id)
        assert saved.arguments == {"order_id": "MH001", "reason": "退款测试"}
        assert saved.execution_status == "completed"
        assert len(session.scalars(select(ConversationAction)).all()) == 1
    assert result["messages"][-2].tool_call_id == "refund-call"


def test_saved_decision_not_asked_again(graph_harness):
    from langgraph.types import Command

    h = graph_harness
    _, _, action = paused(h)
    h.service.decide(action.approval_id, h.run.conversation_id, True)
    _, snapshot = h.invoke(
        initial=False, command=Command(resume={"action_id": action.action_id})
    )
    assert snapshot.next == ()


def test_rejection_no_sqlite_write(graph_harness, tool_database):
    from langgraph.types import Command

    from resolv.models.tool_execution import ToolExecutionReceipt

    h = graph_harness
    _, _, action = paused(h)
    h.service.decide(action.approval_id, h.run.conversation_id, False)
    h.invoke(initial=False, command=Command(resume={"action_id": action.action_id}))
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None
    with h.service.session_factory() as session:
        assert ConversationRepository(session).action(action.action_id).result == {
            "executed": False,
            "reason": "user_rejected",
        }


def test_mixed_write_batch_rejected(graph_harness):
    h = graph_harness
    h.model.responses = [
        AIMessage(
            content="",
            tool_calls=[*write_decision().tool_calls, *read_decision().tool_calls],
        )
    ]
    with pytest.raises(ConversationError, match="unsupported_approval_batch"):
        h.invoke()


def test_no_remaining_round_before_approval(graph_harness):
    h = graph_harness
    h.service.max_iterations = 1
    with h.service.session_factory.begin() as session:
        from resolv.models.conversation import ConversationRun
        row=session.get(ConversationRun,h.run.run_id)
        row.count_config={**row.count_config,"max_iterations":1}
    h.run=h.service.get_run(h.run.run_id)
    h.model.responses = [write_decision()]
    with pytest.raises(ConversationError, match="max_iterations_exceeded"):
        h.invoke()
    with h.service.session_factory() as session:
        assert session.scalar(select(ConversationAction)) is None


def test_action_args_change_conflicts(graph_harness):
    from langgraph.types import Command

    h = graph_harness
    _, _, action = paused(h)
    h.service.decide(action.approval_id, h.run.conversation_id, True)
    with h.service.session_factory.begin() as session:
        session.get(ConversationAction, action.action_id).arguments = {
            "order_id": "MH002",
            "reason": "changed",
        }
    with pytest.raises(ConversationError, match="action_conflict"):
        h.invoke(initial=False, command=Command(resume={"action_id": action.action_id}))


def test_initial_checkpoint_window_uses_saved_iteration_limit(graph_harness):
    h=graph_harness
    h.service.max_iterations=99
    with h.service.execution(h.run.run_id) as execution:
        initial=h.module.ConversationNodes(h.service,execution).initial_state(h.run)
    assert initial["remaining_iterations"]==8
