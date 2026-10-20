"""Explicit resumable nodes. Business authority stays in the repositories."""

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from resolv.core.model_stream import stream_model_decision
from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.schemas.conversation import ConversationError
from resolv.services.context import ContextInput
from resolv.tool_calling import tool_result_to_message
from resolv.tools.executor import (
    ToolExecutionResult,
    execute_persisted_tool_call,
    execute_tool_call,
)
from resolv.tools.registry import TOOL_REGISTRY, ToolEffect, get_registered_tool


class ConversationState(TypedDict):
    conversation_id: str
    run_id: str
    mode: str
    state_schema_version: int
    model_step: int
    remaining_iterations: int
    messages: list[BaseMessage]
    tool_calls: list[dict]
    tool_cursor: int
    action_id: str | None
    summary_version: int
    history_cutoff_seq: int
    knowledge_snapshot: list[dict]
    final_answer: str | None
    phase: str


class ConversationNodes:
    def __init__(self, dependencies, execution):
        self.service, self.execution = dependencies, execution

    def initial_state(self, run):
        return ConversationState(
            conversation_id=run.conversation_id,
            run_id=run.run_id,
            mode=run.mode,
            state_schema_version=run.state_schema_version,
            model_step=0,
            remaining_iterations=run.count_config.get(
                "max_iterations", self.service.max_iterations
            ),
            messages=[HumanMessage(content=run.question)],
            tool_calls=[],
            tool_cursor=0,
            action_id=None,
            summary_version=0,
            history_cutoff_seq=0,
            knowledge_snapshot=[],
            final_answer=None,
            phase="prepare_memory",
        )

    def emit(self, type_, **payload):
        get_stream_writer()({"type": type_, "run_id": self.execution.run_id, **payload})

    def prepare_memory(self, state):
        run = self.service.prepare_memory(state["run_id"], self.execution)
        return {
            "summary_version": run.summary_version,
            "history_cutoff_seq": run.history_cutoff_seq,
            "phase": "retrieve_knowledge",
        }

    def retrieve_knowledge(self, state):
        self.execution.assert_owned()
        run = self.service.get_run(state["run_id"])
        if not run.knowledge_prepared:
            try:
                hits = (
                    self.service.knowledge.retrieve(run.question)
                    if self.service.knowledge
                    else []
                )
            except Exception:  # noqa: BLE001 — retrieval never falls through to generation
                raise ConversationError(
                    "knowledge_retrieval_failed", True, run.run_id
                ) from None
            selected = self.service.builder_for_run(run).select_knowledge(hits)
            self.execution.assert_owned()
            with self.service.session_factory.begin() as session:
                run = ConversationRepository(
                    session, self.service.owner_id
                ).save_knowledge(run.run_id, self.execution.epoch, selected)
        self.emit("knowledge_sources", sources=run.knowledge_snapshot)
        return {"knowledge_snapshot": run.knowledge_snapshot, "phase": "call_model"}

    def call_model(self, state):
        self.execution.assert_owned()
        if state["remaining_iterations"] < 1:
            raise ConversationError("max_iterations_exceeded", run_id=state["run_id"])
        with self.service.session_factory() as session:
            repo = ConversationRepository(session, self.service.owner_id)
            run = repo.run(state["run_id"])
            summary = repo.summary(run.conversation_id, state["summary_version"])
            history = repo.completed_turns(
                run.conversation_id,
                after_seq=summary.covered_until_seq if summary else 0,
                cutoff_seq=state["history_cutoff_seq"],
            )
        schemas = self.service.tool_schemas_for_run(run)
        context = self.service.builder_for_run(run).build(
            ContextInput(
                run.system_prompt_snapshot,
                summary,
                history,
                state["messages"],
                state["knowledge_snapshot"],
                schemas,
            )
        )
        model = (
            self.service.model.bind_tools(
                [entry.tool for entry in TOOL_REGISTRY.values()]
            )
            if run.mode == "agent"
            else self.service.model
        )
        model = model.bind(
            max_completion_tokens=self.service.builder_for_run(
                run
            ).settings.reserved_output_tokens
        )
        response, deltas = stream_model_decision(model, context.messages)
        self.execution.assert_owned()
        calls = response.tool_calls
        if calls:
            if run.mode == "chat":
                raise ConversationError("run_state_conflict", run_id=run.run_id)
            registered = [get_registered_tool(call["name"]) for call in calls]
            if any(tool.effect == ToolEffect.WRITE for tool in registered):
                if len(calls) != 1:
                    raise ConversationError(
                        "unsupported_approval_batch", run_id=run.run_id
                    )
                if state["remaining_iterations"] <= 1:
                    raise ConversationError(
                        "max_iterations_exceeded", run_id=run.run_id
                    )
                phase = "stage_action"
            else:
                phase = "read_tool"
        else:
            if not response.text.strip():
                raise ConversationError("model_response_empty", True, run.run_id)
            for delta in deltas:
                self.emit("text_delta", text=delta)
            phase = "finalize_answer"
        return {
            "messages": [*state["messages"], response],
            "model_step": state["model_step"] + 1,
            "remaining_iterations": state["remaining_iterations"] - 1,
            "tool_calls": calls,
            "tool_cursor": 0,
            "action_id": None,
            "final_answer": response.text if not calls else None,
            "phase": phase,
        }

    def read_tool(self, state):
        self.execution.assert_owned()
        call = state["tool_calls"][state["tool_cursor"]]
        with self.service.session_factory.begin() as session:
            action = ConversationRepository(
                session, self.service.owner_id
            ).stage_action(
                state["run_id"],
                self.execution.epoch,
                state["model_step"],
                call["id"],
                call["name"],
                "read",
                call["args"],
            )
        self.emit(
            "tool_call",
            action_id=action.action_id,
            tool_call_id=call["id"],
            tool=call["name"],
            arguments=action.arguments,
        )
        if action.execution_status == "completed":
            result = ToolExecutionResult(call["id"], call["name"], action.result)
        else:
            result = execute_tool_call(call)
            self.execution.assert_owned()
            with self.service.session_factory.begin() as session:
                ConversationRepository(
                    session, self.service.owner_id
                ).save_action_result(
                    action.action_id, self.execution.epoch, result.output
                )
        self.emit(
            "tool_result",
            action_id=action.action_id,
            tool_call_id=call["id"],
            tool=call["name"],
            result=result.output,
        )
        cursor = state["tool_cursor"] + 1
        return {
            "messages": [*state["messages"], tool_result_to_message(result)],
            "tool_cursor": cursor,
            "action_id": action.action_id,
            "phase": "read_tool" if cursor < len(state["tool_calls"]) else "call_model",
        }

    def finalize_answer(self, state):
        self.execution.assert_owned()
        self.service.commit_answer(
            state["run_id"], self.execution.epoch, state["final_answer"]
        )
        return {"phase": "end"}

    def stage_action(self, state):
        self.execution.assert_owned()
        call = state["tool_calls"][0]
        with self.service.session_factory.begin() as session:
            action = ConversationRepository(
                session, self.service.owner_id
            ).stage_action(
                state["run_id"],
                self.execution.epoch,
                state["model_step"],
                call["id"],
                call["name"],
                "write",
                call["args"],
            )
        return {"action_id": action.action_id, "phase": "await_approval"}

    def saved_action(self, state):
        self.execution.assert_owned()
        with self.service.session_factory() as session:
            action = ConversationRepository(session, self.service.owner_id).action(
                state["action_id"]
            )
        call = state["tool_calls"][0]
        registered = get_registered_tool(call["name"])
        arguments = registered.tool.args_schema.model_validate(
            call["args"]
        ).model_dump()
        if (
            action.run_id != state["run_id"]
            or action.model_step != state["model_step"]
            or action.tool_call_id != call["id"]
            or action.tool_name != call["name"]
            or action.effect != "write"
            or fingerprint(arguments) != action.arguments_fingerprint
            or fingerprint(action.arguments) != action.arguments_fingerprint
        ):
            raise ConversationError("action_conflict", run_id=state["run_id"])
        return action

    def await_approval(self, state):
        action = self.saved_action(state)
        if action.approval_status == "pending":
            reference = interrupt(
                {"action_id": action.action_id, "approval_id": action.approval_id}
            )
            if (
                not isinstance(reference, dict)
                or reference.get("action_id") != action.action_id
            ):
                raise ConversationError("action_conflict", run_id=state["run_id"])
            action = self.saved_action(state)
        if action.approval_status not in {"approved", "rejected"}:
            raise ConversationError("approval_required", run_id=state["run_id"])
        return {"phase": "execute_write"}

    def execute_write(self, state):
        action = self.saved_action(state)
        self.emit(
            "tool_call",
            action_id=action.action_id,
            tool_call_id=action.tool_call_id,
            tool=action.tool_name,
            arguments=action.arguments,
        )
        if action.approval_status == "rejected":
            result = ToolExecutionResult(
                action.tool_call_id,
                action.tool_name,
                {"executed": False, "reason": "user_rejected"},
            )
        else:
            result = execute_persisted_tool_call(
                action.action_id, self.execution, self.service.write_service
            )
        self.execution.assert_owned()
        with self.service.session_factory.begin() as session:
            ConversationRepository(session, self.service.owner_id).save_action_result(
                action.action_id, self.execution.epoch, result.output
            )
        self.emit(
            "tool_result",
            action_id=action.action_id,
            tool_call_id=action.tool_call_id,
            tool=action.tool_name,
            result=result.output,
        )
        return {
            "messages": [*state["messages"], tool_result_to_message(result)],
            "tool_cursor": 1,
            "phase": "call_model",
        }


def build_conversation_graph(nodes, saver):
    graph = StateGraph(ConversationState)
    for name in [
        "prepare_memory",
        "retrieve_knowledge",
        "call_model",
        "read_tool",
        "finalize_answer",
        "stage_action",
        "await_approval",
        "execute_write",
    ]:
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "prepare_memory")
    graph.add_edge("prepare_memory", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "call_model")
    graph.add_conditional_edges(
        "call_model",
        lambda state: state["phase"],
        {
            "read_tool": "read_tool",
            "finalize_answer": "finalize_answer",
            "stage_action": "stage_action",
        },
    )
    graph.add_conditional_edges(
        "read_tool",
        lambda state: state["phase"],
        {"read_tool": "read_tool", "call_model": "call_model"},
    )
    graph.add_edge("finalize_answer", END)
    graph.add_edge("stage_action", "await_approval")
    graph.add_edge("await_approval", "execute_write")
    graph.add_edge("execute_write", "call_model")
    return graph.compile(checkpointer=saver)
