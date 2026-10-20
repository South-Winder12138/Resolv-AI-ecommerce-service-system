"""An explicit, bounded Agent loop with one-call write approval suspension."""

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from langchain_core.messages import BaseMessage

from resolv.agent_events import (
    AgentApprovalRequiredEvent,
    AgentDoneEvent,
    AgentEvent,
    AgentMessageEvent,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from resolv.approval import (
    ApprovalRequest,
    PendingToolCall,
    SuspendedAgentState,
    consume_suspended_state,
    suspend_agent,
)
from resolv.core.llm import get_chat_model
from resolv.core.model_stream import (
    ToolCallParseError,  # noqa: F401 — legacy public exception
)
from resolv.core.model_stream import stream_model_decision as _stream_response
from resolv.core.prompts import AGENT_LOOP_PROMPT
from resolv.tool_calling import tool_result_to_message
from resolv.tools.cancel import cancel_order
from resolv.tools.executor import (
    ToolExecutionResult,
    execute_approved_tool_call,
    execute_tool_call,
)
from resolv.tools.logistics import query_logistics
from resolv.tools.order import query_order
from resolv.tools.refund import apply_refund, query_refund_status
from resolv.tools.registry import ToolEffect, get_registered_tool

logger = logging.getLogger(__name__)


class MaxIterationsExceeded(RuntimeError):
    """The model did not produce a final answer within the allowed rounds."""


class UnsupportedApprovalBatchError(RuntimeError):
    """This phase cannot safely suspend a batch containing a WRITE call."""


@dataclass(frozen=True)
class AgentCompleted:
    text: str


@dataclass(frozen=True)
class AgentApprovalRequired:
    request: ApprovalRequest
    state: SuspendedAgentState


AgentRunResult = AgentCompleted | AgentApprovalRequired


def run_agent(
    question: str,
    *,
    history: Sequence[BaseMessage] = (),
    max_iterations: int = 8,
) -> AgentRunResult:
    """Run until final text or suspend before one WRITE Tool call."""
    return _final_result(
        stream_agent_events(question, history=history, max_iterations=max_iterations)
    )


def stream_agent_events(
    question: str,
    *,
    history: Sequence[BaseMessage] = (),
    max_iterations: int = 8,
) -> Iterator[AgentEvent]:
    """Yield step-level events from the same execution path as run_agent."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    prompt_value = AGENT_LOOP_PROMPT.invoke(
        {"history": list(history), "question": question}
    )
    yield from _continue_agent_events(prompt_value.to_messages(), max_iterations)


def resume_agent(state: SuspendedAgentState, *, approved: bool) -> AgentRunResult:
    """Resolve the exact pending call once, then continue its original messages."""
    if type(approved) is not bool:
        raise TypeError("approved must be a bool")
    snapshot, grant = consume_suspended_state(state, approved=approved)
    messages = list(snapshot.messages)
    pending = snapshot.pending_call
    if approved:
        assert grant is not None
        result = execute_approved_tool_call(grant)
    else:
        result = ToolExecutionResult(
            pending.tool_call_id,
            pending.name,
            {"executed": False, "reason": "user_rejected"},
        )
    messages.append(tool_result_to_message(result))
    return _final_result(
        _continue_agent_events(messages, snapshot.remaining_iterations)
    )


def _final_result(events: Iterator[AgentEvent]) -> AgentRunResult:
    result: AgentRunResult | None = None
    for event in events:
        if isinstance(event, AgentMessageEvent):
            result = AgentCompleted(event.text)
        elif isinstance(event, AgentApprovalRequiredEvent):
            result = AgentApprovalRequired(event.request, event.state)
    if result is None:
        raise RuntimeError("Agent stopped without a final result")
    return result


def _continue_agent_events(
    messages: list[BaseMessage], remaining: int
) -> Iterator[AgentEvent]:
    model = get_chat_model()
    model_with_tools = model.bind_tools(
        [query_order, query_logistics, query_refund_status, cancel_order, apply_refund]
    )

    for iteration in range(remaining):
        response, text_deltas = _stream_response(model_with_tools, messages)
        messages.append(response)
        logger.info("Agent iteration: %s tool call(s)", len(response.tool_calls))

        if not response.tool_calls:
            for text in text_deltas:
                yield AgentTextDeltaEvent(text)
            yield AgentMessageEvent(response.text)
            yield AgentDoneEvent()
            return

        registered = [get_registered_tool(call["name"]) for call in response.tool_calls]
        write_indices = [
            index
            for index, entry in enumerate(registered)
            if entry.effect is ToolEffect.WRITE
        ]
        if write_indices:
            if len(response.tool_calls) != 1:
                raise UnsupportedApprovalBatchError(
                    "A batch containing WRITE calls cannot be approved in this phase."
                )
            remaining_after_call = remaining - iteration - 1
            if remaining_after_call < 1:
                raise MaxIterationsExceeded(
                    "No model iteration remains after approval."
                )
            pending = PendingToolCall.from_tool_call(
                response.tool_calls[0], ToolEffect.WRITE
            )
            state = suspend_agent(messages, pending, remaining_after_call)
            arguments = json.dumps(pending.args, ensure_ascii=False, sort_keys=True)
            request = ApprovalRequest(
                pending, f"即将执行 {pending.name}，参数：{arguments}。是否确认？"
            )
            yield AgentApprovalRequiredEvent(request, state)
            yield AgentDoneEvent()
            return

        for tool_call in response.tool_calls:
            yield AgentToolCallEvent(
                tool_call["id"], tool_call["name"], dict(tool_call["args"])
            )
            result = execute_tool_call(tool_call)
            messages.append(tool_result_to_message(result))
            yield AgentToolResultEvent(result.tool_call_id, result.name, result.output)

    raise MaxIterationsExceeded(f"Agent exceeded {remaining} remaining iterations")
