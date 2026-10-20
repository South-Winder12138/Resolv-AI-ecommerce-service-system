"""Agent execution events, independent of HTTP and SSE formatting."""

from dataclasses import dataclass
from typing import Literal

from resolv.approval import ApprovalRequest, SuspendedAgentState


@dataclass(frozen=True)
class AgentToolCallEvent:
    tool_call_id: str
    tool: str
    arguments: dict[str, object]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class AgentToolResultEvent:
    tool_call_id: str
    tool: str
    result: object
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True)
class AgentApprovalRequiredEvent:
    request: ApprovalRequest
    state: SuspendedAgentState
    type: Literal["approval_required"] = "approval_required"


@dataclass(frozen=True)
class AgentMessageEvent:
    text: str
    type: Literal["message"] = "message"


@dataclass(frozen=True)
class AgentTextDeltaEvent:
    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True)
class AgentDoneEvent:
    type: Literal["done"] = "done"


@dataclass(frozen=True)
class AgentErrorEvent:
    code: str
    message: str
    type: Literal["error"] = "error"


@dataclass(frozen=True)
class ConversationEvent:
    type: str
    run_id: str
    data: dict


AgentEvent = (
    ConversationEvent
    | AgentToolCallEvent
    | AgentToolResultEvent
    | AgentApprovalRequiredEvent
    | AgentMessageEvent
    | AgentTextDeltaEvent
    | AgentDoneEvent
    | AgentErrorEvent
)
