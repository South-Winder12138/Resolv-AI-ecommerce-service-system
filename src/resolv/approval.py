"""One-use, in-process snapshots for approval of a specific write call."""

import json
from copy import deepcopy
from dataclasses import dataclass
from secrets import token_urlsafe
from typing import Any

from langchain_core.messages import BaseMessage, ToolCall

from resolv.tools.registry import ToolEffect


class InvalidApprovalState(RuntimeError):
    """The approval state or grant was forged, changed, or already consumed."""


@dataclass(frozen=True)
class PendingToolCall:
    tool_call_id: str
    name: str
    _args_json: str
    effect: ToolEffect

    @classmethod
    def from_tool_call(cls, call: ToolCall, effect: ToolEffect) -> "PendingToolCall":
        return cls(
            tool_call_id=call["id"],
            name=call["name"],
            _args_json=json.dumps(call["args"], sort_keys=True, ensure_ascii=False),
            effect=effect,
        )

    @property
    def args(self) -> dict[str, Any]:
        """Return a copy; callers cannot change the frozen argument snapshot."""
        return json.loads(self._args_json)


@dataclass(frozen=True)
class ApprovalRequest:
    pending_call: PendingToolCall
    message: str


@dataclass(frozen=True, eq=False)
class SuspendedAgentState:
    messages: tuple[BaseMessage, ...]
    pending_call: PendingToolCall
    remaining_iterations: int
    resume_id: str


@dataclass(frozen=True)
class _RuntimeSnapshot:
    messages: tuple[BaseMessage, ...]
    pending_call: PendingToolCall
    remaining_iterations: int


@dataclass(frozen=True, eq=False)
class _ApprovedCallGrant:
    pending_call: PendingToolCall
    grant_id: str


_pending: dict[str, tuple[SuspendedAgentState, _RuntimeSnapshot]] = {}
_grants: dict[str, _ApprovedCallGrant] = {}


def suspend_agent(
    messages: list[BaseMessage], pending_call: PendingToolCall, remaining: int
) -> SuspendedAgentState:
    resume_id = token_urlsafe(24)
    state = SuspendedAgentState(
        messages=tuple(deepcopy(messages)),
        pending_call=pending_call,
        remaining_iterations=remaining,
        resume_id=resume_id,
    )
    _pending[resume_id] = (
        state,
        _RuntimeSnapshot(tuple(deepcopy(messages)), deepcopy(pending_call), remaining),
    )
    return state


def consume_suspended_state(
    state: SuspendedAgentState, *, approved: bool
) -> tuple[_RuntimeSnapshot, _ApprovedCallGrant | None]:
    entry = _pending.get(state.resume_id)
    if entry is None or entry[0] is not state:
        raise InvalidApprovalState("Approval state is not pending in this process.")
    _pending.pop(state.resume_id)
    snapshot = entry[1]
    if not approved:
        return snapshot, None
    grant_id = token_urlsafe(24)
    grant = _ApprovedCallGrant(snapshot.pending_call, grant_id)
    _grants[grant_id] = grant
    return snapshot, grant


def redeem_approved_call(grant: object) -> PendingToolCall:
    if not isinstance(grant, _ApprovedCallGrant):
        raise InvalidApprovalState("No valid approval grant was supplied.")
    if _grants.get(grant.grant_id) is not grant:
        raise InvalidApprovalState("Approval grant is invalid or already used.")
    _grants.pop(grant.grant_id)
    return grant.pending_call
