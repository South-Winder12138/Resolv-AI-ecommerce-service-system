from dataclasses import dataclass

from langchain_core.messages import ToolCall
from pydantic import ValidationError

from resolv.approval import redeem_approved_call
from resolv.tools.registry import (
    ToolEffect,
    ToolExecutionError,
    ToolInputError,
    WriteToolApprovalRequired,
    get_registered_tool,
)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    name: str
    output: object


def execute_tool_call(tool_call: ToolCall) -> ToolExecutionResult:
    """Resolve and run one requested tool; do not call the model again."""
    name = tool_call["name"]
    registered = get_registered_tool(name)
    if registered.effect is ToolEffect.WRITE:
        raise WriteToolApprovalRequired(f"Write tool requires approval: {name}")
    output = _invoke_tool(name, tool_call["args"])
    return ToolExecutionResult(tool_call_id=tool_call["id"], name=name, output=output)


def execute_approved_tool_call(grant: object) -> ToolExecutionResult:
    """Execute only a one-use grant minted from a suspended Agent state."""
    pending = redeem_approved_call(grant)
    registered = get_registered_tool(pending.name)
    if (
        pending.effect is not ToolEffect.WRITE
        or registered.effect is not ToolEffect.WRITE
    ):
        raise WriteToolApprovalRequired("Approved execution requires a WRITE tool.")
    output = _invoke_tool(pending.name, pending.args)
    return ToolExecutionResult(pending.tool_call_id, pending.name, output)


def _invoke_tool(name: str, args: dict[str, object]) -> object:
    registered = get_registered_tool(name)
    try:
        return registered.tool.invoke(args)
    except ValidationError as exc:
        raise ToolInputError(f"Invalid input for tool {name}") from exc
    except Exception as exc:
        raise ToolExecutionError(f"Tool execution failed: {name}") from exc


def execute_persisted_tool_call(
    action_id, execution, write_service
) -> ToolExecutionResult:
    from resolv.repositories.conversation import ConversationRepository
    from resolv.schemas.conversation import ConversationError

    execution.assert_owned()
    with write_service.conversation_session_factory() as session:
        action = ConversationRepository(session, execution.service.owner_id).action(
            action_id
        )
        if (
            action.run_id != execution.run_id
            or action.effect != "write"
            or get_registered_tool(action.tool_name).effect != ToolEffect.WRITE
        ):
            raise ConversationError("action_conflict", run_id=execution.run_id)
    return write_service.execute(action_id, execution)
