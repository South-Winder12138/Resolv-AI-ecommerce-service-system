from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from langchain_core.tools import BaseTool

from resolv.tools.cancel import cancel_order, cancel_order_in_transaction
from resolv.tools.logistics import query_logistics
from resolv.tools.order import query_order
from resolv.tools.refund import (
    apply_refund,
    apply_refund_in_transaction,
    query_refund_status,
)


class ToolEffect(StrEnum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class RegisteredTool:
    tool: BaseTool
    effect: ToolEffect
    transaction_handler: Callable | None = None

    def __post_init__(self):
        if self.effect is ToolEffect.WRITE and self.transaction_handler is None:
            raise ValueError("WRITE registry entries require a transaction handler")


TOOL_REGISTRY: dict[str, RegisteredTool] = {
    query_order.name: RegisteredTool(query_order, ToolEffect.READ),
    query_logistics.name: RegisteredTool(query_logistics, ToolEffect.READ),
    query_refund_status.name: RegisteredTool(query_refund_status, ToolEffect.READ),
    cancel_order.name: RegisteredTool(
        cancel_order, ToolEffect.WRITE, cancel_order_in_transaction
    ),
    apply_refund.name: RegisteredTool(
        apply_refund, ToolEffect.WRITE, apply_refund_in_transaction
    ),
}


class ToolExecutionError(Exception):
    """Tool execution failed before producing a business result."""


class UnknownToolError(ToolExecutionError):
    """A requested tool is not registered in this application."""


class ToolInputError(ToolExecutionError):
    """A requested tool's arguments failed schema validation."""


class WriteToolApprovalRequired(ToolExecutionError):
    """Automatic execution of a write tool is not allowed."""


def get_registered_tool(name: str) -> RegisteredTool:
    try:
        return TOOL_REGISTRY[name]
    except KeyError:
        raise UnknownToolError(f"Unknown tool: {name}") from None


def get_tool(name: str) -> BaseTool:
    """Keep the Phase 8 lookup API available to existing callers."""
    return get_registered_tool(name).tool
