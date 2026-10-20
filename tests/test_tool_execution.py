import pytest

from resolv.tools.executor import ToolExecutionResult, execute_tool_call
from resolv.tools.order import query_order
from resolv.tools.registry import ToolInputError, UnknownToolError, get_tool

pytestmark = pytest.mark.usefixtures("tool_database")


def test_registry_finds_query_order_tool() -> None:
    assert get_tool("query_order") is query_order


def test_executor_runs_order_tool_and_preserves_call_id() -> None:
    result = execute_tool_call(
        {"name": "query_order", "args": {"order_id": "MH001"}, "id": "test-call-1"}
    )

    assert isinstance(result, ToolExecutionResult)
    assert result.tool_call_id == "test-call-1"
    assert result.name == "query_order"
    assert result.output == {
        "order_id": "MH001",
        "found": True,
        "status": "shipped",
        "tracking_no": "SF001",
    }


def test_order_not_found_is_business_result() -> None:
    result = execute_tool_call(
        {"name": "query_order", "args": {"order_id": "MH999"}, "id": "test-call-2"}
    )

    assert result.output == {"order_id": "MH999", "found": False}


def test_unknown_tool_raises_domain_error() -> None:
    with pytest.raises(UnknownToolError, match="Unknown tool: unknown_tool"):
        execute_tool_call({"name": "unknown_tool", "args": {}, "id": "test-call-3"})


@pytest.mark.parametrize("args", [{}, {"order_id": 123}])
def test_invalid_args_are_rejected_by_tool_schema(args) -> None:
    with pytest.raises(ToolInputError, match="Invalid input for tool query_order"):
        execute_tool_call({"name": "query_order", "args": args, "id": "test-call-4"})
