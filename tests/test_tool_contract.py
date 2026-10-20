import pytest

from resolv.tools.executor import ToolExecutionResult, execute_tool_call
from resolv.tools.logistics import query_logistics
from resolv.tools.order import query_order
from resolv.tools.registry import (
    ToolEffect,
    ToolExecutionError,
    get_registered_tool,
    get_tool,
)

pytestmark = pytest.mark.usefixtures("tool_database")


@pytest.mark.parametrize(
    ("name", "expected_tool"),
    [("query_order", query_order), ("query_logistics", query_logistics)],
)
def test_registered_read_tool_has_effect_without_model_parameter(name, expected_tool):
    registration = get_registered_tool(name)
    assert registration.tool is expected_tool
    assert registration.effect is ToolEffect.READ
    assert get_tool(name) is expected_tool
    assert "effect" not in expected_tool.args


def test_not_found_is_successful_tool_execution() -> None:
    result = execute_tool_call(
        {"name": "query_order", "args": {"order_id": "MH999"}, "id": "call-missing"}
    )
    assert isinstance(result, ToolExecutionResult)
    assert result.tool_call_id == "call-missing"
    assert result.output == {"found": False, "order_id": "MH999"}


def test_backend_failure_is_safe_project_error(monkeypatch) -> None:
    from resolv.services.order import OrderService

    def broken_lookup(_self, _order_no):
        raise RuntimeError("database path /private/secret.db")

    monkeypatch.setattr(OrderService, "get_order", broken_lookup)
    with pytest.raises(ToolExecutionError) as captured:
        execute_tool_call(
            {"name": "query_order", "args": {"order_id": "MH001"}, "id": "call-1"}
        )
    assert str(captured.value) == "Tool execution failed: query_order"
    assert "/private/secret.db" not in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)
