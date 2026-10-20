import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from resolv.models import Order
from resolv.repositories.order import OrderRepository
from resolv.services.order import OrderService
from resolv.tools.cancel import cancel_order
from resolv.tools.executor import execute_tool_call
from resolv.tools.registry import (
    ToolEffect,
    WriteToolApprovalRequired,
    get_registered_tool,
)

pytestmark = pytest.mark.usefixtures("tool_database")


def order_status(factory: sessionmaker[Session], order_no: str) -> str:
    with factory() as session:
        order = session.scalar(select(Order).where(Order.order_no == order_no))
        assert order is not None
        return order.status


def test_cancel_tool_contract_is_write_without_internal_arguments() -> None:
    registered = get_registered_tool("cancel_order")
    assert registered.tool is cancel_order
    assert registered.effect is ToolEffect.WRITE
    assert set(cancel_order.args) == {"order_id"}


def test_processing_order_is_cancelled_and_committed(
    tool_database: sessionmaker[Session],
) -> None:
    assert order_status(tool_database, "MH002") == "processing"
    assert cancel_order.invoke({"order_id": "MH002"}) == {
        "success": True,
        "order_id": "MH002",
        "status": "cancelled",
        "changed": True,
    }
    assert order_status(tool_database, "MH002") == "cancelled"


def test_cancel_is_idempotent(tool_database: sessionmaker[Session]) -> None:
    first = cancel_order.invoke({"order_id": "MH002"})
    second = cancel_order.invoke({"order_id": "MH002"})
    assert first["changed"] is True
    assert second == {
        "success": True,
        "order_id": "MH002",
        "status": "cancelled",
        "changed": False,
    }
    assert order_status(tool_database, "MH002") == "cancelled"


def test_shipped_and_unknown_orders_are_business_results(
    tool_database: sessionmaker[Session],
) -> None:
    assert cancel_order.invoke({"order_id": "MH001"}) == {
        "success": False,
        "order_id": "MH001",
        "status": "shipped",
        "reason": "order_not_cancellable",
    }
    assert cancel_order.invoke({"order_id": "MH999"}) == {
        "success": False,
        "order_id": "MH999",
        "reason": "order_not_found",
    }
    assert order_status(tool_database, "MH001") == "shipped"


def test_unrecognized_status_is_not_cancellable(
    tool_database: sessionmaker[Session],
) -> None:
    with tool_database.begin() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH002"))
        assert order is not None
        order.status = "on_hold"

    assert cancel_order.invoke({"order_id": "MH002"}) == {
        "success": False,
        "order_id": "MH002",
        "status": "on_hold",
        "reason": "order_not_cancellable",
    }
    assert order_status(tool_database, "MH002") == "on_hold"


def test_repository_does_not_commit_service_change(
    tool_database: sessionmaker[Session],
) -> None:
    with tool_database() as session:
        result = OrderService(OrderRepository(session)).cancel_order("MH002")
        assert result["changed"] is True
    assert order_status(tool_database, "MH002") == "processing"


def test_exception_after_change_rolls_back(
    tool_database: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = OrderRepository.set_status

    def fail_after_change(self: OrderRepository, order: Order, status: str) -> None:
        original(self, order, status)
        self.session.flush()
        raise RuntimeError("failure after SQL UPDATE")

    monkeypatch.setattr(OrderRepository, "set_status", fail_after_change)
    with pytest.raises(RuntimeError, match="failure after SQL UPDATE"):
        cancel_order.invoke({"order_id": "MH002"})
    assert order_status(tool_database, "MH002") == "processing"


def test_executor_rejects_write_before_side_effect(
    tool_database: sessionmaker[Session],
) -> None:
    with pytest.raises(WriteToolApprovalRequired, match="cancel_order"):
        execute_tool_call(
            {"name": "cancel_order", "args": {"order_id": "MH002"}, "id": "call-1"}
        )
    assert order_status(tool_database, "MH002") == "processing"


def test_read_tools_remain_automatically_executable() -> None:
    for name, args in (
        ("query_order", {"order_id": "MH001"}),
        ("query_logistics", {"tracking_no": "SF001"}),
    ):
        result = execute_tool_call({"name": name, "args": args, "id": "call-read"})
        assert result.output["found"] is True
