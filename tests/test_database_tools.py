import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import resolv.tools.order as order_module
from resolv.db.engine import get_session_factory
from resolv.models import Order, Shipment
from resolv.tools.executor import execute_tool_call
from resolv.tools.logistics import query_logistics
from resolv.tools.order import query_order
from resolv.tools.registry import get_tool

pytestmark = pytest.mark.usefixtures("tool_database")


def test_order_tool_reads_database_and_not_found() -> None:
    assert query_order.invoke({"order_id": "MH001"}) == {
        "found": True,
        "order_id": "MH001",
        "status": "shipped",
        "tracking_no": "SF001",
    }
    assert query_order.invoke({"order_id": "MH999"}) == {
        "found": False,
        "order_id": "MH999",
    }


def test_logistics_tool_reads_database_and_not_found() -> None:
    assert query_logistics.invoke({"tracking_no": "SF001"}) == {
        "found": True,
        "tracking_no": "SF001",
        "status": "in_transit",
        "location": "上海转运中心",
    }
    assert query_logistics.invoke({"tracking_no": "SF999"}) == {
        "found": False,
        "tracking_no": "SF999",
    }


def test_database_change_is_visible_to_tool(
    tool_database: sessionmaker[Session],
) -> None:
    with tool_database.begin() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH001"))
        assert order is not None
        order.status = "delivered"

    assert query_order.invoke({"order_id": "MH001"})["status"] == "delivered"


def test_tool_result_does_not_expose_orm_objects() -> None:
    for result in (
        query_order.invoke({"order_id": "MH001"}),
        query_logistics.invoke({"tracking_no": "SF001"}),
    ):
        assert isinstance(result, dict)
        assert all(
            not isinstance(value, (Order, Shipment, Session))
            for value in result.values()
        )


def test_tool_closes_its_session(
    tool_database: sessionmaker[Session], monkeypatch
) -> None:
    closed: list[bool] = []

    class TrackedSession(Session):
        def close(self) -> None:
            closed.append(True)
            super().close()

    tracked_factory = sessionmaker(bind=tool_database.kw["bind"], class_=TrackedSession)
    monkeypatch.setattr(
        order_module, "get_session_factory", lambda _url: tracked_factory
    )

    query_order.invoke({"order_id": "MH001"})

    assert closed == [True]


def test_registry_and_executor_still_run_both_tools() -> None:
    for name, args in (
        ("query_order", {"order_id": "MH001"}),
        ("query_logistics", {"tracking_no": "SF001"}),
    ):
        assert get_tool(name).name == name
        result = execute_tool_call({"name": name, "args": args, "id": f"call-{name}"})
        assert result.tool_call_id == f"call-{name}"
        assert result.output["found"] is True


def test_missing_database_requires_explicit_initialization(tmp_path) -> None:
    database_file = tmp_path / "missing.db"
    with pytest.raises(RuntimeError, match="initialize and seed"):
        get_session_factory(f"sqlite:///{database_file}")
    assert not database_file.exists()
