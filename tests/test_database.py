from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.models import Order, Shipment


@pytest.fixture
def session_factory(tmp_path: Path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    yield create_session_factory(engine)
    engine.dispose()


def test_creates_domain_tables(session_factory) -> None:
    assert set(inspect(session_factory.kw["bind"]).get_table_names()) == {
        "orders",
        "refunds",
        "shipments",
        "tool_execution_receipts",
        "refund_applications",
        "refund_submission_receipts",
    }


def test_insert_and_select_order(session_factory) -> None:
    with session_factory.begin() as session:
        session.add(Order(order_no="MH100", status="processing"))
    with session_factory() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH100"))
        assert order is not None
        assert order.status == "processing"
        assert order.created_at is not None


def test_order_number_is_unique(session_factory) -> None:
    with pytest.raises(IntegrityError), session_factory.begin() as session:
        session.add_all(
            [
                Order(order_no="MH100", status="processing"),
                Order(order_no="MH100", status="shipped"),
            ]
        )


def test_shipment_foreign_key_is_enforced(session_factory) -> None:
    with pytest.raises(IntegrityError), session_factory.begin() as session:
        session.add(Shipment(order_id=999, tracking_no="SF999", status="in_transit"))


def test_order_shipment_relationship(session_factory) -> None:
    with session_factory.begin() as session:
        session.add(
            Order(
                order_no="MH100",
                status="shipped",
                shipment=Shipment(
                    tracking_no="SF100", status="in_transit", location="上海转运中心"
                ),
            )
        )
    with session_factory() as session:
        order = session.scalar(select(Order).where(Order.order_no == "MH100"))
        assert order is not None
        assert order.shipment is not None
        assert order.shipment.tracking_no == "SF100"
        assert order.shipment.order is order


def test_seed_is_idempotent(session_factory) -> None:
    with session_factory.begin() as session:
        seed_data(session)
    with session_factory.begin() as session:
        seed_data(session)
    with session_factory() as session:
        assert len(session.scalars(select(Order)).all()) == 2
        shipments = session.scalars(select(Shipment)).all()
        assert len(shipments) == 1
        assert shipments[0].tracking_no == "SF001"
