from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.models import Order, Shipment
from resolv.repositories.order import OrderRepository
from resolv.repositories.shipment import ShipmentRepository
from resolv.services.order import OrderService
from resolv.services.shipment import ShipmentService


@pytest.fixture
def session_factory(tmp_path: Path) -> sessionmaker[Session]:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'services.db'}")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        seed_data(session)
    yield factory
    engine.dispose()


def test_order_repository_returns_order(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        order = OrderRepository(session).get_by_order_no("MH001")
        assert isinstance(order, Order)
        assert order.status == "shipped"


def test_order_repository_returns_none_when_missing(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        assert OrderRepository(session).get_by_order_no("MH999") is None


def test_shipment_repository_finds_by_tracking_and_order_id(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        shipment_repo = ShipmentRepository(session)
        shipment = shipment_repo.get_by_tracking_no("SF001")
        assert isinstance(shipment, Shipment)
        assert shipment_repo.get_by_order_id(shipment.order_id) is shipment
        assert shipment_repo.get_by_tracking_no("SF999") is None
        assert shipment_repo.get_by_order_id(999) is None


def test_order_service_returns_business_result(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        service = OrderService(OrderRepository(session))
        result = service.get_order("MH001")
    assert result == {
        "found": True,
        "order_id": "MH001",
        "status": "shipped",
        "tracking_no": "SF001",
    }
    assert not isinstance(result, (Order, Session))


def test_order_service_handles_missing_order_and_no_shipment(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        service = OrderService(OrderRepository(session))
        assert service.get_order("MH999") == {"found": False, "order_id": "MH999"}
        assert service.get_order("MH002") == {
            "found": True,
            "order_id": "MH002",
            "status": "processing",
            "tracking_no": None,
        }


def test_shipment_service_returns_business_result_and_not_found(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        service = ShipmentService(ShipmentRepository(session))
        result = service.get_shipment("SF001")
        missing = service.get_shipment("SF999")
    assert result == {
        "found": True,
        "tracking_no": "SF001",
        "status": "in_transit",
        "location": "上海转运中心",
    }
    assert missing == {"found": False, "tracking_no": "SF999"}
    assert not isinstance(result, (Shipment, Session))
