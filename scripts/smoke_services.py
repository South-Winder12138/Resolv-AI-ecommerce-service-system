from resolv.core.config import DatabaseSettings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.repositories.order import OrderRepository
from resolv.repositories.shipment import ShipmentRepository
from resolv.services.order import OrderService
from resolv.services.shipment import ShipmentService


def main() -> None:
    engine = create_db_engine(DatabaseSettings().database_url)
    try:
        init_db(engine)
        session_factory = create_session_factory(engine)
        with session_factory.begin() as session:
            seed_data(session)
        with session_factory() as session:
            orders = OrderService(OrderRepository(session))
            shipments = ShipmentService(ShipmentRepository(session))
            print("MH001:", orders.get_order("MH001"))
            print("MH999:", orders.get_order("MH999"))
            print("SF001:", shipments.get_shipment("SF001"))
            print("SF999:", shipments.get_shipment("SF999"))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
