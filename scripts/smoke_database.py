from sqlalchemy import select

from resolv.core.config import DatabaseSettings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.models import Order


def main() -> None:
    engine = create_db_engine(DatabaseSettings().database_url)
    init_db(engine)
    session_factory = create_session_factory(engine)
    with session_factory.begin() as session:
        seed_data(session)
    with session_factory() as session:
        orders = session.scalars(select(Order).order_by(Order.order_no)).all()
        print("Orders:", [(order.order_no, order.status) for order in orders])
        mh001 = session.scalar(select(Order).where(Order.order_no == "MH001"))
        assert mh001 is not None
        print("MH001:", mh001.order_no, mh001.status)
        shipment = mh001.shipment
        print("MH001 shipment:", shipment.tracking_no if shipment else None)
        print("SF001 status:", shipment.status if shipment else None)
        print("SF001 location:", shipment.location if shipment else None)
    engine.dispose()


if __name__ == "__main__":
    main()
