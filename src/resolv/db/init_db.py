from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from resolv.db.base import Base
from resolv.models import Order, Shipment  # register all mappings


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def seed_data(session: Session) -> None:
    mh001 = session.scalar(select(Order).where(Order.order_no == "MH001"))
    if mh001 is None:
        mh001 = Order(order_no="MH001", status="shipped")
        session.add(mh001)
    if session.scalar(select(Order).where(Order.order_no == "MH002")) is None:
        session.add(Order(order_no="MH002", status="processing"))
    if session.scalar(select(Shipment).where(Shipment.tracking_no == "SF001")) is None:
        mh001.shipment = Shipment(
            tracking_no="SF001", status="in_transit", location="上海转运中心"
        )
