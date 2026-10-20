from sqlalchemy import select
from sqlalchemy.orm import Session

from resolv.models import Shipment


class ShipmentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_tracking_no(self, tracking_no: str) -> Shipment | None:
        return self.session.scalar(
            select(Shipment).where(Shipment.tracking_no == tracking_no)
        )

    def get_by_order_id(self, order_id: int) -> Shipment | None:
        return self.session.scalar(
            select(Shipment).where(Shipment.order_id == order_id)
        )
