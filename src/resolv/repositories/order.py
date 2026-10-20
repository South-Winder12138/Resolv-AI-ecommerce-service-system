from sqlalchemy import select
from sqlalchemy.orm import Session

from resolv.models import Order


class OrderRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_order_no(self, order_no: str) -> Order | None:
        return self.session.scalar(select(Order).where(Order.order_no == order_no))

    def set_status(self, order: Order, status: str) -> None:
        order.status = status
