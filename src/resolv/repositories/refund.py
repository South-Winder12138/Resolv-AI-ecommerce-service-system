"""Refund record access within a caller-owned SQLAlchemy Session."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from resolv.models import Refund


class RefundRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_order_id(self, order_id: int) -> Refund | None:
        return self.session.scalar(select(Refund).where(Refund.order_id == order_id))

    def add(self, refund: Refund) -> None:
        self.session.add(refund)
