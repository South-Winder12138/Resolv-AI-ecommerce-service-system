from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from resolv.db.base import Base

if TYPE_CHECKING:
    from resolv.models.refund import Refund
    from resolv.models.shipment import Shipment


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_no: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utc_now
    )
    shipment: Mapped["Shipment | None"] = relationship(
        back_populates="order", uselist=False
    )
    refund: Mapped["Refund | None"] = relationship(
        back_populates="order", uselist=False
    )
