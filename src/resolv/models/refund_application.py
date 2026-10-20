"""Canonical applications and command receipts live in the same SQLite transaction."""

from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from resolv.db.base import Base
from resolv.db.conversation_base import UTCDateTime
from resolv.db.exact_json import ExactJSON


class RefundApplication(Base):
    __tablename__ = "refund_applications"
    __table_args__ = (
        UniqueConstraint("order_source", "order_id", name="uq_refund_application_order"),
        CheckConstraint("status IN ('pending','approved','rejected','completed')", name="ck_refund_application_status"),
        CheckConstraint("origin IN ('legacy','offer')", name="ck_refund_application_origin"),
    )
    application_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_source: Mapped[str] = mapped_column(String(128))
    order_id: Mapped[str] = mapped_column(String(128))
    subject_ref: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text)
    money: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    origin: Mapped[str] = mapped_column(String(16))
    offer_id: Mapped[str | None] = mapped_column(String(64))
    command_id: Mapped[str | None] = mapped_column(String(64))
    decision_hash: Mapped[str | None] = mapped_column(String(64))
    evidence_hash: Mapped[str | None] = mapped_column(String(64))
    revalidation_hash: Mapped[str | None] = mapped_column(String(64))
    snapshots: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    legacy_refund_id: Mapped[int | None] = mapped_column(ForeignKey("refunds.id"), unique=True)


class RefundSubmissionReceipt(Base):
    __tablename__ = "refund_submission_receipts"
    command_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("refund_applications.application_id"))
    request_hash: Mapped[str] = mapped_column(String(64))
    command_hash: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(ExactJSON)
    result_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
