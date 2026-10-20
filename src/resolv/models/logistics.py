"""One durable audit row per attempted remote logistics read."""

from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from resolv.db.conversation_base import ConversationBase, UTCDateTime, utc_now
from resolv.db.exact_json import ExactJSON
from resolv.models.conversation import OPTIONS


class ConversationLogisticsAttempt(ConversationBase):
    __tablename__ = "conversation_logistics_attempts"
    __table_args__ = (
        UniqueConstraint("action_id", "attempt_no", name="uq_logistics_action_attempt"),
        CheckConstraint("state IN ('prepared','completed','failed','unknown')", name="ck_logistics_attempt_state"),
        CheckConstraint("attempt_no > 0", name="ck_logistics_attempt_no"),
        OPTIONS,
    )
    attempt_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    action_id: Mapped[str] = mapped_column(ForeignKey("conversation_actions.action_id"))
    run_id: Mapped[str] = mapped_column(ForeignKey("conversation_runs.run_id"))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.conversation_id"))
    owner_id: Mapped[str] = mapped_column(String(64))
    execution_epoch: Mapped[int] = mapped_column(Integer)
    attempt_no: Mapped[int] = mapped_column(Integer)
    config_hash: Mapped[str] = mapped_column(String(64))
    arguments_hash: Mapped[str] = mapped_column(String(64))
    tracking_no: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(24), default="prepared")
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    elapsed_ms: Mapped[int | None] = mapped_column(Integer)
    envelope: Mapped[dict | None] = mapped_column(ExactJSON)
    projection: Mapped[dict | None] = mapped_column(ExactJSON)
    result_hash: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(64))

