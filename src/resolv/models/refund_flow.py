"""Conversation-owned refund state; business applications live in SQLite."""

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from resolv.db.conversation_base import ConversationBase, UTCDateTime, utc_now
from resolv.db.exact_json import ExactJSON
from resolv.models.conversation import OPTIONS


class ConversationRefundRun(ConversationBase):
    __tablename__ = "conversation_refund_runs"
    __table_args__ = (OPTIONS,)
    run_id: Mapped[str] = mapped_column(ForeignKey("conversation_runs.run_id"), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.conversation_id"))
    understanding_id: Mapped[str] = mapped_column(ForeignKey("conversation_understandings.understanding_id"))
    primary_task_id: Mapped[str] = mapped_column(ForeignKey("conversation_understanding_requests.task_id"))
    primary_hash: Mapped[str] = mapped_column(String(64))
    primary_json: Mapped[dict] = mapped_column(ExactJSON)
    profile_json: Mapped[dict] = mapped_column(ExactJSON)
    profile_hash: Mapped[str] = mapped_column(String(64))
    stage_data: Mapped[dict] = mapped_column(ExactJSON, default=dict)
    model_requests: Mapped[list] = mapped_column(ExactJSON, default=list)
    observations: Mapped[list] = mapped_column(ExactJSON, default=list)
    audits: Mapped[list] = mapped_column(ExactJSON, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    step_limit: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now)


class ConversationRefundOffer(ConversationBase):
    __tablename__ = "conversation_refund_offers"
    __table_args__ = (UniqueConstraint("run_id", name="uq_refund_offer_run"), OPTIONS)
    offer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("conversation_refund_runs.run_id"))
    owner_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.conversation_id"))
    payload: Mapped[dict] = mapped_column(ExactJSON)
    payload_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="available")
    invalid_reason: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ConversationRefundSubmission(ConversationBase):
    __tablename__ = "conversation_refund_submissions"
    __table_args__ = (UniqueConstraint("offer_id", name="uq_refund_submission_offer"), OPTIONS)
    command_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    offer_id: Mapped[str] = mapped_column(ForeignKey("conversation_refund_offers.offer_id"))
    owner_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.conversation_id"))
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="created")
    execution_epoch: Mapped[int] = mapped_column(Integer, default=0)
    execution_owner_id: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    revalidations: Mapped[list] = mapped_column(ExactJSON, default=list)
    receipt: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    receipt_hash: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(64))
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now)


class ConversationRefundRequestAlias(ConversationBase):
    __tablename__ = "conversation_refund_request_aliases"
    __table_args__ = (OPTIONS,)
    owner_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    command_id: Mapped[str] = mapped_column(ForeignKey("conversation_refund_submissions.command_id"))
    request_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
