"""Additive understanding tables, separate from conversation runs/messages."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from resolv.db.conversation_base import ConversationBase, UTCDateTime, utc_now
from resolv.models.conversation import BODY, OPTIONS


class ConversationUnderstanding(ConversationBase):
    __tablename__ = "conversation_understandings"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "request_id", name="uq_understanding_request"
        ),
        CheckConstraint(
            "status IN ('created','running','finalizing','ready','needs_clarification','interrupted','failed')",
            name="ck_understanding_status",
        ),
        OPTIONS,
    )
    understanding_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id")
    )
    request_id: Mapped[str] = mapped_column(String(128))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    original_text: Mapped[str] = mapped_column(BODY)
    clarifies_understanding_id: Mapped[str | None] = mapped_column(String(64))
    continue_task_id: Mapped[str | None] = mapped_column(String(64))
    clarification_child_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(24), default="created")
    snapshot: Mapped[dict] = mapped_column(JSON)
    resolution: Mapped[dict | None] = mapped_column(JSON)
    classification: Mapped[dict | None] = mapped_column(JSON)
    final_json: Mapped[dict | None] = mapped_column(JSON)
    execution_owner_id: Mapped[str | None] = mapped_column(String(64))
    execution_epoch: Mapped[int] = mapped_column(Integer, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    invocations: Mapped[list] = mapped_column(JSON, default=list)
    error_code: Mapped[str | None] = mapped_column(String(64))
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, onupdate=utc_now
    )


class ConversationUnderstandingRequest(ConversationBase):
    __tablename__ = "conversation_understanding_requests"
    __table_args__ = (
        UniqueConstraint(
            "understanding_id", "ordinal", name="uq_understanding_ordinal"
        ),
        CheckConstraint(
            "role IN ('primary','secondary')", name="ck_understanding_role"
        ),
        CheckConstraint(
            "queue_status IN ('not_queued','pending','selected')",
            name="ck_understanding_queue",
        ),
        CheckConstraint(
            "intent IS NULL OR intent IN ('logistics','order','product_consultation','refund_return','after_sales','complaint','human','chitchat','other')",
            name="ck_understanding_intent",
        ),
        CheckConstraint(
            "branch IS NULL OR branch IN ('business','knowledge','refund_flow','escalate','fallback_script')",
            name="ck_understanding_branch",
        ),
        OPTIONS,
    )
    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    understanding_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_understandings.understanding_id")
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(BODY)
    intent: Mapped[str | None] = mapped_column(String(32))
    branch: Mapped[str | None] = mapped_column(String(24))
    evidence_refs: Mapped[list] = mapped_column(JSON)
    bindings: Mapped[list] = mapped_column(JSON)
    ambiguities: Mapped[list] = mapped_column(JSON)
    unresolved_references: Mapped[list] = mapped_column(JSON)
    queue_status: Mapped[str] = mapped_column(String(16))
    selected_by_understanding_id: Mapped[str | None] = mapped_column(
        String(64), unique=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
