"""Conversation-owned knowledge state; separate from the knowledge corpus store."""

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from resolv.db.conversation_base import ConversationBase, UTCDateTime, utc_now
from resolv.db.exact_json import ExactJSON
from resolv.models.conversation import OPTIONS


class ConversationKnowledgeRun(ConversationBase):
    __tablename__ = "conversation_knowledge_runs"
    __table_args__ = (OPTIONS,)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_runs.run_id"), primary_key=True
    )
    owner_id: Mapped[str] = mapped_column(String(64))
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id")
    )
    understanding_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_understandings.understanding_id")
    )
    primary_task_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_understanding_requests.task_id")
    )
    primary_hash: Mapped[str] = mapped_column(String(64))
    primary_json: Mapped[dict] = mapped_column(ExactJSON)
    profile_json: Mapped[dict] = mapped_column(ExactJSON)
    profile_hash: Mapped[str] = mapped_column(String(64))
    query_payload: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    query_input_hash: Mapped[str | None] = mapped_column(String(64))
    query_output_hash: Mapped[str | None] = mapped_column(String(64))
    retrieval_payload: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    retrieval_input_hash: Mapped[str | None] = mapped_column(String(64))
    retrieval_output_hash: Mapped[str | None] = mapped_column(String(64))
    check_payload: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    check_input_hash: Mapped[str | None] = mapped_column(String(64))
    check_output_hash: Mapped[str | None] = mapped_column(String(64))
    answer_input_payload: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    answer_input_input_hash: Mapped[str | None] = mapped_column(String(64))
    answer_input_output_hash: Mapped[str | None] = mapped_column(String(64))
    answer_payload: Mapped[dict | None] = mapped_column(ExactJSON(none_as_null=True))
    answer_input_hash: Mapped[str | None] = mapped_column(String(64))
    answer_output_hash: Mapped[str | None] = mapped_column(String(64))
    audits: Mapped[list] = mapped_column(ExactJSON, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, onupdate=utc_now
    )


class ConversationKnowledgeReviewItem(ConversationBase):
    __tablename__ = "conversation_knowledge_review_items"
    __table_args__ = (OPTIONS,)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_knowledge_runs.run_id"), primary_key=True
    )
    item_json: Mapped[dict] = mapped_column(ExactJSON)
    item_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
