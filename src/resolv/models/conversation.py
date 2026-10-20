from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from resolv.db.conversation_base import ConversationBase, UTCDateTime, utc_now
from resolv.db.exact_json import ExactJSON

OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
}
BODY = Text().with_variant(LONGTEXT(), "mysql")


class Conversation(ConversationBase):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active','archived')", name="ck_conversation_status"
        ),
        OPTIONS,
    )
    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(64), default="local")
    thread_id: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(Text, default="")
    next_message_seq: Mapped[int] = mapped_column(Integer, default=1)
    current_run_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="active")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, onupdate=utc_now
    )


class ConversationRun(ConversationBase):
    __tablename__ = "conversation_runs"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "request_id", name="uq_conversation_request"
        ),
        CheckConstraint(
            "status IN ('running','waiting_approval','interrupted','finalizing','completed','failed')",
            name="ck_run_status",
        ),
        CheckConstraint("mode IN ('agent','chat')", name="ck_run_mode"),
        OPTIONS,
    )
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id")
    )
    request_id: Mapped[str] = mapped_column(String(128))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(16), default="agent")
    status: Mapped[str] = mapped_column(String(24), default="running")
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    execution_owner_id: Mapped[str | None] = mapped_column(String(64))
    execution_epoch: Mapped[int] = mapped_column(Integer, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    question_message_id: Mapped[str] = mapped_column(String(64))
    answer_message_id: Mapped[str] = mapped_column(String(64))
    history_cutoff_seq: Mapped[int] = mapped_column(Integer, default=0)
    summary_version: Mapped[int] = mapped_column(Integer, default=0)
    memory_prepared: Mapped[bool] = mapped_column(Boolean, default=False)
    knowledge_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    knowledge_prepared: Mapped[bool] = mapped_column(Boolean, default=False)
    model_name: Mapped[str] = mapped_column(String(128), default="")
    prompt_version: Mapped[str] = mapped_column(String(64), default="v1")
    system_prompt_snapshot: Mapped[str] = mapped_column(BODY, default="")
    tool_schema_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    understanding_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversation_understandings.understanding_id", name="fk_run_understanding"
        ),
        unique=True,
    )
    orchestration_version: Mapped[str] = mapped_column(String(24), default="legacy-v1")
    routing_input: Mapped[dict] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict | None] = mapped_column(ExactJSON(only_reply_v2=True))
    business_config: Mapped[dict | None] = mapped_column(JSON)
    knowledge_config: Mapped[dict | None] = mapped_column(ExactJSON)
    refund_config: Mapped[dict | None] = mapped_column(ExactJSON)
    model_audits: Mapped[list] = mapped_column(JSON, default=list)
    state_schema_version: Mapped[int] = mapped_column(Integer, default=1)
    count_config: Mapped[dict] = mapped_column(JSON, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(64))
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, onupdate=utc_now
    )


class ConversationMessage(ConversationBase):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uq_message_seq"),
        UniqueConstraint("run_id", "role", name="uq_run_message_role"),
        CheckConstraint("role IN ('user','assistant')", name="ck_message_role"),
        OPTIONS,
    )
    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id")
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("conversation_runs.run_id"))
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(BODY)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class ConversationSummaryRecord(ConversationBase):
    __tablename__ = "conversation_summaries"
    __table_args__ = (OPTIONS,)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[dict] = mapped_column(JSON)
    covered_until_seq: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    source_run_id: Mapped[str] = mapped_column(ForeignKey("conversation_runs.run_id"))


class ConversationAction(ConversationBase):
    __tablename__ = "conversation_actions"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "model_step", "tool_call_id", name="uq_action_model_call"
        ),
        CheckConstraint("effect IN ('read','write')", name="ck_action_effect"),
        CheckConstraint(
            "approval_status IN ('not_required','pending','approved','rejected')",
            name="ck_action_approval",
        ),
        CheckConstraint(
            "execution_status IN ('pending','completed')", name="ck_action_execution"
        ),
        OPTIONS,
    )
    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("conversation_runs.run_id"))
    model_step: Mapped[int] = mapped_column(Integer)
    tool_call_id: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(128))
    effect: Mapped[str] = mapped_column(String(16))
    arguments: Mapped[dict] = mapped_column(JSON)
    arguments_fingerprint: Mapped[str] = mapped_column(String(64))
    approval_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    approval_status: Mapped[str] = mapped_column(String(24), default="not_required")
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    execution_status: Mapped[str] = mapped_column(String(24), default="pending")
    result: Mapped[dict | None] = mapped_column(JSON)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
