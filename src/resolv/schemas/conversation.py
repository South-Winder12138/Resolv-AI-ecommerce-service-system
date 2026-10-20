"""Strict public records. Execution capabilities never enter these JSON DTOs."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationError(RuntimeError):
    def __init__(self, code: str, retryable: bool = False, run_id: str | None = None):
        self.code, self.retryable, self.run_id = code, retryable, run_id
        super().__init__(code)


class RunStatus(StrEnum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    INTERRUPTED = "interrupted"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class Record(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class ConversationSummary(Record):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_goal: str = ""
    confirmed_facts: list[str] = Field(default_factory=list)
    completed_actions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ConversationRecord(Record):
    conversation_id: str
    owner_id: str = "local"
    thread_id: str
    title: str
    next_message_seq: int = 1
    current_run_id: str | None = None
    status: Literal["active", "archived"] = "active"
    created_at: datetime
    updated_at: datetime
    run_status: RunStatus | None = None


class MessageRecord(Record):
    message_id: str
    conversation_id: str
    run_id: str
    seq: int
    role: Literal["user", "assistant"]
    text: str
    created_at: datetime


class RunRecord(Record):
    run_id: str
    conversation_id: str
    request_id: str
    request_fingerprint: str
    mode: Literal["agent", "chat"]
    status: RunStatus
    attempt: int
    execution_owner_id: str | None = None
    execution_epoch: int = 0
    lease_expires_at: datetime | None = None
    question_message_id: str
    answer_message_id: str
    question: str = ""
    history_cutoff_seq: int = 0
    summary_version: int = 0
    memory_prepared: bool = False
    knowledge_snapshot: list[dict] = Field(default_factory=list)
    knowledge_prepared: bool = False
    model_name: str
    prompt_version: str
    system_prompt_snapshot: str
    tool_schema_fingerprint: str
    understanding_id: str | None = None
    orchestration_version: Literal["legacy-v1", "routed-v2", "routed-v3", "routed-v4"] = "legacy-v1"
    routing_input: dict = Field(default_factory=dict)
    result_json: dict | None = None
    business_config: dict | None = None
    knowledge_config: dict | None = Field(default=None, exclude=True)
    refund_config: dict | None = Field(default=None, exclude=True)
    model_audits: list[dict] = Field(default_factory=list)
    state_schema_version: int
    count_config: dict = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    created_at: datetime
    updated_at: datetime


class ActionRecord(Record):
    action_id: str
    run_id: str
    model_step: int
    tool_call_id: str
    tool_name: str
    effect: Literal["read", "write"]
    arguments: dict
    arguments_fingerprint: str
    approval_id: str | None = None
    approval_status: Literal["not_required", "pending", "approved", "rejected"]
    decided_at: datetime | None = None
    execution_status: Literal["pending", "completed"]
    result: dict | None = None
    completed_at: datetime | None = None


class SummaryRecord(Record):
    conversation_id: str
    version: int
    content: ConversationSummary
    covered_until_seq: int
    model_name: str
    created_at: datetime
    source_run_id: str


class CompletedTurn(Record):
    user: MessageRecord
    assistant: MessageRecord
    actions: list[ActionRecord] = Field(default_factory=list)


class SummaryCandidate(Record):
    content: ConversationSummary
    covered_until_seq: int
    previous_version: int
    model_name: str = ""


class ConversationDetail(Record):
    conversation: ConversationRecord
    messages: list[MessageRecord]
    current_run: RunRecord | None = None
    actions: list[ActionRecord] = Field(default_factory=list)
    runs: list[RunRecord] = Field(default_factory=list)
    next_before_seq: int | None = None


def public_reply_metadata(run: RunRecord) -> dict:
    """One projection for JSON, SSE and replay; only persisted results are final."""
    from resolv.schemas.branch_execution import ReplyMetadata

    if run.result_json is not None:
        metadata = ReplyMetadata.model_validate(run.result_json)
        if (
            run.orchestration_version != "routed-v2"
            or metadata.understanding_id != run.understanding_id
        ):
            raise ConversationError("run_state_conflict", run_id=run.run_id)
        return metadata.model_dump(mode="json")
    if run.orchestration_version == "routed-v2":
        raise ConversationError("run_state_conflict", run_id=run.run_id)
    return {
        "schema_version": 1,
        "reply_kind": None,
        "understanding_id": None,
        "primary_task_id": None,
        "branch": None,
        "clarifies_understanding_id": None,
        "suggested_actions": [],
    }
