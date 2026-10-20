"""Independent understanding contracts. No tools, permissions or chat state."""

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_serializer,
    model_validator,
)

from resolv.schemas.conversation import (
    CompletedTurn,
    ConversationRecord,
    SummaryRecord,
)

Intent = Literal[
    "logistics",
    "order",
    "product_consultation",
    "refund_return",
    "after_sales",
    "complaint",
    "human",
    "chitchat",
    "other",
]
Branch = Literal["business", "knowledge", "refund_flow", "escalate", "fallback_script"]
SourceKind = Literal["input", "message", "action", "summary", "understanding", "task"]
UnderstandingStatus = Literal[
    "created",
    "running",
    "finalizing",
    "ready",
    "needs_clarification",
    "interrupted",
    "failed",
]


class UnderstandingError(RuntimeError):
    def __init__(self, code, retryable=False, understanding_id=None, related_id=None):
        self.code, self.retryable = code, retryable
        self.understanding_id, self.related_id = understanding_id, related_id
        super().__init__(code)


class StrictRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


class UnderstandingInput(StrictRecord):
    request_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    clarifies_understanding_id: str | None = Field(
        default=None, min_length=1, max_length=64
    )
    continue_task_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("request_id", "text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("nonblank value required")
        return value

    @model_validator(mode="after")
    def exclusive_links(self):
        if self.clarifies_understanding_id and self.continue_task_id:
            raise ValueError("understanding links are mutually exclusive")
        return self


class TextSpan(StrictRecord):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def nonempty(self):
        if self.end <= self.start:
            raise ValueError("nonempty Python character range required")
        return self


class EvidenceRef(StrictRecord):
    kind: SourceKind
    source_id: str = Field(min_length=1)
    span: TextSpan | None
    json_path: list[str | int]
    quote: str = Field(min_length=1)


class EvidenceSource(StrictRecord):
    kind: SourceKind
    source_id: str
    text: str | None
    json_value: JsonValue
    observed_at: datetime | None
    source_hash: str


class SourceFingerprint(StrictRecord):
    kind: SourceKind
    source_id: str
    source_hash: str


class EntityBinding(StrictRecord):
    span: TextSpan
    entity_type: Literal["order", "shipment", "refund", "product", "other"]
    value: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1)


class UnresolvedReference(StrictRecord):
    span: TextSpan
    candidate_refs: list[EvidenceRef]
    question: str = Field(min_length=1)


class ResolutionResult(StrictRecord):
    resolved_text: str = Field(min_length=1)
    bindings: list[EntityBinding]
    unresolved_references: list[UnresolvedReference]
    continuation_selection: str | None


class Ambiguity(StrictRecord):
    kind: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef]


class ClassifiedRequest(StrictRecord):
    text: str = Field(min_length=1)
    intent: Intent | None
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    binding_indexes: list[int]
    explicit_priority: int | None = Field(ge=0)
    priority_evidence_refs: list[EvidenceRef]
    ambiguities: list[Ambiguity]


class ClassificationResult(StrictRecord):
    requests: list[ClassifiedRequest] = Field(min_length=1)


class FrozenConfig(StrictRecord):
    model_name: str
    model_api_mode: Literal["responses", "chat_completions"] = "responses"
    output_format: Literal["json_object", "json_schema"] = "json_object"
    enable_thinking: bool | None = None
    thinking_budget: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    schema_constraint_version: (
        Literal[
            "source-catalog-v1",
            "source-catalog-v2",
            "source-catalog-v3",
            "source-catalog-v4",
            "source-catalog-v5",
            "source-catalog-v6",
        ]
        | None
    ) = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    api_address_fingerprint: str
    tokenizer_profile: Literal["o200k_base", "cl100k_base"]
    model_context_window_tokens: int = Field(gt=0)
    app_max_input_tokens: int = Field(gt=0)
    provider_input_overhead_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(gt=0)
    safety_margin_tokens: int = Field(ge=0)
    recent_history_target_tokens: int = Field(ge=0)
    resolution_prompt: str
    intent_prompt: str
    resolution_version: str
    intent_version: str
    resolution_schema: dict[str, JsonValue]
    intent_schema: dict[str, JsonValue]
    mapping_version: Literal["intent-branches-v1"]
    state_schema_version: Literal[1]
    snapshot_version: Literal[1]

    @model_serializer(mode="wrap")
    def preserve_omitted_defaults(self, handler):
        # Optional profile fields added later must not change an older record's
        # wire representation or snapshot hash. New contexts explicitly set them.
        value = handler(self)
        return {
            key: item
            for key, item in value.items()
            if key in self.model_fields_set
            or type(self).model_fields[key].is_required()
        }

    @property
    def input_limit(self):
        return min(
            self.app_max_input_tokens,
            self.model_context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens,
        )


class StageAudit(StrictRecord):
    stage: Literal["resolution", "intents"]
    invocation_id: str
    attempt: int
    input_estimate: int
    input_tokens: int | None
    output_tokens: int | None
    started_at: datetime
    finished_at: datetime | None
    outcome: Literal["started", "succeeded", "failed", "unknown"]
    verified_refs: list[SourceFingerprint]


class ResolutionStageResult(StrictRecord):
    output: ResolutionResult
    audit: StageAudit


class IntentStageResult(StrictRecord):
    output: ClassificationResult
    audit: StageAudit


class RouteSelection(StrictRecord):
    primary_index: int
    secondary_indexes: list[int]
    branch: Branch | None
    route_ready: bool
    clarification_question: str | None
    continuation_task_id: str | None


class UnderstandingRequestRecord(StrictRecord):
    task_id: str
    understanding_id: str
    conversation_id: str
    ordinal: int
    role: Literal["primary", "secondary"]
    text: str
    intent: Intent | None
    branch: Branch | None
    evidence_refs: list[EvidenceRef]
    bindings: list[EntityBinding]
    ambiguities: list[Ambiguity]
    unresolved_references: list[UnresolvedReference]
    queue_status: Literal["not_queued", "pending", "selected"]
    selected_by_understanding_id: str | None
    created_at: datetime


class ContextSnapshot(StrictRecord):
    conversation_id: str
    owner_id: str
    original_text: str
    linked_inputs: dict[str, str]
    cutoff_seq: int
    summary_version: int
    summary: SummaryRecord | None
    model_sources: list[EvidenceSource]
    source_fingerprints: list[SourceFingerprint]
    omitted_ranges: list[TextSpan]
    pending_requests: list[UnderstandingRequestRecord]
    parent: dict[str, JsonValue] | None
    config: FrozenConfig
    snapshot_hash: str


class SnapshotSources(StrictRecord):
    conversation: ConversationRecord
    turns: list[CompletedTurn]
    summary: SummaryRecord | None
    pending_requests: list[UnderstandingRequestRecord]
    parent: dict[str, JsonValue] | None
    linked_inputs: dict[str, str] = Field(default_factory=dict)


class UnderstandingRecord(StrictRecord):
    understanding_id: str
    conversation_id: str
    request_id: str
    request_fingerprint: str
    original_text: str
    clarifies_understanding_id: str | None
    continue_task_id: str | None
    clarification_child_id: str | None
    status: UnderstandingStatus
    snapshot: ContextSnapshot
    resolution: ResolutionStageResult | None
    classification: IntentStageResult | None
    final_json: dict[str, JsonValue] | None
    execution_owner_id: str | None
    execution_epoch: int
    attempt: int
    invocations: list[StageAudit]
    error_code: str | None
    retryable: bool
    created_at: datetime
    updated_at: datetime


class UnderstandingView(StrictRecord):
    schema_version: Literal[1]
    understanding_id: str
    conversation_id: str
    status: UnderstandingStatus
    original_text: str
    resolution: ResolutionResult | None
    primary_request: UnderstandingRequestRecord | None
    branch: Branch | None
    route_ready: bool
    pending_requests: list[UnderstandingRequestRecord]
    clarification_question: str | None
    context_refs: list[SourceFingerprint]
    versions: dict[str, str | int]
    stage_results: dict[str, JsonValue]
    followup_ids: dict[str, str | None]
    error_code: str | None
    retryable: bool
