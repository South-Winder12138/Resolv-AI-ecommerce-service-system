"""Versioned routing and business contracts. No credentials or capabilities."""

from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import Field, SerializeAsAny, field_validator, model_validator

from resolv.schemas.logistics import Digest, LogisticsBackendProfile
from resolv.schemas.logistics_execution import BusinessToolPolicy
from resolv.schemas.understanding import (
    Branch,
    StrictRecord,
    UnderstandingRequestRecord,
)


class RoutingInput(StrictRecord):
    clarifies_understanding_id: str | None = Field(
        default=None, min_length=1, max_length=64
    )
    continue_task_id: str | None = Field(default=None, min_length=1, max_length=64)
    reassesses_refund_offer_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def exclusive_links(self):
        if sum(bool(link) for link in (self.clarifies_understanding_id, self.continue_task_id, self.reassesses_refund_offer_id)) > 1:
            raise ValueError("understanding links are mutually exclusive")
        return self


class ActionDraft(StrictRecord):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SuggestedAction(StrictRecord):
    suggestion_id: str = Field(min_length=1)
    type: Literal["transfer_human", "create_ticket"]
    label: str = Field(min_length=1)
    draft: ActionDraft
    preview_only: Literal[True] = True

    @field_validator("preview_only", mode="before")
    @classmethod
    def only_true(cls, value):
        if value is not True:
            raise ValueError("preview_only must be boolean true")
        return value


class ReplyMetadata(StrictRecord):
    schema_version: Literal[1] = 1
    reply_kind: Literal["answer", "clarification", "unavailable", "step_limit"]
    understanding_id: str
    primary_task_id: str | None
    branch: Branch | None
    clarifies_understanding_id: str | None = None
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)


class BranchReply(StrictRecord):
    answer: str = Field(min_length=1)
    metadata: ReplyMetadata


def parse_reply_metadata(value: dict):
    from resolv.schemas.knowledge_branch import ReplyMetadataV2, canonical_json
    from resolv.schemas.refund_submission import ReplyMetadataV3

    version = value.get("schema_version")
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Unknown reply metadata version")
    target = {1: ReplyMetadata, 2: ReplyMetadataV2, 3: ReplyMetadataV3}[version]
    return target.model_validate_json(canonical_json(value))


class PrimaryExecution(StrictRecord):
    conversation_id: str
    understanding_id: str
    snapshot_hash: str
    primary: UnderstandingRequestRecord

    @model_validator(mode="after")
    def ready_primary(self):
        p = self.primary
        if (
            p.role != "primary"
            or p.conversation_id != self.conversation_id
            or p.understanding_id != self.understanding_id
            or p.intent is None
            or p.branch is None
            or p.ambiguities
            or p.unresolved_references
        ):
            raise ValueError("ready primary required")
        return self


class BusinessPolicy(StrictRecord):
    allowed_order_ids: list[str]
    explicit_tracking_nos: list[str]
    cancel_kind: Literal["not_requested", "explicit", "conditional", "ambiguous"]
    cancel_condition: Literal["processing"] | None = None
    policy_version: Literal["business-policy-v1"] = "business-policy-v1"


class BusinessRunConfig(StrictRecord):
    max_agent_steps: int = Field(default=6, ge=1, le=6)
    model_name: str
    model_api_mode: Literal["chat_completions"] = "chat_completions"
    api_address_fingerprint: str
    prompt_version: str = "business-v3"
    decision_policy_version: Literal["auto-v1", "require-observations-v1"] = "auto-v1"
    system_prompt: str
    tool_schema_fingerprint: str
    top_p: float = Field(default=0.1, gt=0, le=1)
    enable_thinking: Literal[False] = False
    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: Literal[0] = 0
    tokenizer_profile: Literal["o200k_base"] = "o200k_base"
    model_context_window_tokens: int = Field(default=256000, gt=0)
    app_max_input_tokens: int = Field(default=12000, gt=0)
    provider_input_overhead_tokens: int = Field(default=6500, ge=0)
    reserved_output_tokens: int = Field(default=2000, gt=0)
    safety_margin_tokens: int = Field(default=1000, ge=0)
    recent_history_target_tokens: int = Field(default=2000, ge=0)
    budget_version: str = "business-budget-v2-qwen20261014"
    config_version: Literal[1] = 1
    summary_profile: dict | None = None

    @property
    def input_limit(self):
        return min(
            self.app_max_input_tokens,
            self.model_context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens,
        )


class BusinessRunConfigV2(BusinessRunConfig):
    config_version: Literal[2] = 2
    prompt_version: Literal["business-logistics-mcp-v1"] = "business-logistics-mcp-v1"
    tool_policy_version: Literal["logistics-phase-tools-v1"] = "logistics-phase-tools-v1"
    logistics_backend: LogisticsBackendProfile


def parse_business_config(value: dict) -> BusinessRunConfig | BusinessRunConfigV2:
    if not isinstance(value, dict) or type(value.get("config_version", 1)) is not int:
        raise ValueError("integer business config version required")
    target = {1: BusinessRunConfig, 2: BusinessRunConfigV2}.get(value.get("config_version", 1))
    if target is None:
        raise ValueError("unknown business config version")
    return target.model_validate(value)


class BusinessAudit(StrictRecord):
    invocation_id: str
    purpose: Literal["business", "summary"] = "business"
    require_tool: bool = False
    model_step: int = Field(ge=1, le=6)
    attempt: int = Field(ge=1)
    model_name: str
    profile_fingerprint: str
    estimate: int = Field(ge=0)
    actual_input: int | None = Field(default=None, ge=0)
    actual_output: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    error_code: str | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0)


class LogisticsBusinessAudit(BusinessAudit):
    audit_version: Literal[2] = 2
    tool_policy: BusinessToolPolicy
    tool_policy_fingerprint: Digest
    returned_tool_names: list[str] = Field(default_factory=list)


class BusinessDecision(StrictRecord):
    message: AIMessage
    text_deltas: tuple[str, ...]
    audit: SerializeAsAny[BusinessAudit]
