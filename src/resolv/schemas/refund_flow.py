"""Frozen refund preparation, reasoning and audit contracts."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from resolv.schemas.hybrid_knowledge import Hash64, HybridRetrievalProfile
from resolv.schemas.knowledge_branch import (
    FrozenJSON,
    KnowledgeEvidenceBundle,
    KnowledgeRetrievalStage,
)
from resolv.schemas.knowledge_release import ReleaseSelection
from resolv.schemas.refund import (
    AdapterDescriptor,
    OrderBinding,
    RefundContract,
    RefundObservation,
    RefundTimestamp,
)
from resolv.schemas.understanding import EvidenceRef
from resolv.services.hybrid_fingerprints import canonical_hash

Operation = Literal["eligibility", "apply", "status"]


class RefundRequest(RefundContract):
    operation: Operation
    reason: str | None = Field(default=None, min_length=1, max_length=500)
    reason_refs: tuple[EvidenceRef, ...] = ()
    user_claims: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reason_source(self):
        if self.reason is not None and (
            not self.reason.strip() or not self.reason_refs
        ):
            raise ValueError("Refund reason requires a nonblank sourced value")
        return self


class RefundScopeReply(RefundContract):
    kind: Literal["clarification", "human_draft", "unavailable"]
    reason: str = Field(min_length=1)


class RefundScopeResult(RefundContract):
    request: RefundRequest | None = None
    scope_reply: RefundScopeReply | None = None

    @model_validator(mode="after")
    def exclusive(self):
        if (self.request is None) == (self.scope_reply is None):
            raise ValueError("Exactly one scope result required")
        return self


class RefundBindingStage(RefundScopeResult):
    binding: OrderBinding | None = None

    @model_validator(mode="after")
    def executable_binding(self):
        if self.request is not None and self.binding is None:
            raise ValueError("Executable request requires a sourced order binding")
        return self


class RefundQueryOutput(RefundContract):
    standard: str = Field(min_length=1)
    expanded: tuple[Annotated[str, Field(max_length=32)], ...] = Field(max_length=8)


class PreparedRefundQuery(RefundQueryOutput):
    bm25_text: str
    status: Literal["prepared", "fallback"]
    fallback_reason: str | None = None
    input_hash: Hash64

    @model_validator(mode="after")
    def query_shape(self):
        if not self.standard.strip() or len(set(self.expanded)) != len(self.expanded):
            raise ValueError("Invalid query")
        if any(not s or s != s.strip() or s == self.standard for s in self.expanded):
            raise ValueError("Invalid expansion")
        if self.bm25_text != "\n".join((self.standard, *self.expanded)):
            raise ValueError("BM25 text mismatch")
        if (self.status == "fallback") != bool(self.fallback_reason):
            raise ValueError("Fallback reason mismatch")
        return self


RefundRetrievalStage = KnowledgeRetrievalStage


class RefundPolicyCitation(RefundContract):
    citation_number: int = Field(ge=1)
    window_id: Hash64
    quote: str = Field(min_length=1)


class RefundFactRef(RefundContract):
    observation_id: str
    field: str


class RefundRecheckCondition(RefundContract):
    """A policy-defined comparison; vague policy prose cannot create a deadline."""

    fact_ref: RefundFactRef
    policy_window_id: Hash64
    policy_quote: str = Field(min_length=1)
    operator: Literal["equals", "elapsed_seconds_lte", "calendar_days_lte"]
    expected: str | int | bool


class RefundRuleSupport(RefundContract):
    rule: str = Field(min_length=1)
    policy_citations: tuple[RefundPolicyCitation, ...] = Field(min_length=1)
    fact_refs: tuple[RefundFactRef, ...] = Field(min_length=1)


class RefundDecision(RefundContract):
    operation: Operation
    verdict: Literal["eligible", "ineligible", "undetermined"] | None
    order_binding_hash: Hash64
    evidence_hash: Hash64
    observation_refs: tuple[str, ...] = ()
    policy_citations: tuple[RefundPolicyCitation, ...] = ()
    rules: tuple[RefundRuleSupport, ...] = ()
    reason: str = Field(min_length=1, max_length=2000)
    missing_information: tuple[str, ...] = ()
    unverified_conditions: tuple[str, ...] = ()
    critical_fields: tuple[RefundFactRef, ...] = ()
    recheck_conditions: tuple[RefundRecheckCondition, ...] = ()
    next_action: Literal[
        "show_offer", "show_status", "request_information", "human_draft", "none"
    ]

    @model_validator(mode="after")
    def semantics(self):
        if (self.operation == "status") != (self.verdict is None):
            raise ValueError("Status must not assert eligibility")
        if self.verdict in {"eligible", "ineligible"} and (
            not self.rules or not self.policy_citations or not self.observation_refs
        ):
            raise ValueError("Eligibility verdict requires sourced policy and facts")
        if self.verdict in {"eligible", "ineligible"} and (
            self.missing_information or self.unverified_conditions
        ):
            raise ValueError("Unresolved conditions forbid eligibility verdict")
        if self.next_action == "show_offer" and self.verdict != "eligible":
            raise ValueError("Only eligible decisions can offer an application")
        return self


ValidatedRefundDecision = RefundDecision


class RefundDecisionOutput(RefundDecision):
    """Complete wire output for fresh runs; persisted decisions keep old defaults."""

    observation_refs: tuple[str, ...] = Field(...)
    policy_citations: tuple[RefundPolicyCitation, ...] = Field(...)
    rules: tuple[RefundRuleSupport, ...] = Field(...)
    missing_information: tuple[str, ...] = Field(...)
    unverified_conditions: tuple[str, ...] = Field(...)
    critical_fields: tuple[RefundFactRef, ...] = Field(...)
    recheck_conditions: tuple[RefundRecheckCondition, ...] = Field(...)


class RefundModelProfile(RefundContract):
    purpose: Literal["refund_query", "refund_decision", "refund_summary"]
    model_name: str
    model_api_mode: Literal["chat_completions", "responses"]
    api_address_fingerprint: Hash64
    tokenizer_profile: Literal["o200k_base", "cl100k_base"] = "o200k_base"
    model_context_window_tokens: int = Field(default=256000, gt=0)
    app_max_input_tokens: int = Field(default=12000, gt=0)
    provider_input_overhead_tokens: int = Field(default=6500, ge=0)
    reserved_output_tokens: int = Field(default=2000, gt=0)
    safety_margin_tokens: int = Field(default=1000, ge=0)
    recent_history_target_tokens: int = Field(default=2000, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: Literal[0] = 0
    enable_thinking: bool | None = False
    thinking_budget: int | None = Field(default=None, gt=0)
    top_p: float | None = 0.1
    reasoning_effort: str | None = None
    system_prompt: str = Field(min_length=1)
    prompt_version: str
    tool_schema_fingerprint: Hash64
    output_schema: FrozenJSON | None = None

    @property
    def input_limit(self):
        return min(
            self.app_max_input_tokens,
            self.model_context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens,
        )

    @model_validator(mode="after")
    def budget(self):
        if self.input_limit <= 0:
            raise ValueError("Invalid model budget")
        return self


class RefundProfile(RefundContract):
    query_model: RefundModelProfile
    decision_model: RefundModelProfile
    summary_profile: FrozenJSON
    adapter: AdapterDescriptor
    selection: ReleaseSelection = ReleaseSelection(status="disabled")
    retrieval_profile: HybridRetrievalProfile = HybridRetrievalProfile()
    evidence_max_tokens: int = Field(default=3000, gt=0)
    max_agent_steps: int = Field(default=6, ge=1, le=6)
    offer_ttl_seconds: int = Field(default=1800, gt=0)
    business_timezone: str = "Asia/Shanghai"
    config_version: Literal[1, 2, 3, 4, 5, 6, 7] = 1

    def final_decision_round(self, model_step):
        # Methods do not add serialized fields to historical frozen profiles.
        return self.config_version >= 3 and model_step == self.max_agent_steps

    def required_read_tool(self, observations):
        if self.config_version < 7:
            return None
        observed = {o.tool_name for o in observations}
        return next(
            (
                name
                for name in ("query_refund_order", "query_refund_status")
                if name not in observed
            ),
            None,
        )

    def context_requires_tool(self, model_step, observations):
        if self.final_decision_round(model_step):
            return False
        if self.config_version < 4:
            return True  # Historical context estimators reserved required-tool params.
        return not {"query_refund_order", "query_refund_status"} <= {
            o.tool_name for o in observations
        }

    @model_validator(mode="after")
    def model_purposes(self):
        if (
            (
                self.config_version == 4
                and self.decision_model.prompt_version
                != "refund-decision-v3-json-schema"
            )
            or (
                self.config_version == 5
                and self.decision_model.prompt_version
                != "refund-decision-v4-complete-json-schema"
            )
            or (
                self.config_version == 6
                and self.decision_model.prompt_version
                != "refund-decision-v5-context-identity"
            )
            or (
                self.config_version == 7
                and self.decision_model.prompt_version
                != "refund-decision-v6-native-return"
            )
        ):
            raise ValueError("Refund output protocol conflicts with config version")
        if self.evidence_max_tokens != (3000 if self.config_version == 1 else 15000):
            raise ValueError("Refund evidence budget conflicts with config version")
        if (
            self.query_model.purpose != "refund_query"
            or self.decision_model.purpose != "refund_decision"
        ):
            raise ValueError("Refund model purpose mismatch")
        return self


class RefundModelAudit(RefundContract):
    invocation_id: str
    purpose: Literal["refund_query", "refund_decision", "refund_summary"]
    profile_fingerprint: Hash64
    input_hash: Hash64
    model_step: int = Field(default=0, ge=0, le=6)
    attempt: int = Field(default=1, ge=1)
    input_estimate: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    started_at: RefundTimestamp
    finished_at: RefundTimestamp | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    outcome: Literal["started", "succeeded", "failed", "unknown"]
    error_code: str | None = None


class RefundReadAudit(RefundContract):
    invocation_id: str
    model_step: int = Field(ge=1, le=6)
    tool_call_id: str
    tool_name: Literal[
        "query_refund_order",
        "query_refund_payment",
        "query_refund_delivery",
        "query_refund_status",
    ]
    binding_hash: Hash64
    profile_fingerprint: Hash64
    input_hash: Hash64
    attempt: int = Field(ge=1)
    started_at: RefundTimestamp
    finished_at: RefundTimestamp | None = None
    outcome: Literal["started", "succeeded", "failed", "unknown"]
    observation_id: str | None = None
    error_code: str | None = None


class RefundToolCall(RefundContract):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1)
    args: FrozenJSON


class RefundModelInput(RefundContract):
    messages: FrozenJSON
    input_estimate: int = Field(ge=0)
    input_hash: Hash64


class RefundModelResponse(RefundContract):
    model_step: int = Field(ge=1, le=6)
    tool_calls: tuple[RefundToolCall, ...] = ()
    decision: RefundDecision | None = None
    input_hash: Hash64
    output_hash: Hash64
    audit: RefundModelAudit

    @model_validator(mode="after")
    def one_result(self):
        if bool(self.tool_calls) == (self.decision is not None):
            raise ValueError("Exactly one refund model result required")
        if len({c.id for c in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("Duplicate tool call ID")
        if (
            self.audit.purpose != "refund_decision"
            or self.audit.model_step != self.model_step
            or self.audit.outcome != "succeeded"
            or self.audit.input_hash != self.input_hash
        ):
            raise ValueError("Refund response audit mismatch")
        value = {
            "tool_calls": [c.model_dump(mode="json") for c in self.tool_calls],
            "decision": self.decision.model_dump(mode="json")
            if self.decision
            else None,
        }
        if self.output_hash != canonical_hash(value):
            raise ValueError("Refund response output hash mismatch")
        return self


class RefundRunRecord(RefundContract):
    run_id: str
    owner_id: str
    conversation_id: str
    understanding_id: str
    primary_task_id: str
    primary_hash: Hash64
    primary_json: FrozenJSON
    profile: RefundProfile
    binding: OrderBinding | None = None
    request: RefundRequest | None = None
    scope_reply: RefundScopeReply | None = None
    query: PreparedRefundQuery | None = None
    retrieval: KnowledgeRetrievalStage | None = None
    decision: RefundDecision | None = None
    model_requests: tuple[RefundModelResponse, ...] = ()
    observations: tuple[RefundObservation, ...] = ()
    audits: tuple[RefundModelAudit, ...] = ()
    read_audits: tuple[RefundReadAudit, ...] = ()
    stage_hashes: tuple[tuple[str, Hash64, Hash64], ...] = ()
    error_code: str | None = None
    retryable: bool = False
    step_limit: bool = False

    @property
    def evidence(self) -> KnowledgeEvidenceBundle | None:
        return self.retrieval.bundle if self.retrieval else None
