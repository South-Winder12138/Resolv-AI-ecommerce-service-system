"""Immutable offer, command, business receipt and public projection contracts."""

from typing import Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from resolv.schemas.hybrid_knowledge import Hash64
from resolv.schemas.knowledge_branch import (
    FrozenJSON,
    KnowledgeEvidenceBundle,
    ReplyMetadataV2,
    canonical_json,
)
from resolv.schemas.knowledge_release import ReleaseSelection
from resolv.schemas.refund import (
    OrderBinding,
    RefundContract,
    RefundFieldValue,
    RefundMoney,
    RefundPublicMoney,
    RefundTimestamp,
)
from resolv.schemas.refund_flow import (
    Operation,
    RefundPolicyCitation,
    RefundProfile,
    RefundRecheckCondition,
)
from resolv.services.hybrid_fingerprints import canonical_hash


class RefundSubmitInput(RefundContract):
    request_id: str = Field(min_length=1, max_length=128)
    confirm: Literal[True]

    @field_validator("confirm", mode="before")
    @classmethod
    def boolean_true(cls, value):
        if value is not True:
            raise ValueError("Explicit boolean confirmation required")
        return value

    @field_validator("request_id")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Nonblank request ID required")
        return value


class RefundConditionResult(RefundContract):
    passed: bool
    reasons: tuple[str, ...] = ()


class RefundOfferRecord(RefundContract):
    offer_id: str
    owner_id: str
    conversation_id: str
    run_id: str
    primary_task_id: str
    binding: OrderBinding
    subject_ref: str
    money: RefundMoney
    reason: str = Field(min_length=1, max_length=500)
    selection: ReleaseSelection
    evidence_hash: Hash64
    decision_hash: Hash64
    critical_facts: FrozenJSON
    conditions: tuple[RefundRecheckCondition, ...] = ()
    profile: RefundProfile
    created_at: RefundTimestamp
    expires_at: RefundTimestamp
    status: Literal["available", "needs_reassessment", "submitted"] = "available"
    invalid_reason: str | None = None

    @model_validator(mode="after")
    def usable_content(self):
        if (
            not self.reason.strip()
            or self.money.minor_units <= 0
            or self.expires_at <= self.created_at
        ):
            raise ValueError("Offer requires positive money, reason and valid lifetime")
        return self


class RefundRegistrationCommand(RefundContract):
    command_id: str
    offer_id: str
    request_hash: Hash64
    binding: OrderBinding
    subject_ref: str
    money: RefundMoney
    reason: str = Field(min_length=1, max_length=500)
    decision_hash: Hash64
    evidence_hash: Hash64
    revalidation_hash: Hash64
    snapshots: FrozenJSON
    created_at: RefundTimestamp

    @model_validator(mode="after")
    def application_content(self):
        if not self.reason.strip() or self.money.minor_units <= 0:
            raise ValueError(
                "Application needs a reason and positive refundable amount"
            )
        return self


class RefundReceipt(RefundContract):
    application_id: str
    order_source: str
    order_id: str
    status: Literal["pending", "approved", "rejected", "completed"]
    created: bool
    money: RefundPublicMoney | None
    reason: str
    created_at: RefundTimestamp


class SubmissionLease(RefundContract):
    command_id: str
    owner_id: str
    worker_id: str
    epoch: int = Field(ge=1)


class RefundRevalidation(RefundContract):
    outcome: Literal["passed", "needs_reassessment", "unavailable"]
    observations: FrozenJSON
    differences: tuple[str, ...] = ()
    checked_at: RefundTimestamp
    revalidation_hash: Hash64

    @classmethod
    def create(cls, *, outcome, observations, differences, checked_at):
        clock = TypeAdapter(RefundTimestamp).validate_python(checked_at)
        data = {
            "outcome": outcome,
            "observations": FrozenJSON.from_value(observations).model_dump(mode="json"),
            "differences": list(differences),
            "checked_at": TypeAdapter(RefundTimestamp).dump_python(clock, mode="json"),
        }
        return cls.model_validate_json(
            canonical_json(data | {"revalidation_hash": canonical_hash(data)})
        )

    @model_validator(mode="after")
    def immutable_recheck(self):
        if (
            canonical_hash(self.model_dump(mode="json", exclude={"revalidation_hash"}))
            != self.revalidation_hash
        ):
            raise ValueError("Revalidation hash mismatch")
        return self


class RefundCriticalFact(RefundContract):
    tool_name: Literal[
        "query_refund_order",
        "query_refund_payment",
        "query_refund_delivery",
        "query_refund_status",
    ]
    field: str = Field(min_length=1)
    observation_id: str
    fact: RefundFieldValue


class RefundCriticalSnapshot(RefundContract):
    binding_hash: Hash64
    subject_ref: str = Field(min_length=1)
    fields: tuple[RefundCriticalFact, ...]

    @model_validator(mode="after")
    def unique_refs(self):
        refs = [(f.tool_name, f.field, f.observation_id) for f in self.fields]
        if len(set(refs)) != len(refs):
            raise ValueError("Duplicate critical fact reference")
        return self


class RefundRevalidationAudit(RefundContract):
    attempt_id: str
    outcome: Literal["started", "succeeded", "failed", "unknown"]
    at: RefundTimestamp
    result_hash: Hash64 | None = None
    error_code: str | None = None


class RefundSubmissionRecord(RefundContract):
    submission_id: str
    offer_id: str
    owner_id: str
    conversation_id: str
    request_hash: Hash64
    status: Literal["created", "running", "interrupted", "completed", "failed"] = (
        "created"
    )
    execution_epoch: int = 0
    execution_owner_id: str | None = None
    lease_expires_at: RefundTimestamp | None = None
    revalidations: tuple[RefundRevalidation, ...] = ()
    revalidation_audits: tuple[RefundRevalidationAudit, ...] = ()
    receipt: RefundReceipt | None = None
    error_code: str | None = None
    retryable: bool = False
    created_at: RefundTimestamp
    updated_at: RefundTimestamp


class RefundSubmissionView(RefundContract):
    submission_id: str
    status: Literal["created", "running", "interrupted", "completed", "failed"]
    receipt: RefundReceipt | None = None
    error_code: str | None = None
    retryable: bool = False


class RefundOfferView(RefundContract):
    offer_id: str
    order_id: str
    money: RefundPublicMoney
    reason: str
    created_at: RefundTimestamp
    expires_at: RefundTimestamp
    validity: Literal[
        "available", "expired", "needs_reassessment", "submitted", "unavailable"
    ]
    submission: RefundSubmissionView | None = None


class RefundPublicResult(RefundContract):
    operation: Operation | None = None
    verdict: Literal["eligible", "ineligible", "undetermined"] | None = None
    order_id: str | None = None
    reason: str = Field(min_length=1)
    missing_information: tuple[str, ...] = ()
    unverified_conditions: tuple[str, ...] = ()
    next_action: Literal[
        "show_offer", "show_status", "request_information", "human_draft", "none"
    ] = "none"
    release_id: str | None = None
    index_version: str | None = None
    evidence: KnowledgeEvidenceBundle | None = None
    policy_citations: tuple[RefundPolicyCitation, ...] = ()
    offer_id: str | None = None
    application: RefundReceipt | None = None
    status_scope: Literal["local_only"] = "local_only"

    @model_validator(mode="after")
    def public_proof(self):
        if self.operation == "status" and self.verdict is not None:
            raise ValueError("Status cannot assert eligibility")
        if (self.release_id is None) != (self.index_version is None):
            raise ValueError("Incomplete public release identity")
        if self.evidence is None and (
            self.policy_citations or self.release_id is not None
        ):
            raise ValueError("Missing public evidence")
        if self.evidence is not None:
            if self.evidence.windows and self.release_id is None:
                raise ValueError("Policy windows require a real release")
            bodies = {b.chunk.chunk_id: b.chunk for b in self.evidence.chunks}
            for reference in self.policy_citations:
                actual = next(
                    (
                        c
                        for c in self.evidence.citations
                        if c.citation_number == reference.citation_number
                    ),
                    None,
                )
                if (
                    actual is None
                    or actual.window_id != reference.window_id
                    or reference.quote
                    not in "\n".join(bodies[c].text for c in actual.chunk_ids)
                ):
                    raise ValueError("Invalid public policy reference")
        if self.verdict in {"eligible", "ineligible"} and (
            self.evidence is None or not self.policy_citations
        ):
            raise ValueError("Eligibility requires actual public policy proof")
        if self.offer_id is not None and (
            self.verdict != "eligible"
            or self.next_action != "show_offer"
            or self.application is not None
        ):
            raise ValueError("Invalid public offer reference")
        if self.next_action == "show_offer" and self.offer_id is None:
            raise ValueError("Missing public offer reference")
        return self


class ReplyMetadataV3(ReplyMetadataV2):
    schema_version: Literal[3] = 3
    refund: RefundPublicResult | None = None

    @model_validator(mode="after")
    def refund_branch(self):
        if self.refund is not None and self.branch != "refund_flow":
            raise ValueError("Refund result belongs to the refund branch")
        if self.branch == "refund_flow" and self.knowledge is not None:
            raise ValueError("Refund has independent evidence")
        return self


class BranchReplyV3(RefundContract):
    answer: str = Field(min_length=1)
    metadata: ReplyMetadataV3


class MigrationReceipt(RefundContract):
    path: str
    backup_path: str
    backup_hash: Hash64
    before_hash: Hash64
    after_hash: Hash64
    legacy_count: int = Field(ge=0)
    application_count: int = Field(ge=0)
    schema_version: Literal[1] = 1
