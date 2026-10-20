"""Strict refund identities, authoritative observations and exact money."""

import json
import re
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BeforeValidator,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from resolv.schemas.hybrid_knowledge import Hash64
from resolv.schemas.knowledge_release import KnowledgeContract
from resolv.schemas.understanding import EvidenceRef
from resolv.services.hybrid_fingerprints import canonical_hash

Availability = Literal[
    "available", "unknown", "not_supported", "not_connected", "not_found", "forbidden"
]
ApplicationStatus = Literal["pending", "approved", "rejected", "completed"]


class RefundFlowError(RuntimeError):
    def __init__(self, code, retryable=False, run_id=None, submission_id=None):
        self.code, self.retryable = code, retryable
        self.run_id, self.submission_id = run_id, submission_id
        super().__init__(code)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _aware_refund_time(value):
    if isinstance(value, str):
        if not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})",
            value,
        ):
            raise ValueError("RFC3339 aware datetime required")
        value = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        )
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("RFC3339 aware datetime required")
    return value


RefundTimestamp = Annotated[
    datetime,
    BeforeValidator(_aware_refund_time),
    AfterValidator(lambda value: value.astimezone(UTC)),
]


class RefundContract(KnowledgeContract):
    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        # Validate raw JSON before Pydantic can discard duplicate object members.
        json.loads(
            json_data,
            object_pairs_hook=unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
        return super().model_validate_json(json_data, **kwargs)


class RefundPublicMoney(RefundContract):
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    minor_units_decimal: str = Field(pattern=r"^(0|[1-9][0-9]*)$")
    unit_precision: int = Field(ge=0, le=9)


class RefundMoney(RefundContract):
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    minor_units: int = Field(ge=0, le=2**63 - 1)
    unit_precision: int = Field(ge=0, le=9)

    def public(self) -> RefundPublicMoney:
        return RefundPublicMoney(
            currency=self.currency,
            minor_units_decimal=str(self.minor_units),
            unit_precision=self.unit_precision,
        )


class OrderBinding(RefundContract):
    order_source: str = Field(min_length=1, max_length=128)
    order_id: str = Field(min_length=1, max_length=128)
    understanding_id: str = Field(min_length=1, max_length=64)
    primary_task_id: str = Field(min_length=1, max_length=64)
    primary_hash: Hash64
    source_refs: tuple[EvidenceRef, ...] = Field(min_length=1)
    binding_hash: Hash64

    @classmethod
    def create(cls, **data):
        # Canonicalize nested source records before hashing; never hash repr().
        payload = {
            k: [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in v]
            if k == "source_refs"
            else v
            for k, v in data.items()
        }
        return cls.model_validate_json(
            json.dumps(payload | {"binding_hash": canonical_hash(payload)})
        )

    @model_validator(mode="after")
    def identity(self):
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("Refund binding hash mismatch")
        return self


class AdapterDescriptor(RefundContract):
    namespace: str = Field(min_length=1, max_length=128)
    implementation_version: str = Field(min_length=1)
    authorization_version: str = Field(min_length=1)
    connected: bool = False


class RefundAccessContext(RefundContract):
    """Server-owned identity capability; never a browser or model argument."""

    subject_ref: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    authorization_version: str = Field(min_length=1)
    trusted: Literal[True]

    @field_validator("trusted", mode="before")
    @classmethod
    def actual_true(cls, value):
        if value is not True:
            raise ValueError("Trusted identity required")
        return value


class RefundFieldValue(RefundContract):
    availability: Availability
    applicability: Literal["applicable", "not_applicable", "unknown"] = "applicable"
    value: JsonValue = None
    original_time: str | None = None

    @model_validator(mode="after")
    def available_value(self):
        if self.availability != "available" and self.value is not None:
            raise ValueError("Unavailable field cannot carry a verified value")
        if self.applicability != "applicable" and self.value is not None:
            raise ValueError("Unknown/inapplicable field cannot carry a value")
        return self


class RefundObservation(RefundContract):
    observation_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1, max_length=128)
    tool_name: Literal[
        "query_refund_order",
        "query_refund_payment",
        "query_refund_delivery",
        "query_refund_status",
    ]
    binding_hash: Hash64
    source: str = Field(min_length=1)
    observed_at: RefundTimestamp
    original_observed_at: str
    provider_version: str | None
    arguments_hash: Hash64
    availability: Availability
    fields: dict[str, RefundFieldValue]
    error_code: str | None = None
    audit_hash: Hash64

    @classmethod
    def create(cls, **data):
        time = data.get("observed_at")
        time = TypeAdapter(AwareDatetime).validate_python(time, strict=True)
        payload = data | {
            "observed_at": time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "original_observed_at": time.isoformat(),
        }
        payload["fields"] = {
            k: RefundFieldValue.model_validate(
                v.model_dump() if hasattr(v, "model_dump") else v
            ).model_dump(mode="json")
            for k, v in data.get("fields", {}).items()
        }
        payload.setdefault("error_code", None)
        return cls.model_validate_json(
            json.dumps(payload | {"audit_hash": canonical_hash(payload)})
        )

    @model_validator(mode="after")
    def identity(self):
        if self.audit_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"audit_hash"})
        ):
            raise ValueError("Refund observation hash mismatch")
        if self.availability != "available" and any(
            v.value is not None for v in self.fields.values()
        ):
            raise ValueError("Unavailable observation carries verified fields")
        return self


class RefundApplicationRecord(RefundContract):
    application_id: str
    order_source: str
    order_id: str
    status: ApplicationStatus
    reason: str
    money: RefundMoney | None = None
    subject_ref: str | None = Field(default=None, exclude=True)
    created_at: RefundTimestamp
    legacy_refund_id: int | None = None
    origin: Literal["legacy", "offer"]
