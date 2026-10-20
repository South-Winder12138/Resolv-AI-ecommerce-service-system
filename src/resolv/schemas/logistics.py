"""Versioned mock logistics facts; no credentials or execution authority."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

TrackingNo = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
RequestID = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Location = Annotated[str, Field(min_length=1, max_length=256)]
ShipmentStatus = Literal[
    "not_shipped", "in_transit", "out_for_delivery", "delivered", "exception", "unknown"
]


def canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class LogisticsFailure(RuntimeError):
    def __init__(self, code: str, retryable: bool = False):
        self.code, self.retryable = code, retryable
        super().__init__(code)


class LogisticsRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("observed_at", "shipped_at", "signed_at", "occurred_at", mode="before", check_fields=False)
    @classmethod
    def explicit_timezone(cls, value):
        # AwareDatetime also accepts numeric JSON strings as Unix timestamps.
        # Check the input representation before that conversion can invent UTC.
        if value is None:
            return value
        try:
            parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
            if not isinstance(parsed, datetime) or parsed.utcoffset() is None:
                raise ValueError("explicit timezone required")
        except (ValueError, TypeError):
            raise ValueError("explicit timezone required") from None
        return parsed

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def exact_version_type(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("simulated", mode="before", check_fields=False)
    @classmethod
    def exact_simulated_type(cls, value):
        if type(value) is not bool:
            raise ValueError("simulated must be a boolean")
        return value


class TrackingEvent(LogisticsRecord):
    event_id: str = Field(min_length=1, max_length=64)
    occurred_at: AwareDatetime
    status: ShipmentStatus
    location: Location | None
    description: str = Field(max_length=512)


class ShipmentRecord(LogisticsRecord):
    status: ShipmentStatus
    location: Location | None
    shipped_at: AwareDatetime | None
    signed_at: AwareDatetime | None
    events: list[TrackingEvent]
    events_truncated: bool

    @model_validator(mode="after")
    def unique_events(self):
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("duplicate event_id")
        return self


class ShipmentSnapshot(ShipmentRecord):
    events: list[TrackingEvent] = Field(max_length=20)

    @model_validator(mode="after")
    def chronological_events(self):
        times = [event.occurred_at for event in self.events]
        if times != sorted(times):
            raise ValueError("events must be chronological")
        return self


class LogisticsDataset(LogisticsRecord):
    schema_version: Literal[1]
    source: Literal["mock-logistics-v1"]
    simulated: Literal[True]
    dataset_version: str = Field(min_length=1, max_length=64)
    dataset_hash: Digest
    shipments: dict[TrackingNo, ShipmentRecord]


class ShipmentEnvelope(LogisticsRecord):
    schema_version: Literal[1]
    source: Literal["mock-logistics-v1"]
    simulated: Literal[True]
    dataset_version: str = Field(min_length=1, max_length=64)
    dataset_hash: Digest
    request_id: RequestID
    tracking_no: TrackingNo
    observed_at: AwareDatetime
    outcome: Literal["found", "not_found"]
    shipment: ShipmentSnapshot | None

    @field_validator("observed_at")
    @classmethod
    def utc_observation(cls, value):
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def consistent_outcome(self):
        if (self.outcome == "found") != (self.shipment is not None):
            raise ValueError("outcome and shipment disagree")
        return self


class LogisticsErrorEnvelope(LogisticsRecord):
    code: str = Field(pattern=r"^logistics_[a-z_]+$", max_length=64)
    retryable: bool
    request_id: RequestID | None


class LogisticsBackendProfile(LogisticsRecord):
    backend: Literal["mock-logistics-mcp-v1"]
    source: Literal["mock-logistics-v1"]
    dataset_version: str = Field(min_length=1, max_length=64)
    dataset_hash: Digest
    mcp_url: str
    api_url: str
    input_schema_hash: Digest
    output_schema_hash: Digest
    operation_timeout_seconds: Literal[15] = 15
    api_timeout_seconds: Literal[5] = 5

    @field_validator("mcp_url", "api_url")
    @classmethod
    def loopback_only(cls, value, info):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.port
            or parsed.path != ("/mcp" if info.field_name == "mcp_url" else "")
        ):
            raise ValueError("explicit loopback service URL required")
        return value

    @field_validator("operation_timeout_seconds", "api_timeout_seconds", mode="before")
    @classmethod
    def integer_deadline(cls, value):
        if type(value) is not int:
            raise ValueError("integer deadline required")
        return value

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()
