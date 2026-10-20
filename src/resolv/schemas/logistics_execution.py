"""SDK-free records for persisted logistics execution."""

from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from resolv.schemas.logistics import (
    Digest,
    LogisticsRecord,
    RequestID,
    ShipmentEnvelope,
    TrackingNo,
    canonical_json_bytes,
)


class LogisticsAttemptRecord(LogisticsRecord):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, from_attributes=True)
    attempt_id: RequestID
    action_id: str
    run_id: str
    conversation_id: str
    owner_id: str
    execution_epoch: int = Field(ge=1)
    attempt_no: int = Field(ge=1)
    config_hash: Digest
    arguments_hash: Digest
    tracking_no: TrackingNo
    state: Literal["prepared", "completed", "failed", "unknown"]
    started_at: datetime
    finished_at: datetime | None
    elapsed_ms: int | None = Field(ge=0)
    envelope: dict | None
    projection: dict | None
    result_hash: Digest | None
    error_code: str | None


class BusinessToolPolicy(LogisticsRecord):
    policy_version: Literal["logistics-phase-tools-v1"] = "logistics-phase-tools-v1"
    model_step: int = Field(ge=1, le=6)
    phase: Literal["order", "logistics", "answer", "step_limit"]
    allowed_tool_names: tuple[Literal["query_order", "query_logistics"], ...]
    allowed_order_ids: tuple[str, ...]
    allowed_tracking_nos: tuple[TrackingNo, ...]
    tool_choice: Literal["query_order", "query_logistics", "none"]
    schema_fingerprint: Digest

    @model_validator(mode="after")
    def consistent_phase(self):
        expected = {"order": "query_order", "logistics": "query_logistics"}.get(self.phase, "none")
        if self.tool_choice != expected or self.allowed_tool_names != (() if expected == "none" else (expected,)):
            raise ValueError("phase tools differ")
        if self.phase == "order" and (not self.allowed_order_ids or self.allowed_tracking_nos):
            raise ValueError("order phase requires only current order scope")
        if self.phase == "logistics" and not self.allowed_tracking_nos:
            raise ValueError("logistics phase requires tracking scope")
        return self

    @property
    def fingerprint(self):
        return sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()



def project_logistics_observation(envelope: ShipmentEnvelope, *, attempt_id: str) -> dict:
    shipment = envelope.shipment.model_dump(mode="json") if envelope.shipment else {
        "status": None, "location": None, "shipped_at": None, "signed_at": None,
        "events": [], "events_truncated": False,
    }
    return {
        "schema_version": 1, "found": envelope.outcome == "found",
        "tracking_no": envelope.tracking_no, **shipment,
        "provenance": {
            "source": envelope.source, "simulated": envelope.simulated,
            "dataset_version": envelope.dataset_version, "dataset_hash": envelope.dataset_hash,
            "request_id": envelope.request_id, "observed_at": envelope.observed_at.isoformat(),
            "attempt_id": attempt_id, "result_hash": sha256(canonical_json_bytes(envelope.model_dump(mode="json"))).hexdigest(),
        },
    }
