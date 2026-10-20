"""Read-only authoritative mock dataset and bounded wire JSON decoding."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from resolv.schemas.logistics import (
    LogisticsDataset,
    LogisticsFailure,
    ShipmentRecord,
    ShipmentSnapshot,
    canonical_json_bytes,
)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("nonfinite value")


def strict_json_loads(data: bytes | str, *, max_bytes: int | None = 65536) -> dict:
    try:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        if max_bytes is not None and len(raw) > max_bytes:
            raise ValueError("response too large")
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
        if not isinstance(result, dict):
            raise TypeError("object required")
        return result
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise LogisticsFailure("logistics_result_invalid") from None


def dataset_hash(payload: dict) -> str:
    return hashlib.sha256(
        canonical_json_bytes({key: value for key, value in payload.items() if key != "dataset_hash"})
    ).hexdigest()


def load_dataset(path: Path) -> LogisticsDataset:
    try:
        # Owned local data is not a network response; full history may exceed a response.
        payload = strict_json_loads(path.read_bytes(), max_bytes=None)
        if payload.get("dataset_hash") != dataset_hash(payload):
            raise LogisticsFailure("logistics_result_invalid")
        return LogisticsDataset.model_validate_json(canonical_json_bytes(payload))
    except (OSError, ValidationError, ValueError, TypeError):
        raise LogisticsFailure("logistics_result_invalid") from None


def shipment_snapshot(record: ShipmentRecord) -> ShipmentSnapshot:
    # Python's stable sort preserves provider order for equal event times.
    events = sorted(record.events, key=lambda event: event.occurred_at)
    return ShipmentSnapshot(
        status=record.status,
        location=record.location,
        shipped_at=record.shipped_at,
        signed_at=record.signed_at,
        events=events[-20:],
        events_truncated=record.events_truncated or len(events) > 20,
    )
