"""Independent logistics wire fixtures; no service, environment or network access."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

REQUEST_ID = "a" * 32
DATASET_VERSION = "20261020-v1"


def shipment(*, status="in_transit", events=0):
    start = datetime(2026, 10, 20, 8, tzinfo=timezone(timedelta(hours=8)))
    return {
        "status": status,
        "location": "上海转运中心",
        "shipped_at": "2026-10-19T10:00:00+08:00",
        "signed_at": None,
        "events": [
            {
                "event_id": f"event-{i:03}",
                "occurred_at": (start + timedelta(minutes=i)).isoformat(),
                "status": status,
                "location": "上海转运中心",
                "description": f"物流轨迹{i}",
            }
            for i in range(events)
        ],
        "events_truncated": False,
    }


def dataset_payload(*, events=0):
    body = {
        "schema_version": 1,
        "source": "mock-logistics-v1",
        "simulated": True,
        "dataset_version": DATASET_VERSION,
        "shipments": {
            "MOCKSF001": shipment(events=events),
            "MOCKSF002": shipment(status="delivered"),
            "MOCKSF003": shipment(),
        },
    }
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {**body, "dataset_hash": hashlib.sha256(canonical.encode()).hexdigest()}


def envelope_payload(**overrides):
    dataset = dataset_payload()
    return deepcopy({
        "schema_version": 1,
        "source": "mock-logistics-v1",
        "simulated": True,
        "dataset_version": DATASET_VERSION,
        "dataset_hash": dataset["dataset_hash"],
        "request_id": REQUEST_ID,
        "tracking_no": "MOCKSF001",
        "observed_at": "2026-10-20T00:00:00Z",
        "outcome": "found",
        "shipment": dataset["shipments"]["MOCKSF001"],
    } | overrides)


def profile_payload(**overrides):
    return {
        "backend": "mock-logistics-mcp-v1",
        "source": "mock-logistics-v1",
        "dataset_version": DATASET_VERSION,
        "dataset_hash": dataset_payload()["dataset_hash"],
        "mcp_url": "http://127.0.0.1:8102/mcp",
        "api_url": "http://127.0.0.1:8101",
        "input_schema_hash": "1" * 64,
        "output_schema_hash": "2" * 64,
        "operation_timeout_seconds": 15,
        "api_timeout_seconds": 5,
    } | overrides
