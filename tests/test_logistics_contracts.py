"""Catch accepted malformed/ambiguous facts before they can become observations."""

import importlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from tests.logistics_fixtures import dataset_payload, envelope_payload, profile_payload


def api(name="resolv.schemas.logistics"):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            pytest.fail(f"required logistics implementation missing: {name}")
        raise


def parse(payload):
    return api().ShipmentEnvelope.model_validate_json(json.dumps(payload))


def test_dataset_hash_excludes_only_own_hash():
    hashing = api("resolv.services.logistics_dataset").dataset_hash
    expected = "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    assert hashing({"a": 1, "dataset_hash": "old"}) == expected
    assert hashing({"a": 1, "dataset_hash": "new"}) == expected
    assert hashing({"a": 2}) != expected


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(extra="untrusted"),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(simulated=1),
    lambda p: p.update(observed_at="2026-10-20T12:00:00"),
    lambda p: p.update(outcome="not_found"),
    lambda p: p.update(shipment=None),
    lambda p: p["shipment"].update(status="refunded"),
    lambda p: p["shipment"].update(location="中" * 257),
    lambda p: p["shipment"].update(signed_at="2026-10-20"),
])
def test_envelope_rejects_unknown_keys_and_object_mismatch(mutate):
    payload = envelope_payload()
    mutate(payload)
    with pytest.raises(ValidationError):
        parse(payload)


def test_unknown_signed_at_stays_null():
    payload = envelope_payload()
    payload["shipment"]["status"] = "delivered"
    value = parse(payload)
    assert value.shipment.status == "delivered"
    assert value.shipment.signed_at is None
    assert value.observed_at.utcoffset().total_seconds() == 0


def test_normal_not_found_has_no_shipment():
    value = parse(envelope_payload(outcome="not_found", shipment=None))
    assert value.outcome == "not_found" and value.shipment is None


@pytest.mark.parametrize("tracking", ["", "../secret", "SF/001", "中001", "A" * 65, " SF001"])
def test_tracking_format_is_strict(tracking):
    with pytest.raises(ValidationError):
        parse(envelope_payload(tracking_no=tracking))


@pytest.mark.parametrize("override", [
    {"mcp_url": "http://example.com/mcp"},
    {"mcp_url": "http://127.0.0.1.evil/mcp"},
    {"mcp_url": "http://secret@127.0.0.1/mcp"},
    {"api_url": "http://127.0.0.1/api?secret=x"},
    {"operation_timeout_seconds": 300},
    {"api_timeout_seconds": 60},
])
def test_profile_rejects_arbitrary_source_or_expanded_deadline(override):
    with pytest.raises(ValidationError):
        api().LogisticsBackendProfile.model_validate(profile_payload(**override))


def test_loader_rejects_duplicate_keys_and_changed_dataset(tmp_path):
    module = api("resolv.services.logistics_dataset")
    path = tmp_path / "dataset.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(api().LogisticsFailure, match="logistics_result_invalid"):
        module.load_dataset(path)
    payload = dataset_payload()
    payload["shipments"]["MOCKSF001"]["location"] = "被修改的地点"
    path.write_text(json.dumps(payload))
    with pytest.raises(api().LogisticsFailure):
        module.load_dataset(path)


def test_events_must_have_timezone_unique_ids_and_bounded_text():
    payload = envelope_payload()
    payload["shipment"] = dataset_payload(events=2)["shipments"]["MOCKSF001"]
    for key, value in [("occurred_at", "2026-10-20T08:00:00"), ("description", "x" * 513)]:
        bad = deepcopy(payload)
        bad["shipment"]["events"][0][key] = value
        with pytest.raises(ValidationError):
            parse(bad)
    payload["shipment"]["events"][1]["event_id"] = "event-000"
    with pytest.raises(ValidationError):
        parse(payload)


@pytest.mark.parametrize("field", ["observed_at", "shipped_at", "signed_at", "occurred_at"])
@pytest.mark.parametrize("value", ["1730000000", "1730000000.125", "2026-10-19T10:00:00"])
def test_numeric_timestamp_strings_cannot_invent_an_implicit_timezone(field, value):
    payload = envelope_payload()
    if field == "observed_at":
        payload[field] = value
    elif field == "occurred_at":
        payload["shipment"] = dataset_payload(events=1)["shipments"]["MOCKSF001"]
        payload["shipment"]["events"][0][field] = value
    else:
        payload["shipment"][field] = value
    with pytest.raises(ValidationError):
        parse(payload)


def test_explicit_offsets_survive_and_internal_aware_values_remain_valid():
    from datetime import datetime, timedelta

    payload = envelope_payload()
    payload["observed_at"] = "2026-10-20T10:00:00+05:30"
    payload["shipment"] = dataset_payload(events=1)["shipments"]["MOCKSF001"]
    payload["shipment"]["events"][0]["occurred_at"] = "2026-10-20T08:00:00+05:30"
    value = parse(payload)
    assert value.observed_at.isoformat() == "2026-10-20T04:30:00+00:00"
    assert value.shipment.events[0].occurred_at.utcoffset() == timedelta(hours=5, minutes=30)
    assert api().ShipmentEnvelope.model_validate(value.model_dump()).observed_at == value.observed_at
    with pytest.raises(ValidationError):
        api().ShipmentEnvelope.model_validate({**value.model_dump(), "observed_at": datetime(2026, 10, 20)})  # noqa: DTZ001 -- deliberately invalid naive input


def test_utf8_response_limit_is_bytes_and_json_keys_are_unique():
    decode = api("resolv.services.logistics_dataset").strict_json_loads
    small = json.dumps({"payload": "🙂" * 16000}, ensure_ascii=False).encode()
    assert decode(small)["payload"] == "🙂" * 16000
    large = json.dumps({"payload": "🙂" * 17000}, ensure_ascii=False).encode()
    with pytest.raises(api().LogisticsFailure):
        decode(large)
    for raw in [b'{"a":1,"a":2}', b'{"a":NaN}', b'[]']:
        with pytest.raises(api().LogisticsFailure):
            decode(raw)
