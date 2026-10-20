"""Exercise the actual isolated HTTP routes without listening on a socket."""

import json
from datetime import datetime

from fastapi.testclient import TestClient

from tests.logistics_fixtures import REQUEST_ID, dataset_payload
from tests.test_logistics_contracts import api


def client_for(tmp_path, *, events=0):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset_payload(events=events)))
    dataset = api("resolv.services.logistics_dataset").load_dataset(path)
    app = api("resolv.api.mock_logistics").create_mock_logistics_app(dataset)
    return TestClient(app, base_url="http://127.0.0.1")


def query(client, tracking="MOCKSF001"):
    return client.get(f"/v1/shipments/{tracking}", headers={"X-Request-ID": REQUEST_ID})


def test_mock_http_not_found_is_business_result(tmp_path):
    with client_for(tmp_path) as client:
        response = query(client, "MOCKSF404")
    assert response.status_code == 200
    assert response.json()["outcome"] == "not_found"
    assert response.json()["shipment"] is None
    assert response.json()["request_id"] == REQUEST_ID


def test_latest_twenty_events_are_explicit(tmp_path):
    with client_for(tmp_path, events=25) as client:
        response = query(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "mock-logistics-v1" and payload["simulated"] is True
    assert payload["shipment"]["location"] == "上海转运中心"
    events = payload["shipment"]["events"]
    assert len(events) == 20
    assert [events[0]["event_id"], events[-1]["event_id"]] == ["event-005", "event-024"]
    assert payload["shipment"]["events_truncated"] is True
    assert datetime.fromisoformat(payload["observed_at"]).utcoffset().total_seconds() == 0


def test_api_never_invents_missing_signed_time(tmp_path):
    with client_for(tmp_path) as client:
        value = query(client, "MOCKSF002").json()
        health = client.get("/healthz").json()
    assert value["shipment"]["status"] == "delivered"
    assert value["shipment"]["signed_at"] is None
    assert health["dataset_hash"] == value["dataset_hash"]
    assert "shipments" not in health


def test_invalid_request_id_and_tracking_are_rejected(tmp_path):
    with client_for(tmp_path) as client:
        missing = client.get("/v1/shipments/MOCKSF001")
        assert missing.status_code == 422
        assert missing.json() == {"error": {"code": "logistics_invalid_request", "retryable": False, "request_id": None}}
        invalid = client.get("/v1/shipments/MOCKSF001", headers={"X-Request-ID": "private-input"})
        assert invalid.status_code == 422
        assert "private-input" not in invalid.text
        assert query(client, "A" * 65).status_code == 422
        assert client.post("/v1/shipments/MOCKSF001").status_code == 405


def test_queries_never_modify_the_dataset_file(tmp_path):
    with client_for(tmp_path) as client:
        path = tmp_path / "dataset.json"
        before = path.read_bytes()
        assert query(client).status_code == 200
        assert query(client, "MOCKSF404").status_code == 200
        assert path.read_bytes() == before
