"""Acceptance validators must fail closed; no TCP or SQL in ordinary pytest."""

from copy import deepcopy

import pytest

from tests.test_logistics_contracts import api


def wire_report():
    rows = []
    for case, calls, api_calls in [("W1", 1, 1), ("W2", 1, 1), ("W3", 1, 1), ("W4", 1, 0), ("W5", 2, 0), ("W6", 1, 0), ("W7", 1, 1), ("W8", 0, 0), ("W9", 2, 0), ("W10", 1, 1)]:
        events = [{"role": "mcp", "phase": "start", "rpc_method": "tools/call", "pid": 123, "event_id": f"mcp-{i}"} for i in range(calls)]
        events += [{"role": "api", "phase": "start", "path": "/v1/shipments/MOCKSF001", "pid": 124, "event_id": f"api-{i}"} for i in range(api_calls)]
        if case == "W8":
            events.append({"role": "mcp", "phase": "start", "rpc_method": "initialize", "pid": 125, "event_id": "init"})
        rows.append({"case": case, "checks": {"expected_outcome": True, "request_identity": True, "bounded_execution": True},
            "events": events, "mcp_tool_calls": calls, "api_requests": api_calls,
            "refresh_extra_calls": 0, "protocol_gets": 0,
            "elapsed_seconds": 15.1 if case == "W8" else 5.1 if case == "W7" else 0.1})
    return {"format": "logistics-wire-v1", "cases": rows, "real_model_calls": 0, "formal_preserved": True,
        "protection_before": "a" * 64, "protection_after": "a" * 64, "source_revision": "b" * 40,
        "source_root": "/immutable/source", "counter_sources": ["api_http_log", "mcp_http_log", "resolv_http_log"]}


def test_wire_report_accepts_complete_counted_matrix():
    api("scripts.smoke_logistics_mcp").validate_wire_report(wire_report())


@pytest.mark.parametrize("mutation", ["missing_case", "wrong_count", "missing_source", "formal_changed", "model_call", "self_reported_only", "reconnect", "timeout_not_reached"])
def test_wire_report_rejects_missing_or_unsupported_evidence(mutation):
    report = deepcopy(wire_report())
    if mutation == "missing_case":
        report["cases"].pop()
    elif mutation == "wrong_count":
        report["cases"][0]["mcp_tool_calls"] = 0
    elif mutation == "missing_source":
        report["counter_sources"].pop()
    elif mutation == "formal_changed":
        report["protection_after"] = "c" * 64
    elif mutation == "model_call":
        report["real_model_calls"] = 1
    elif mutation == "self_reported_only":
        report["cases"][0]["events"] = []
    elif mutation == "reconnect":
        report["cases"][0]["protocol_gets"] = 1
    else:
        report["cases"][7]["elapsed_seconds"] = 0.1
    with pytest.raises(ValueError):
        api("scripts.smoke_logistics_mcp").validate_wire_report(report)


def test_wire_preflight_failure_cannot_start_any_service(tmp_path, monkeypatch):
    module = api("scripts.smoke_logistics_mcp")
    called = []
    monkeypatch.setattr(module, "snapshot_logistics_protection", lambda **kwargs: (_ for _ in ()).throw(ValueError("formal snapshot unavailable")))
    monkeypatch.setattr(module, "run_wire_cases", lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match="formal snapshot unavailable"):
        module.run_wire_matrix(tmp_path / "ownership.json", tmp_path / "report")
    assert called == []


def test_http_audit_records_native_post_once_before_response(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    module = api("scripts.logistics_gate_support")
    assert hasattr(module, "LogisticsHTTPAudit"), "missing real HTTP counter source"
    inner = FastAPI()

    @inner.post("/mcp")
    async def read():
        return {"ok": True}

    log = tmp_path / "events.jsonl"
    app = module.LogisticsHTTPAudit(inner, role="mcp", path=log)
    with TestClient(app) as http:
        result = http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"private": "secret"}})
    assert result.status_code == 200
    events = module.read_http_events(log)
    assert [e["phase"] for e in events] == ["start", "finish"]
    assert events[0]["rpc_method"] == "tools/call" and events[0]["event_id"] == events[1]["event_id"]
    assert "secret" not in log.read_text() and log.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,code", [("wrong_identity", "logistics_result_invalid"), ("duplicate", "logistics_result_invalid"), ("oversize", "logistics_result_invalid"), ("redirect", "logistics_protocol_invalid"), ("sse", "logistics_protocol_invalid")])
async def test_gate_fault_peer_drives_real_sdk_without_sockets(fault, code):
    import httpx2

    from resolv.schemas.logistics import LogisticsFailure
    from resolv.services.logistics_client import MCPLogisticsClient
    from tests.test_logistics_mcp_server import profile

    module = api("scripts.smoke_logistics_mcp")
    client = MCPLogisticsClient(profile(), transport_factory=lambda: httpx2.ASGITransport(app=module.FaultMCP(fault)))
    with pytest.raises(LogisticsFailure, match=code):
        await client.query_async("MOCKSF001", request_id="a" * 32)
