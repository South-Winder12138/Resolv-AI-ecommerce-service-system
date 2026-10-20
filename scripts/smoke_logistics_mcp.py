"""Opt-in actual loopback wire matrix. Ordinary pytest only tests its boundaries."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from uuid import uuid4

import httpx2
from scripts.logistics_gate_support import (
    logistics_child,
    network_events,
    new_logistics_report_dir,
    owned_profile,
    save_json,
    serve_owned_app,
    snapshot_logistics_protection,
    validate_logistics_ownership,
    validate_service_port,
)

from resolv.repositories.conversation import fingerprint
from resolv.schemas.logistics import LogisticsFailure
from resolv.services.logistics_client import MCPLogisticsClient
from resolv.services.logistics_mcp_server import logistics_tool_definition

CASES = tuple(f"W{i}" for i in range(1, 11))
COUNTS = [(1, 1), (1, 1), (1, 1), (1, 0), (2, 0), (1, 0), (1, 1), (0, 0), (2, 0), (1, 1)]


def require(value, message):
    if not value:
        raise ValueError(message)


def counts(events):
    starts = [e for e in events if e.get("phase") == "start"]
    return (
        sum(e.get("role") == "mcp" and e.get("rpc_method") == "tools/call" for e in starts),
        sum(e.get("role") == "api" and e.get("path", "").startswith("/v1/shipments/") for e in starts),
    )


def validate_wire_report(report):
    require(report.get("format") == "logistics-wire-v1", "Unknown wire report")
    require([c.get("case") for c in report.get("cases", [])] == list(CASES), "All ordered wire cases required")
    require(report.get("counter_sources") == ["api_http_log", "mcp_http_log", "resolv_http_log"], "Independent HTTP counters required")
    require(report.get("real_model_calls") == 0 and report.get("formal_preserved") is True, "Models/formal preservation failed")
    require(report.get("protection_before") and report["protection_before"] == report.get("protection_after"), "Protection evidence differs")
    for case, expected in zip(report["cases"], COUNTS, strict=True):
        require(case.get("checks") == {"expected_outcome": True, "request_identity": True, "bounded_execution": True}, "Wire behavior not proven")
        events = case.get("events", [])
        require(events and all(type(e.get("pid")) is int and e["pid"] > 0 and e.get("event_id") for e in events), "Actual HTTP events required")
        require(counts(events) == expected == (case.get("mcp_tool_calls"), case.get("api_requests")), "Wire counts differ")
        require(case.get("protocol_gets") == 0 and case.get("refresh_extra_calls") == 0, "Unexpected reconnect/redispatch")
        elapsed = case.get("elapsed_seconds", -1)
        require(0 <= elapsed < 25, "Unbounded wire case")
        if case["case"] in {"W7", "W8"}:
            require(elapsed >= (4.8 if case["case"] == "W7" else 14.8), "Real deadline was not reached")


class FaultMCP:
    """Gate-only native HTTP peer. Does not appear in the application tool set."""

    def __init__(self, fault):
        self.fault = fault

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                await send({"type": event["type"] + ".complete"})
                if event["type"] == "lifespan.shutdown":
                    return
        if scope["method"] != "POST":
            await send({"type": "http.response.start", "status": 405, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        raw = bytearray()
        while True:
            event = await receive()
            raw.extend(event.get("body", b""))
            if not event.get("more_body"):
                break
        request = json.loads(raw)
        method = request["method"]
        status, content_type = 200, b"application/json"
        headers = []
        if method == "initialize":
            if self.fault == "deadline":
                await asyncio.sleep(30)
            result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "gate-peer", "version": "1"}}
        elif method == "notifications/initialized":
            status, result = 202, None
        elif method == "tools/list":
            result = {"tools": [logistics_tool_definition().model_dump(mode="json", by_alias=True, exclude_none=True)]}
        else:
            require(method == "tools/call", "Unexpected gate RPC")
            if self.fault == "disconnect":
                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", content_type), (b"content-length", b"9999")]})
                raise ConnectionError("gate disconnect")
            if self.fault == "redirect":
                status, headers = 307, [(b"location", b"/mcp-redirect-target")]
            elif self.fault == "sse":
                content_type = b"text/event-stream"
            result = {"content": [], "structuredContent": {"untrusted": True}, "isError": False}
            if self.fault == "wrong_identity":
                from datetime import UTC, datetime

                from resolv.services.logistics_dataset import (
                    load_dataset,
                    shipment_snapshot,
                )
                dataset = load_dataset(Path(__file__).resolve().parents[1] / "logistics/datasets/mock-logistics-v1.json")
                result["structuredContent"] = {
                    "schema_version": 1, "source": dataset.source, "simulated": True,
                    "dataset_version": dataset.dataset_version, "dataset_hash": dataset.dataset_hash,
                    "request_id": request["params"]["arguments"]["request_id"], "tracking_no": "MOCKSF999",
                    "observed_at": datetime.now(UTC).isoformat(), "outcome": "found",
                    "shipment": shipment_snapshot(dataset.shipments["MOCKSF001"]).model_dump(mode="json"),
                }
        payload = b"" if status == 202 else json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
        if method == "tools/call" and self.fault == "duplicate":
            payload = b'{"jsonrpc":"2.0","id":' + json.dumps(request["id"]).encode() + b',"result":{},"result":{}}'
        if method == "tools/call" and self.fault == "oversize":
            payload = b'{"jsonrpc":"2.0","id":' + json.dumps(request["id"]).encode() + b',"result":{"x":"' + b"x" * 65536 + b'"}}'
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", content_type), *headers]})
        await send({"type": "http.response.body", "body": payload})


def capture_delta(ownership, before):
    known = {(e["event_id"], e["phase"]) for e in before}
    return [e for e in network_events(ownership) if (e["event_id"], e["phase"]) not in known]


def run_wire_cases(ownership, report_dir):
    obj, profile = validate_logistics_ownership(ownership), owned_profile(ownership)
    results = []
    with logistics_child(ownership, report_dir, role="api"):
        for case in CASES:
            started, case_events = time.monotonic(), []
            errors, values = [], []
            refresh_extra = 0
            faults = {"W4": ["wrong_identity"], "W5": ["duplicate", "oversize"], "W6": ["disconnect"], "W8": ["deadline"], "W9": ["redirect", "sse"]}.get(case, [None])
            for fault in faults:
                with ExitStack() as stack:
                    stack.enter_context(logistics_child(ownership, report_dir, role="mcp", worker_module="scripts.smoke_logistics_mcp" if fault else None, extra=("--worker-fault", fault) if fault else ()))
                    # Readiness GET is outside the counted operation window.
                    before_sub = network_events(ownership)
                    sub_started = time.monotonic()
                    if case == "W10":
                        stack.enter_context(logistics_child(ownership, report_dir, role="resolv"))
                        base = f"http://127.0.0.1:{obj['ports']['resolv']}"
                        with httpx2.Client(trust_env=False, timeout=30) as http:
                            conversation = http.post(base + "/conversations").json()["conversation_id"]
                            payload = {"session_id": conversation, "request_id": "wire-" + uuid4().hex, "question": "查询订单 MOCK001 的物流"}
                            response = http.post(base + "/agent/chat", json=payload)
                            require(response.status_code == 200 and response.json()["answer"].startswith("【模拟物流】"), "Fake HTTP logistics flow failed")
                            current = counts(network_events(ownership))
                            with ThreadPoolExecutor(max_workers=4) as pool:
                                responses = list(pool.map(lambda _, base=base, conversation=conversation: http.get(base + "/conversations/" + conversation), range(4)))
                            require(all(r.status_code == 200 for r in responses), "Concurrent detail GET failed")
                            replay = http.post(base + "/agent/chat", json=payload)
                            require(replay.json()["answer"] == response.json()["answer"], "HTTP replay answer differs")
                            require(counts(network_events(ownership)) == current, "Replay connected to logistics")
                            values.append(response.json())
                    else:
                        tracking = {"W2": "MOCKSF404", "W3": "MOCKSF002", "W7": "MOCKSF003"}.get(case, "MOCKSF001")
                        try:
                            value = MCPLogisticsClient(profile).query(tracking, request_id=uuid4().hex)
                            values.append(value.model_dump(mode="json"))
                        except LogisticsFailure as error:
                            errors.append(error.code)
                    elapsed = time.monotonic() - sub_started
                    delta = capture_delta(ownership, before_sub)
                    # Capture before process shutdown; flush is synchronous.
                    case_events.extend(delta)
            expected_errors = {
                "W4": ["logistics_result_invalid"], "W5": ["logistics_result_invalid", "logistics_result_invalid"],
                "W6": ["logistics_transport_failed"], "W7": ["logistics_timeout"],
                "W8": ["logistics_timeout"], "W9": ["logistics_protocol_invalid", "logistics_protocol_invalid"],
            }.get(case, [])
            require(errors == expected_errors, "Unexpected wire error for " + case)
            if case == "W1":
                require(values[0]["shipment"]["status"] == "in_transit", "Shipment fact mismatch")
            elif case == "W2":
                require(values[0]["outcome"] == "not_found" and values[0]["shipment"] is None, "Not-found outcome differs")
            elif case == "W3":
                require(values[0]["shipment"]["status"] == "delivered" and values[0]["shipment"]["signed_at"] is None, "Unknown signature time invented")
            mcp_calls, api_calls = counts(case_events)
            results.append({"case": case, "checks": {"expected_outcome": True, "request_identity": True, "bounded_execution": True},
                "events": case_events, "mcp_tool_calls": mcp_calls, "api_requests": api_calls,
                "protocol_gets": sum(e.get("role") == "mcp" and e.get("method") == "GET" and e.get("phase") == "start" for e in case_events),
                "refresh_extra_calls": refresh_extra, "elapsed_seconds": elapsed,
                "values": values, "errors": errors, "matrix_elapsed_seconds": time.monotonic() - started})
            save_json(Path(report_dir) / "wire-progress.json", {"cases": results})
    return results


def run_wire_matrix(ownership, report_dir):
    before = snapshot_logistics_protection()
    directory = new_logistics_report_dir(ownership, report_dir)
    report = {"format": "logistics-wire-v1", "cases": [], "real_model_calls": 0, "formal_preserved": False,
        "protection_before": fingerprint(before), "counter_sources": ["api_http_log", "mcp_http_log", "resolv_http_log"]}
    try:
        identity = validate_logistics_ownership(ownership)
        report.update(source_revision=identity["source_revision"], source_root=identity["source_root"])
        report["cases"] = run_wire_cases(ownership, directory)
        after = snapshot_logistics_protection(layout=before)
        report.update(protection_after=fingerprint(after), formal_preserved=before == after)
        validate_wire_report(report)
        report["status"] = "passed"
        return report
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        save_json(directory / "report.json", report)


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--ownership", required=True, type=Path)
    value.add_argument("--report-dir", type=Path)
    value.add_argument("--port", type=int)
    value.add_argument("--worker-fault", choices=["wrong_identity", "duplicate", "oversize", "disconnect", "deadline", "redirect", "sse"])
    return value


def main():
    args = parser().parse_args()
    if args.worker_fault:
        validate_service_port(args.ownership, "mcp", args.port)
        serve_owned_app(FaultMCP(args.worker_fault), args.ownership, "mcp", args.port)
    else:
        require(args.report_dir is not None, "report-dir required")
        run_wire_matrix(args.ownership, args.report_dir)


if __name__ == "__main__":
    main()
