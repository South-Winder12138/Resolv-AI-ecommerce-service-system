"""Drive the real MCP SDK against an offline HTTP peer and real ASGI server."""

import json
from collections import Counter
from time import monotonic

import anyio
import httpx2
import pytest

from resolv.schemas.logistics import LogisticsFailure
from resolv.services.logistics_mcp_server import (
    create_logistics_mcp_app,
    logistics_tool_definition,
)
from tests.logistics_fixtures import REQUEST_ID, envelope_payload
from tests.test_logistics_contracts import api
from tests.test_logistics_mcp_server import FixedAPI, profile


class RPCPeer:
    def __init__(self, mode="success"):
        self.mode, self.methods, self.closed, self.cancelled = mode, [], 0, False

    def transport(self):
        peer = self

        class Transport(httpx2.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return await peer.respond(request)

            async def aclose(self):
                peer.closed += 1

        return Transport()

    async def respond(self, request):
        from mcp.types import HEADER_MISMATCH, InputRequiredResult

        body = json.loads(await request.aread())
        method = body.get("method")
        self.methods.append(method)
        if method == "initialize":
            if self.mode == "stalled_initialize":
                try:
                    await anyio.sleep_forever()
                finally:
                    self.cancelled = True
            result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "mock-logistics", "version": "1"}}
        elif "id" not in body:
            return httpx2.Response(202)
        elif method == "tools/list":
            tool = logistics_tool_definition().model_dump(mode="json", by_alias=True, exclude_none=True)
            if self.mode == "schema_drift":
                tool["inputSchema"]["additionalProperties"] = True
            result = {"tools": [tool]}
        else:
            assert method == "tools/call"
            assert request.method == "POST"
            assert body["params"]["name"] == "get_shipment"
            if self.mode == "header_mismatch":
                return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": HEADER_MISMATCH, "message": "private-provider-error"}})
            if self.mode == "input_required":
                result = InputRequiredResult(request_state="untrusted-state").model_dump(mode="json", by_alias=True, exclude_none=True)
            elif self.mode == "sse":
                return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, content=b"data: untrusted\n\n")
            else:
                arguments = body["params"]["arguments"]
                payload = envelope_payload(tracking_no=arguments["tracking_no"], request_id=arguments["request_id"])
                if self.mode == "wrong_identity":
                    payload["tracking_no"] = "OTHER001"
                if self.mode == "not_found":
                    payload.update(outcome="not_found", shipment=None)
                result = {"content": [], "structuredContent": payload, "isError": False}
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "not_found"])
async def test_client_uses_native_mcp_messages_and_preserves_business_result(mode):
    module = api("resolv.services.logistics_client")
    peer = RPCPeer(mode)
    result = await module.MCPLogisticsClient(profile(), transport_factory=peer.transport).query_async("MOCKSF001", request_id=REQUEST_ID)
    assert result.tracking_no == "MOCKSF001" and result.request_id == REQUEST_ID
    assert result.outcome == ("found" if mode == "success" else "not_found")
    assert Counter(peer.methods) == {"initialize": 1, "notifications/initialized": 1, "tools/list": 1, "tools/call": 1}
    assert peer.closed == 1


@pytest.mark.asyncio
async def test_header_mismatch_never_resends():
    module = api("resolv.services.logistics_client")
    peer = RPCPeer("header_mismatch")
    with pytest.raises(LogisticsFailure) as error:
        await module.MCPLogisticsClient(profile(), transport_factory=peer.transport).query_async("MOCKSF001", request_id=REQUEST_ID)
    assert peer.methods.count("tools/call") == 1
    assert peer.methods.count("tools/list") == 1
    assert "private-provider-error" not in str(error.value)


@pytest.mark.asyncio
async def test_input_required_never_resends():
    module = api("resolv.services.logistics_client")
    peer = RPCPeer("input_required")
    with pytest.raises(LogisticsFailure):
        await module.MCPLogisticsClient(profile(), transport_factory=peer.transport).query_async("MOCKSF001", request_id=REQUEST_ID)
    assert peer.methods.count("tools/call") == 1


@pytest.mark.asyncio
async def test_sse_is_rejected_without_reconnect():
    module = api("resolv.services.logistics_client")
    peer = RPCPeer("sse")
    with pytest.raises(LogisticsFailure):
        await module.MCPLogisticsClient(profile(), transport_factory=peer.transport).query_async("MOCKSF001", request_id=REQUEST_ID)
    assert peer.methods.count("tools/call") == 1
    assert len(peer.methods) == 4


@pytest.mark.asyncio
async def test_schema_drift_stops_before_tool_call():
    module = api("resolv.services.logistics_client")
    peer = RPCPeer("schema_drift")
    with pytest.raises(LogisticsFailure, match="logistics_configuration_conflict"):
        await module.MCPLogisticsClient(profile(), transport_factory=peer.transport).query_async("MOCKSF001", request_id=REQUEST_ID)
    assert peer.methods.count("tools/call") == 0


@pytest.mark.asyncio
async def test_client_rejects_valid_schema_for_wrong_object():
    module = api("resolv.services.logistics_client")
    peer = RPCPeer("wrong_identity")
    with pytest.raises(LogisticsFailure, match="logistics_result_invalid"):
        await module.MCPLogisticsClient(profile(), transport_factory=peer.transport).query_async("MOCKSF001", request_id=REQUEST_ID)
    assert peer.methods.count("tools/call") == 1


@pytest.mark.asyncio
async def test_full_operation_deadline_includes_initialization(monkeypatch):
    module = api("resolv.services.logistics_client")
    peer, deadlines = RPCPeer("stalled_initialize"), []

    def short_deadline(seconds):
        deadlines.append(seconds)
        return anyio.fail_after(0.02)

    monkeypatch.setattr(module, "fail_after", short_deadline)
    started = monotonic()
    with pytest.raises(LogisticsFailure, match="logistics_timeout"):
        await module.MCPLogisticsClient(profile(), transport_factory=peer.transport).query_async("MOCKSF001", request_id=REQUEST_ID)
    assert deadlines == [15]
    assert monotonic() - started < 1
    assert peer.cancelled and peer.closed == 1
    assert peer.methods == ["initialize"]


@pytest.mark.asyncio
async def test_real_sdk_http_and_asgi_server_interoperate_without_sockets():
    module = api("resolv.services.logistics_client")
    app = create_logistics_mcp_app(profile(), FixedAPI())
    async with app.app.router.lifespan_context(app.app):
        client = module.MCPLogisticsClient(profile(), transport_factory=lambda: httpx2.ASGITransport(app=app))
        value = await client.query_async("MOCKSF001", request_id=REQUEST_ID)
    assert value.shipment.location == "上海转运中心"
    assert value.simulated is True


def test_sync_entry_owns_a_fresh_closed_event_loop():
    module = api("resolv.services.logistics_client")
    peer = RPCPeer()
    client = module.MCPLogisticsClient(profile(), transport_factory=peer.transport)
    assert client.query("MOCKSF001", request_id=REQUEST_ID).outcome == "found"
    assert client.query("MOCKSF001", request_id="b" * 32).request_id == "b" * 32
    assert peer.closed == 2 and peer.methods.count("tools/call") == 2
