"""MCP server and API boundary behavior, entirely in-memory and offline."""

import hashlib
import importlib
import json
from time import monotonic

import pytest
from fastapi.testclient import TestClient

from resolv.schemas.logistics import (
    LogisticsBackendProfile,
    LogisticsFailure,
    ShipmentEnvelope,
)
from tests.logistics_fixtures import REQUEST_ID, envelope_payload, profile_payload
from tests.test_logistics_contracts import api


def profile():
    definition = api("resolv.services.logistics_mcp_server").logistics_tool_definition()

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    return LogisticsBackendProfile.model_validate(profile_payload(
        input_schema_hash=digest(definition.input_schema),
        output_schema_hash=digest(definition.output_schema),
    ))


class FixedAPI:
    def __init__(self, failure=None):
        self.requests = []
        self.failure = failure

    async def query(self, tracking_no, *, request_id):
        self.requests.append((tracking_no, request_id))
        if self.failure:
            raise self.failure
        return ShipmentEnvelope.model_validate_json(json.dumps(
            envelope_payload(tracking_no=tracking_no, request_id=request_id)
        ))


@pytest.mark.asyncio
async def test_only_get_shipment_is_exposed():
    module = api("resolv.services.logistics_mcp_server")
    from mcp import Client

    async with Client(module.create_logistics_mcp_server(profile(), FixedAPI())) as client:
        result = await client.list_tools()
    assert [tool.name for tool in result.tools] == ["get_shipment"]
    schema = result.tools[0].input_schema
    assert set(schema["properties"]) == {"tracking_no", "request_id"}
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"tracking_no", "request_id"}
    assert result.tools[0].output_schema["additionalProperties"] is False


@pytest.mark.asyncio
async def test_typed_output_and_request_id_survive_mcp():
    module = api("resolv.services.logistics_mcp_server")
    from mcp import Client

    upstream = FixedAPI()
    async with Client(module.create_logistics_mcp_server(profile(), upstream)) as client:
        result = await client.call_tool("get_shipment", {"tracking_no": "MOCKSF001", "request_id": REQUEST_ID})
    assert result.is_error is False
    assert result.structured_content["tracking_no"] == "MOCKSF001"
    assert result.structured_content["request_id"] == REQUEST_ID
    assert result.structured_content["shipment"]["location"] == "上海转运中心"
    assert upstream.requests == [("MOCKSF001", REQUEST_ID)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,code", [
    (LogisticsFailure("logistics_timeout", True), "logistics_timeout"),
    (RuntimeError("private-provider-secret"), "logistics_service_failed"),
])
async def test_server_error_is_stable_without_private_exception(failure, code):
    module = api("resolv.services.logistics_mcp_server")
    from mcp import Client

    async with Client(module.create_logistics_mcp_server(profile(), FixedAPI(failure))) as client:
        result = await client.call_tool("get_shipment", {"tracking_no": "MOCKSF001", "request_id": REQUEST_ID})
    assert result.is_error is True
    assert result.structured_content == {"code": code, "retryable": True, "request_id": REQUEST_ID}
    assert "private-provider-secret" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_extra_tool_argument_never_reaches_logistics_api():
    module = api("resolv.services.logistics_mcp_server")
    from mcp import Client

    upstream = FixedAPI()
    async with Client(module.create_logistics_mcp_server(profile(), upstream)) as client:
        result = await client.call_tool("get_shipment", {"tracking_no": "MOCKSF001", "request_id": REQUEST_ID, "tenant": "someone-else"})
    assert result.is_error is True
    assert upstream.requests == []


def test_stateless_http_app_has_a_working_lifespan():
    module = api("resolv.services.logistics_mcp_server")
    app = module.create_logistics_mcp_app(profile(), FixedAPI())
    with TestClient(app, base_url="http://127.0.0.1:8102") as client:
        result = client.post("/mcp", headers={"Accept": "application/json, text/event-stream"}, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "isolated-test", "version": "1"}},
        })
    assert result.status_code == 200
    assert result.headers["content-type"].startswith("application/json")
    assert "result" in result.json() and "error" not in result.json()
    assert "Mcp-Session-Id" not in result.headers


def test_mcp_http_rejects_duplicate_json_keys_before_sdk_decodes():
    module = api("resolv.services.logistics_mcp_server")
    upstream = FixedAPI()
    with TestClient(module.create_logistics_mcp_app(profile(), upstream), base_url="http://127.0.0.1:8102") as client:
        response = client.post("/mcp", headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}, content='{"jsonrpc":"2.0","id":1,"id":2,"method":"initialize","params":{}}')
    assert response.status_code == 400
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "DELETE", "HEAD"])
async def test_json_stateless_endpoint_rejects_stream_and_session_methods(method):
    import asyncio

    import httpx2

    module = api("resolv.services.logistics_mcp_server")
    forwarded = []

    async def sdk_stream(scope, receive, send):
        forwarded.append(True)
        await asyncio.Event().wait()

    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=module.StrictLogisticsMCPApp(sdk_stream)), base_url="http://127.0.0.1:8102") as http:
        response = await asyncio.wait_for(http.request(method, "/mcp"), timeout=0.2)
    assert response.status_code == 405 and response.headers["allow"] == "POST"
    assert forwarded == []


@pytest.mark.asyncio
async def test_server_rejects_upstream_identity_mismatch():
    module = api("resolv.services.logistics_mcp_server")
    from mcp import Client

    class WrongAPI:
        async def query(self, tracking_no, *, request_id):
            return ShipmentEnvelope.model_validate_json(json.dumps(envelope_payload(tracking_no="MOCKSF999")))

    async with Client(module.create_logistics_mcp_server(profile(), WrongAPI())) as client:
        result = await client.call_tool("get_shipment", {"tracking_no": "MOCKSF001", "request_id": REQUEST_ID})
    assert result.is_error is True
    assert result.structured_content["code"] == "logistics_result_invalid"


@pytest.mark.asyncio
async def test_api_client_transports_request_id_and_validates_returned_object():
    module = api("resolv.services.logistics_api_client")
    http = importlib.import_module("httpx2")
    requests = []

    async def provider(request):
        requests.append((request.method, request.url.path, request.headers["X-Request-ID"]))
        return http.Response(200, json=envelope_payload())

    async with http.AsyncClient(transport=http.MockTransport(provider)) as client:
        result = await module.LogisticsAPIClient(profile(), http_client=client).query("MOCKSF001", request_id=REQUEST_ID)
        assert result.shipment.location == "上海转运中心"
        with pytest.raises(LogisticsFailure, match="logistics_result_invalid"):
            await module.LogisticsAPIClient(profile(), http_client=client).query("MOCKSF002", request_id=REQUEST_ID)
    assert requests == [("GET", "/v1/shipments/MOCKSF001", REQUEST_ID), ("GET", "/v1/shipments/MOCKSF002", REQUEST_ID)]


@pytest.mark.asyncio
async def test_api_client_does_not_follow_redirects():
    module = api("resolv.services.logistics_api_client")
    http = importlib.import_module("httpx2")
    requests = []

    async def provider(request):
        requests.append(request.url.path)
        return http.Response(307, headers={"Location": "/another-order"})

    async with http.AsyncClient(transport=http.MockTransport(provider)) as client:
        with pytest.raises(LogisticsFailure, match="logistics_protocol_invalid"):
            await module.LogisticsAPIClient(profile(), http_client=client).query("MOCKSF001", request_id=REQUEST_ID)
    assert requests == ["/v1/shipments/MOCKSF001"]


@pytest.mark.asyncio
async def test_api_declared_invalid_result_is_not_misreported_as_outage():
    module = api("resolv.services.logistics_api_client")
    import httpx2

    async def provider(request):
        return httpx2.Response(502, json={"error": {"code": "logistics_result_invalid", "retryable": False, "request_id": REQUEST_ID}})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(provider)) as client:
        with pytest.raises(LogisticsFailure) as caught:
            await module.LogisticsAPIClient(profile(), http_client=client).query("MOCKSF001", request_id=REQUEST_ID)
    assert caught.value.code == "logistics_result_invalid"
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_api_client_five_second_deadline_cancels_operation(monkeypatch):
    module = api("resolv.services.logistics_api_client")
    import anyio
    import httpx2

    deadlines, cancelled = [], []

    def fast_deadline(seconds):
        deadlines.append(seconds)
        return anyio.fail_after(0.01)

    async def stalled(request):
        try:
            await anyio.sleep_forever()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(module, "fail_after", fast_deadline)
    started = monotonic()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(stalled)) as client:
        with pytest.raises(LogisticsFailure, match="logistics_timeout"):
            await module.LogisticsAPIClient(profile(), http_client=client).query("MOCKSF001", request_id=REQUEST_ID)
    assert monotonic() - started < 1
    assert deadlines == [5] and cancelled == [True]
