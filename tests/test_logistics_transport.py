"""Inspect real HTTP message boundaries before the SDK can decode or redirect."""

import json

import httpx2
import pytest

from resolv.schemas.logistics import LogisticsFailure
from tests.test_logistics_contracts import api

URL = "http://127.0.0.1:8102/mcp"
CALL = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_shipment", "arguments": {}}}


@pytest.mark.asyncio
async def test_same_origin_307_is_not_followed():
    cls = api("resolv.services.logistics_transport").GuardedLogisticsTransport
    paths = []

    async def peer(request):
        paths.append(request.url.path)
        return httpx2.Response(307, headers={"Location": "/another-tool"})

    async with httpx2.AsyncClient(transport=cls(httpx2.MockTransport(peer)), follow_redirects=True) as client:
        with pytest.raises(LogisticsFailure, match="logistics_protocol_invalid"):
            await client.post(URL, json=CALL)
    assert paths == ["/mcp"]


@pytest.mark.asyncio
async def test_duplicate_keys_rejected_before_sdk():
    cls = api("resolv.services.logistics_transport").GuardedLogisticsTransport
    response = httpx2.Response(200, headers={"Content-Type": "application/json"}, content=b'{"jsonrpc":"2.0","id":1,"result":{"tracking_no":"A1","tracking_no":"B2"}}')
    async with httpx2.AsyncClient(transport=cls(httpx2.MockTransport(lambda request: response))) as client:
        with pytest.raises(LogisticsFailure, match="logistics_result_invalid"):
            await client.post(URL, json=CALL)


@pytest.mark.asyncio
async def test_multibyte_response_limit_is_bytes():
    cls = api("resolv.services.logistics_transport").GuardedLogisticsTransport
    for characters, rejected in [(15000, False), (17000, True)]:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"text": "🙂" * characters}}, ensure_ascii=False).encode()
        response = httpx2.Response(200, headers={"Content-Type": "application/json"}, content=body)
        async with httpx2.AsyncClient(transport=cls(httpx2.MockTransport(lambda request, response=response: response))) as client:
            if rejected:
                with pytest.raises(LogisticsFailure):
                    await client.post(URL, json=CALL)
            else:
                result = await client.post(URL, json=CALL)
                assert result.json()["result"]["text"] == "🙂" * characters


@pytest.mark.asyncio
async def test_compressed_response_rejected():
    cls = api("resolv.services.logistics_transport").GuardedLogisticsTransport

    async def peer(request):
        return httpx2.Response(200, headers={"Content-Type": "application/json", "Content-Encoding": "gzip"}, stream=httpx2.ByteStream(b"not-a-valid-gzip"))

    async with httpx2.AsyncClient(transport=cls(httpx2.MockTransport(peer))) as client:
        with pytest.raises(LogisticsFailure):
            await client.post(URL, json=CALL)


@pytest.mark.asyncio
async def test_initialized_notification_202_is_not_a_call_result():
    cls = api("resolv.services.logistics_transport").GuardedLogisticsTransport
    async with httpx2.AsyncClient(transport=cls(httpx2.MockTransport(lambda request: httpx2.Response(202)))) as client:
        accepted = await client.post(URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert accepted.status_code == 202
        with pytest.raises(LogisticsFailure):
            await client.post(URL, json=CALL)


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{"Content-Type": "text/event-stream"}, {"Content-Type": "application/json", "Mcp-Session-Id": "unexpected-session"}])
async def test_stateful_or_sse_response_cannot_start_another_stream(headers):
    cls = api("resolv.services.logistics_transport").GuardedLogisticsTransport
    async with httpx2.AsyncClient(transport=cls(httpx2.MockTransport(lambda request: httpx2.Response(200, headers=headers, json={"jsonrpc": "2.0", "id": 1, "result": {}})))) as client:
        with pytest.raises(LogisticsFailure, match="logistics_protocol_invalid"):
            await client.post(URL, json=CALL)
