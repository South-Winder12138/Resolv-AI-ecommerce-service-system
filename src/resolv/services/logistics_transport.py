"""Raw HTTP guard for a stateless, JSON-only, single-attempt MCP operation."""

from collections import Counter
from urllib.parse import urlsplit

import httpx2

from resolv.schemas.logistics import LogisticsFailure
from resolv.services.logistics_dataset import strict_json_loads


class GuardedLogisticsTransport(httpx2.AsyncBaseTransport):
    def __init__(self, inner, *, max_bytes: int = 65536):
        self.inner, self.max_bytes = inner, max_bytes
        self.failure = None
        self.sent_methods = Counter()
        self.endpoint = None

    async def handle_async_request(self, request):
        response = None
        try:
            url = urlsplit(str(request.url))
            if (
                request.method != "POST" or url.scheme != "http"
                or url.hostname != "127.0.0.1" or not url.port
                or url.username or url.password or url.query or url.fragment
                or url.path != "/mcp"
                or (self.endpoint is not None and self.endpoint != str(request.url))
            ):
                raise LogisticsFailure("logistics_protocol_invalid")
            self.endpoint = str(request.url)
            outbound = strict_json_loads(await request.aread(), max_bytes=self.max_bytes)
            method = outbound.get("method")
            if method not in {
                "initialize", "notifications/initialized", "notifications/cancelled",
                "tools/list", "tools/call",
            } or self.sent_methods[method]:
                raise LogisticsFailure("logistics_protocol_invalid")
            self.sent_methods[method] += 1
            response = await self.inner.handle_async_request(request)
            if response.status_code == 202:
                if "id" in outbound or method != "notifications/initialized":
                    raise LogisticsFailure("logistics_protocol_invalid")
                return httpx2.Response(202, request=request)
            if (
                300 <= response.status_code < 400
                or "mcp-session-id" in response.headers
                or response.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json"
                or response.headers.get("content-encoding", "identity") != "identity"
            ):
                raise LogisticsFailure("logistics_protocol_invalid")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > self.max_bytes:
                    raise LogisticsFailure("logistics_result_invalid")
                body.extend(chunk)
            payload = strict_json_loads(bytes(body), max_bytes=self.max_bytes)
            if (
                payload.get("jsonrpc") != "2.0"
                or set(payload) - {"jsonrpc", "id", "result", "error"}
                or ("result" in payload) == ("error" in payload)
                or type(payload.get("id")) is not type(outbound.get("id"))
                or payload.get("id") != outbound.get("id")
                or "id" not in outbound
            ):
                raise LogisticsFailure("logistics_protocol_invalid")
            return httpx2.Response(
                response.status_code, headers=response.headers, content=bytes(body),
                request=request, extensions=response.extensions,
            )
        except LogisticsFailure as error:
            self.failure = self.failure or error
            raise
        except httpx2.TimeoutException:
            self.failure = LogisticsFailure("logistics_timeout", True)
            raise self.failure from None
        except httpx2.RequestError:
            self.failure = LogisticsFailure("logistics_transport_failed", True)
            raise self.failure from None
        finally:
            if response is not None:
                await response.aclose()

    async def aclose(self):
        await self.inner.aclose()
