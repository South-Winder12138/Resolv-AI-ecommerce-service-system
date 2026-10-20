"""A bounded MCP session; domain facts never come from text content blocks."""

import asyncio
import hashlib
from typing import Protocol

import httpx2
from anyio import fail_after
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import TypeAdapter, ValidationError

from resolv.schemas.logistics import (
    LogisticsBackendProfile,
    LogisticsErrorEnvelope,
    LogisticsFailure,
    RequestID,
    ShipmentEnvelope,
    TrackingNo,
    canonical_json_bytes,
)
from resolv.services.logistics_api_client import validate_envelope_identity
from resolv.services.logistics_transport import GuardedLogisticsTransport


class LogisticsFactsClient(Protocol):
    def query(self, tracking_no: str, *, request_id: str) -> ShipmentEnvelope: ...


def _leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _leaves(child)]
    return [error]


class MCPLogisticsClient:
    def __init__(self, profile: LogisticsBackendProfile, *, transport_factory=None):
        self.profile = profile
        self.transport_factory = transport_factory

    def query(self, tracking_no: str, *, request_id: str) -> ShipmentEnvelope:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.query_async(tracking_no, request_id=request_id))
        raise LogisticsFailure("logistics_protocol_invalid")

    async def query_async(self, tracking_no: str, *, request_id: str) -> ShipmentEnvelope:
        transport = None
        try:
            TypeAdapter(TrackingNo).validate_python(tracking_no, strict=True)
            TypeAdapter(RequestID).validate_python(request_id, strict=True)
            inner = (
                self.transport_factory() if self.transport_factory
                else httpx2.AsyncHTTPTransport(retries=0, trust_env=False)
            )
            transport = GuardedLogisticsTransport(inner)
            with fail_after(self.profile.operation_timeout_seconds):
                async with (
                    httpx2.AsyncClient(
                        transport=transport, trust_env=False, follow_redirects=False,
                        timeout=httpx2.Timeout(self.profile.operation_timeout_seconds),
                        headers={"Accept-Encoding": "identity"},
                    ) as http_client,
                    streamable_http_client(
                        self.profile.mcp_url, http_client=http_client,
                        terminate_on_close=False, max_sse_event_size=65536,
                    ) as (read, write),
                    ClientSession(
                        read, write, read_timeout_seconds=self.profile.operation_timeout_seconds,
                    ) as session,
                ):
                    initialized = await session.initialize()
                    if initialized.protocol_version != "2025-11-25":
                        raise LogisticsFailure("logistics_configuration_conflict")
                    listing = await session.list_tools()
                    if (
                        listing.next_cursor is not None or len(listing.tools) != 1
                        or listing.tools[0].name != "get_shipment"
                    ):
                        raise LogisticsFailure("logistics_configuration_conflict")
                    tool = listing.tools[0]
                    for schema, expected in (
                        (tool.input_schema, self.profile.input_schema_hash),
                        (tool.output_schema, self.profile.output_schema_hash),
                    ):
                        if hashlib.sha256(canonical_json_bytes(schema)).hexdigest() != expected:
                            raise LogisticsFailure("logistics_configuration_conflict")
                    result = await session.call_tool(
                        "get_shipment", {"tracking_no": tracking_no, "request_id": request_id},
                        allow_input_required=False, allow_claimed=False,
                    )
                    value = self._result(result, tracking_no, request_id)
            return value
        except Exception as error:  # noqa: BLE001 -- normalize task-group/provider failures
            leaves = _leaves(error)
            if any(isinstance(leaf, (TimeoutError, httpx2.TimeoutException)) for leaf in leaves):
                raise LogisticsFailure("logistics_timeout", True) from None
            if transport is not None and transport.failure is not None:
                raise transport.failure from None
            domain = [leaf for leaf in leaves if isinstance(leaf, LogisticsFailure)]
            if domain and len(domain) == len(leaves):
                raise domain[0] from None
            if any(isinstance(leaf, ValidationError) for leaf in leaves):
                raise LogisticsFailure("logistics_result_invalid") from None
            raise LogisticsFailure("logistics_protocol_invalid") from None

    def _result(self, result, tracking_no, request_id):
        if not isinstance(result.structured_content, dict):
            raise LogisticsFailure("logistics_result_invalid")
        raw = canonical_json_bytes(result.structured_content)
        if result.is_error:
            failure = LogisticsErrorEnvelope.model_validate_json(raw)
            codes = {
                "logistics_invalid_request": False, "logistics_result_invalid": False,
                "logistics_configuration_conflict": False, "logistics_protocol_invalid": False,
                "logistics_timeout": True, "logistics_transport_failed": True,
                "logistics_service_failed": True,
            }
            if (
                failure.request_id != request_id or failure.code not in codes
                or failure.retryable is not codes[failure.code]
            ):
                raise LogisticsFailure("logistics_result_invalid")
            raise LogisticsFailure(failure.code, failure.retryable)
        envelope = ShipmentEnvelope.model_validate_json(raw)
        return validate_envelope_identity(envelope, self.profile, tracking_no, request_id)
