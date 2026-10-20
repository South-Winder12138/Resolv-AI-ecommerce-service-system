"""Five-second read boundary from MCP Server to the fixed logistics API."""

import httpx2
from anyio import fail_after
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
from resolv.services.logistics_dataset import strict_json_loads


def validate_envelope_identity(envelope, profile, tracking_no, request_id):
    if (
        not isinstance(envelope, ShipmentEnvelope)
        or envelope.tracking_no != tracking_no
        or envelope.request_id != request_id
        or envelope.source != profile.source
        or envelope.dataset_version != profile.dataset_version
        or envelope.dataset_hash != profile.dataset_hash
    ):
        raise LogisticsFailure("logistics_result_invalid")
    return envelope


class LogisticsAPIClient:
    def __init__(
        self, profile: LogisticsBackendProfile, *, http_client: httpx2.AsyncClient | None = None
    ):
        self.profile, self.http_client = profile, http_client

    async def query(self, tracking_no: str, *, request_id: str) -> ShipmentEnvelope:
        try:
            TypeAdapter(TrackingNo).validate_python(tracking_no, strict=True)
            TypeAdapter(RequestID).validate_python(request_id, strict=True)
            with fail_after(self.profile.api_timeout_seconds):
                if self.http_client is not None:
                    return await self._query(self.http_client, tracking_no, request_id)
                async with httpx2.AsyncClient(
                    trust_env=False, follow_redirects=False,
                    timeout=httpx2.Timeout(self.profile.api_timeout_seconds),
                ) as client:
                    return await self._query(client, tracking_no, request_id)
        except (TimeoutError, httpx2.TimeoutException):
            raise LogisticsFailure("logistics_timeout", True) from None
        except httpx2.RequestError:
            raise LogisticsFailure("logistics_transport_failed", True) from None
        except ValidationError:
            raise LogisticsFailure("logistics_result_invalid") from None

    async def _query(self, client, tracking_no, request_id):
        async with client.stream(
            "GET", self.profile.api_url + "/v1/shipments/" + tracking_no,
            headers={"X-Request-ID": request_id, "Accept-Encoding": "identity"},
            follow_redirects=False,
        ) as response:
            if 300 <= response.status_code < 400:
                raise LogisticsFailure("logistics_protocol_invalid")
            if (
                response.headers.get("content-type", "").split(";", 1)[0] != "application/json"
                or response.headers.get("content-encoding", "identity") != "identity"
            ):
                if response.status_code >= 500:
                    raise LogisticsFailure("logistics_service_failed", True)
                raise LogisticsFailure("logistics_protocol_invalid")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > 65536:
                    raise LogisticsFailure("logistics_result_invalid")
                body.extend(chunk)
            payload = strict_json_loads(bytes(body))
            if response.status_code >= 400:
                if set(payload) != {"error"}:
                    raise LogisticsFailure("logistics_protocol_invalid")
                failure = LogisticsErrorEnvelope.model_validate_json(
                    canonical_json_bytes(payload["error"])
                )
                expected_retryable = {
                    "logistics_invalid_request": False,
                    "logistics_result_invalid": False,
                    "logistics_timeout": True,
                    "logistics_service_failed": True,
                }.get(failure.code)
                if (
                    failure.request_id != request_id
                    or expected_retryable is None
                    or failure.retryable is not expected_retryable
                ):
                    raise LogisticsFailure("logistics_protocol_invalid")
                raise LogisticsFailure(failure.code, failure.retryable)
            if response.status_code != 200:
                raise LogisticsFailure("logistics_protocol_invalid")
            envelope = ShipmentEnvelope.model_validate_json(canonical_json_bytes(payload))
            return validate_envelope_identity(envelope, self.profile, tracking_no, request_id)
