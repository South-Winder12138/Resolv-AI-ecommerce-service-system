"""One explicitly declared MCP tool, backed only by the logistics API."""

import hashlib

from mcp.server import Server
from mcp.types import CallToolResult, ListToolsResult, Tool, ToolAnnotations
from pydantic import ValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from resolv.schemas.logistics import (
    LogisticsBackendProfile,
    LogisticsErrorEnvelope,
    LogisticsFailure,
    LogisticsRecord,
    RequestID,
    ShipmentEnvelope,
    TrackingNo,
    canonical_json_bytes,
)
from resolv.services.logistics_api_client import (
    LogisticsAPIClient,
    validate_envelope_identity,
)
from resolv.services.logistics_dataset import strict_json_loads


class GetShipmentArguments(LogisticsRecord):
    tracking_no: TrackingNo
    request_id: RequestID


def logistics_tool_definition() -> Tool:
    return Tool(
        name="get_shipment",
        description="只读查询模拟运单的物流状态、轨迹与已知签收事实。",
        input_schema=GetShipmentArguments.model_json_schema(),
        output_schema=ShipmentEnvelope.model_json_schema(),
        annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False),
    )


def create_logistics_mcp_server(
    profile: LogisticsBackendProfile, api_client: LogisticsAPIClient
) -> Server:
    definition = logistics_tool_definition()
    for schema, expected in (
        (definition.input_schema, profile.input_schema_hash),
        (definition.output_schema, profile.output_schema_hash),
    ):
        if hashlib.sha256(canonical_json_bytes(schema)).hexdigest() != expected:
            raise LogisticsFailure("logistics_configuration_conflict")

    async def list_tools(ctx, params):
        return ListToolsResult(tools=[definition])

    async def call_tool(ctx, params):
        arguments = None
        try:
            if params.name != "get_shipment":
                raise LogisticsFailure("logistics_invalid_request")
            arguments = GetShipmentArguments.model_validate(params.arguments or {})
            result = await api_client.query(
                arguments.tracking_no, request_id=arguments.request_id
            )
            validate_envelope_identity(
                result, profile, arguments.tracking_no, arguments.request_id
            )
            return CallToolResult(content=[], structured_content=result.model_dump(mode="json"))
        except ValidationError:
            error = LogisticsFailure("logistics_invalid_request")
        except LogisticsFailure as caught:
            error = caught
        except Exception:  # noqa: BLE001 -- external errors never enter the wire
            error = LogisticsFailure("logistics_service_failed", True)
        failure = LogisticsErrorEnvelope(
            code=error.code, retryable=error.retryable,
            request_id=arguments.request_id if arguments else None,
        )
        return CallToolResult(
            content=[], is_error=True, structured_content=failure.model_dump(mode="json")
        )

    return Server(
        "resolv-mock-logistics", version="1",
        on_list_tools=list_tools, on_call_tool=call_tool,
    )


class StrictLogisticsMCPApp:
    """Bound and inspect raw JSON before SDK decoding or forwarding a response."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["method"] != "POST":
            return await JSONResponse(
                {"error": "Only stateless JSON POST is supported"},
                status_code=405, headers={"Allow": "POST"},
            )(scope, receive, send)
        body = bytearray()
        if scope["method"] == "POST":
            try:
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > 65536:
                        raise LogisticsFailure("logistics_result_invalid")
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
                strict_json_loads(bytes(body))
            except LogisticsFailure:
                return await JSONResponse(
                    {"error": {"code": "logistics_invalid_request", "retryable": False, "request_id": None}},
                    status_code=400,
                )(scope, receive, send)
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if scope["method"] == "POST" and not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        response, chunks, size = [], [], 0

        async def bounded_send(message):
            nonlocal size
            if message["type"] == "http.response.body":
                size += len(message.get("body", b""))
                if size <= 65536:
                    chunks.append(message)
            else:
                response.append(message)

        await self.app(scope, replay_receive, bounded_send)
        if size > 65536:
            return await JSONResponse(
                {"error": {"code": "logistics_result_invalid", "retryable": False, "request_id": None}},
                status_code=502,
            )(scope, receive, send)
        for message in [*response, *chunks]:
            await send(message)


def create_logistics_mcp_app(profile, api_client) -> ASGIApp:
    server = create_logistics_mcp_server(profile, api_client)
    return StrictLogisticsMCPApp(server.streamable_http_app(
        json_response=True, stateless_http=True, host="127.0.0.1",
    ))
