"""Separate deterministic logistics API; never mounted into the customer app."""

import asyncio
import re
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException

from resolv.schemas.logistics import (
    LogisticsDataset,
    LogisticsErrorEnvelope,
    RequestID,
    ShipmentEnvelope,
    TrackingNo,
    canonical_json_bytes,
)
from resolv.services.logistics_dataset import shipment_snapshot


def create_mock_logistics_app(
    dataset: LogisticsDataset, *, fault: str | None = None
) -> FastAPI:
    if fault not in {None, "api_timeout"}:
        raise ValueError("unsupported isolated fault")
    app = FastAPI(title="Resolv isolated mock logistics", docs_url=None, redoc_url=None)

    def error_response(code, status, request_id=None):
        value = LogisticsErrorEnvelope(code=code, retryable=False, request_id=request_id)
        return JSONResponse({"error": value.model_dump(mode="json")}, status_code=status)

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(HTTPException)
    async def invalid_request(request, error):
        request_id = request.headers.get("X-Request-ID", "")
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            request_id = None
        return error_response(
            "logistics_invalid_request",
            getattr(error, "status_code", 422),
            request_id,
        )

    @app.get("/healthz")
    def health():
        return {
            "status": "ok", "schema_version": dataset.schema_version,
            "source": dataset.source, "simulated": True,
            "dataset_version": dataset.dataset_version, "dataset_hash": dataset.dataset_hash,
        }

    @app.get("/v1/shipments/{tracking_no}")
    async def query(
        tracking_no: TrackingNo,
        request_id: Annotated[RequestID, Header(alias="X-Request-ID")],
    ):
        if fault == "api_timeout" and tracking_no == "MOCKSF003":
            await asyncio.sleep(6)
        record = dataset.shipments.get(tracking_no)
        value = ShipmentEnvelope(
            schema_version=1, source=dataset.source, simulated=True,
            dataset_version=dataset.dataset_version, dataset_hash=dataset.dataset_hash,
            request_id=request_id, tracking_no=tracking_no,
            observed_at=datetime.now(UTC), outcome="found" if record else "not_found",
            shipment=shipment_snapshot(record) if record else None,
        )
        body = canonical_json_bytes(value.model_dump(mode="json"))
        if len(body) > 65536:
            return error_response("logistics_result_invalid", 502, request_id)
        return Response(body, media_type="application/json")

    return app
