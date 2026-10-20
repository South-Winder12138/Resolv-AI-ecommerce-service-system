"""Refund boundary contracts reject coercion and invented provenance."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("confirm", [1, 0, "true", False, None])
def test_strict_submission_rejects_coerced_confirmation(confirm):
    from resolv.schemas.refund_submission import RefundSubmitInput

    with pytest.raises(ValidationError):
        RefundSubmitInput.model_validate_json(json.dumps({"request_id": "r", "confirm": confirm}))


def test_strict_submission_and_money():
    from resolv.schemas.refund import RefundMoney
    from resolv.schemas.refund_submission import RefundSubmitInput

    assert RefundSubmitInput(request_id="r" * 128, confirm=True).confirm is True
    for value in ({"request_id": "", "confirm": True}, {"request_id": " ", "confirm": True}, {"request_id": "r" * 129, "confirm": True}, {"request_id": "r", "confirm": True, "order_id": "other"}):
        with pytest.raises(ValidationError):
            RefundSubmitInput.model_validate(value)
    for amount in (True, 1.0, "100", -1, 2**63):
        with pytest.raises(ValidationError):
            RefundMoney(currency="CNY", minor_units=amount, unit_precision=2)
    money = RefundMoney(currency="CNY", minor_units=9007199254740993, unit_precision=2)
    assert money.public().minor_units_decimal == "9007199254740993"


def test_availability_and_status_decision():
    from resolv.schemas.refund import RefundFieldValue
    from resolv.schemas.refund_flow import RefundDecision

    assert RefundFieldValue(availability="unknown").value is None
    with pytest.raises(ValidationError):
        RefundFieldValue(availability="unknown", value="delivered")
    with pytest.raises(ValidationError):
        RefundDecision(operation="status", verdict="eligible", order_binding_hash="a" * 64,
                       evidence_hash="b" * 64, reason="status", next_action="show_status")


def test_binding_hash_and_namespace():
    from resolv.schemas.refund import OrderBinding

    data = {"order_id": "A001", "understanding_id": "u", "primary_task_id": "t", "primary_hash": "a" * 64,
            "source_refs": ({"kind": "input", "source_id": "current", "span": {"start": 0, "end": 4}, "json_path": [], "quote": "A001"},)}
    a = OrderBinding.create(order_source="tenant-a", **data)
    b = OrderBinding.create(order_source="tenant-b", **data)
    assert a.binding_hash != b.binding_hash
    with pytest.raises(ValidationError):
        OrderBinding.model_validate_json(a.model_dump_json().replace("tenant-a", "tenant-b"))


def test_observation_rejects_naive_numeric_and_preserves_original_timezone():
    from resolv.schemas.refund import RefundObservation

    data = {"observation_id": "obs", "tool_call_id": "call", "tool_name": "query_refund_order",
            "binding_hash": "a" * 64, "source": "tenant-a", "arguments_hash": "b" * 64,
            "provider_version": None, "availability": "available", "fields": {}}
    for time in (datetime(2026, 10, 16, tzinfo=None), 12345):  # noqa: DTZ001 -- intentional invalid input
        with pytest.raises(ValidationError):
            RefundObservation.create(observed_at=time, **data)
    obs = RefundObservation.create(observed_at=datetime(2026, 10, 16, tzinfo=UTC), **data)
    assert obs.observed_at.tzinfo is not None
    assert obs.audit_hash


def test_json_duplicate_keys_and_nonfinite_rejected():
    from resolv.schemas.refund_submission import RefundSubmitInput

    for raw in ('{"request_id":"a","request_id":"b","confirm":true}', '{"request_id":NaN,"confirm":true}'):
        with pytest.raises((ValidationError, ValueError)):
            RefundSubmitInput.model_validate_json(raw)
