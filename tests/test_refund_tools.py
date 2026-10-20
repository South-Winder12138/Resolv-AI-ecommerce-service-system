"""Only four empty-argument reads on one authorized, server-bound order."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from resolv.schemas.knowledge_branch import FrozenJSON
from resolv.schemas.refund import (
    AdapterDescriptor,
    RefundAccessContext,
    RefundFlowError,
)
from resolv.schemas.refund_flow import RefundToolCall
from resolv.services.hybrid_fingerprints import canonical_hash
from tests.refund_fixtures import binding, observation


def harness():
    bound = binding()
    access = RefundAccessContext(
        subject_ref="customer-1",
        namespace=bound.order_source,
        authorization_version="1",
        trusted=True,
    )
    events = []

    class Adapter:
        descriptor = AdapterDescriptor(
            namespace=bound.order_source,
            implementation_version="1",
            authorization_version="1",
            connected=True,
        )
        availability = "available"

        def authorize(self, access, binding):
            events.append("authorize")
            return observation(
                bound=binding,
                fields={"authorized": True} if self.availability == "available" else {},
                availability=self.availability,
                call_id="auth",
            )

        def query_order(self, access, binding, call_id):
            events.append("order")
            return observation(
                bound=binding, call_id=call_id, fields={"order_id": binding.order_id}
            )

        def query_payment(self, access, binding, call_id):
            events.append("payment")
            return observation("query_refund_payment", bound=binding, call_id=call_id)

        def query_delivery(self, access, binding, call_id):
            events.append("delivery")
            return observation("query_refund_delivery", bound=binding, call_id=call_id)

        def query_status(self, access, binding, call_id):
            events.append("status")
            return observation(
                "query_refund_status",
                bound=binding,
                call_id=call_id,
                availability="not_connected",
            )

    class Registration:
        def lookup(self, binding):
            events.append("local")

    return SimpleNamespace(
        binding=bound,
        access=access,
        adapter=Adapter(),
        registration=Registration(),
        observations=(),
        events=events,
    )


def execute(h, name="query_refund_status", args=None, call_id="call-1"):
    from resolv.services.refund_tools import execute_refund_read

    return execute_refund_read(
        RefundToolCall(id=call_id, name=name, args=FrozenJSON.from_value(args or {})),
        binding=h.binding,
        access=h.access,
        adapter=h.adapter,
        registration=h.registration,
        observations=h.observations,
    )


def test_read_tool_scope_and_authorization():
    from resolv.services.refund_tools import refund_tool_schemas

    h = harness()
    schemas = refund_tool_schemas()
    assert len(schemas) == 4 and all(
        s["function"]["parameters"]
        == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        for s in schemas
    )
    result = execute(h)
    assert h.events == ["authorize", "local", "status"]
    assert (
        result.fields["local_application"].value is None
        and result.fields["local_application"].availability == "available"
    )
    assert result.fields["external_status"].availability == "not_connected"
    for availability in ("forbidden", "not_found"):
        h = harness()
        h.adapter.availability = availability
        result = execute(h)
        assert result.availability == availability and h.events == ["authorize"]


@pytest.mark.parametrize(
    "name,args",
    [
        ("apply_refund", {}),
        ("cancel_order", {}),
        ("query_refund_order", {"order_id": "OTHER"}),
        ("query_refund_payment", {"customer_id": "OTHER"}),
        ("query_refund_status", {"tenant": "OTHER"}),
    ],
)
def test_model_cannot_supply_authority_or_write_tool(name, args):
    h = harness()
    with pytest.raises(RefundFlowError):
        execute(h, name, args)
    assert h.events == []


def test_cross_namespace_and_observation_conflicts_are_rejected():
    h = harness()
    h.binding = binding(namespace="other")
    with pytest.raises(RefundFlowError):
        execute(h)
    h = harness()
    first = execute(h, "query_refund_order")
    h.observations = (first,)
    h.events.clear()
    assert execute(h, "query_refund_order") == first and h.events == []
    with pytest.raises(RefundFlowError):
        execute(h, "query_refund_payment")


def test_delivery_can_only_read_bound_order_and_not_user_claims():
    h = harness()
    result = execute(h, "query_refund_delivery")
    assert (
        h.events == ["authorize", "delivery"]
        and result.binding_hash == h.binding.binding_hash
    )
    assert "activated" not in result.fields


def test_order_observation_records_server_identity_for_offer_checks():
    h = harness()
    result = execute(h, "query_refund_order")
    assert result.fields["subject_ref"].value == h.access.subject_ref


def test_reframed_observation_preserves_provider_timezone():
    from resolv.schemas.refund import RefundObservation
    from resolv.services.refund_tools import _reframe

    h = harness()
    obs = RefundObservation.create(
        observation_id="time",
        tool_call_id="x",
        tool_name="query_refund_order",
        binding_hash=h.binding.binding_hash,
        source=h.binding.order_source,
        observed_at=datetime.fromisoformat("2026-10-16T12:00:00+08:00"),
        provider_version=None,
        arguments_hash=canonical_hash({}),
        availability="available",
        fields={},
    )
    call = RefundToolCall(
        id="x", name="query_refund_order", args=FrozenJSON.from_value({})
    )
    assert (
        _reframe(obs, call, h.binding).original_observed_at == obs.original_observed_at
    )


def test_authorized_order_with_unavailable_payment_keeps_capability_status():
    h = harness()
    h.adapter.query_payment = lambda access, binding, call_id: observation(
        "query_refund_payment",
        bound=binding,
        call_id=call_id,
        availability="not_supported",
    )
    result = execute(h, "query_refund_payment")
    assert result.availability == "not_supported" and not any(
        f.value is not None for f in result.fields.values()
    )
