"""External contracts; production defaults explicitly lack identity and facts."""

from typing import Protocol
from uuid import uuid4

from resolv.db.conversation_base import utc_now
from resolv.schemas.refund import (
    AdapterDescriptor,
    OrderBinding,
    RefundAccessContext,
    RefundFlowError,
    RefundObservation,
)
from resolv.services.hybrid_fingerprints import canonical_hash


class RefundIdentityProvider(Protocol):
    def resolve(self, server_context) -> RefundAccessContext: ...


class RefundFactsAdapter(Protocol):
    descriptor: AdapterDescriptor

    def authorize(
        self, access: RefundAccessContext, binding: OrderBinding
    ) -> RefundObservation: ...
    def query_order(self, access, binding, call_id) -> RefundObservation: ...
    def query_payment(self, access, binding, call_id) -> RefundObservation: ...
    def query_delivery(self, access, binding, call_id) -> RefundObservation: ...
    def query_status(self, access, binding, call_id) -> RefundObservation: ...


class UnconnectedRefundIdentity:
    def resolve(self, server_context):
        raise RefundFlowError("refund_identity_unavailable")


class UnconnectedRefundFactsAdapter:
    descriptor = AdapterDescriptor(
        namespace="external-orders-unconnected-v1",
        implementation_version="unconnected-v1",
        authorization_version="unconnected-v1",
        connected=False,
    )

    def _result(self, name, binding, call_id):
        return RefundObservation.create(
            observation_id=uuid4().hex,
            tool_call_id=call_id,
            tool_name=name,
            binding_hash=binding.binding_hash,
            source=self.descriptor.namespace,
            observed_at=utc_now(),
            provider_version=None,
            arguments_hash=canonical_hash({}),
            availability="not_connected",
            fields={},
        )

    def authorize(self, access, binding):
        return self._result("query_refund_order", binding, "authorization")

    def query_order(self, access, binding, call_id):
        return self._result("query_refund_order", binding, call_id)

    def query_payment(self, access, binding, call_id):
        return self._result("query_refund_payment", binding, call_id)

    def query_delivery(self, access, binding, call_id):
        return self._result("query_refund_delivery", binding, call_id)

    def query_status(self, access, binding, call_id):
        return self._result("query_refund_status", binding, call_id)
