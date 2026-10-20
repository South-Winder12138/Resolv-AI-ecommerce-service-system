"""Production defaults never manufacture trusted identity or query a fixture."""

import pytest

from resolv.schemas.refund import RefundFlowError
from tests.refund_fixtures import binding


def test_unconnected_identity_never_queries_order():
    from resolv.services.refund_adapters import (
        UnconnectedRefundFactsAdapter,
        UnconnectedRefundIdentity,
    )

    facts = UnconnectedRefundFactsAdapter()
    with pytest.raises(RefundFlowError, match="refund_identity_unavailable"):
        UnconnectedRefundIdentity().resolve(
            {"owner_id": "local", "customer_id": "invented"}
        )
    assert facts.descriptor.connected is False
    observed = facts.authorize(None, binding(namespace=facts.descriptor.namespace))
    assert observed.availability == "not_connected" and observed.fields == {}
    for name in ("query_order", "query_payment", "query_delivery", "query_status"):
        assert (
            getattr(facts, name)(
                None, binding(namespace=facts.descriptor.namespace), "x"
            ).availability
            == "not_connected"
        )
