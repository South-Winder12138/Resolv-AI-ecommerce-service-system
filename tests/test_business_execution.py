import importlib

import pytest

from resolv.schemas.conversation import ConversationError
from tests.test_business_policy import api as policy_api
from tests.test_business_policy import call, observation, primary


def api():
    try:
        return importlib.import_module("resolv.services.business_execution")
    except ModuleNotFoundError:
        pytest.fail("isolated business read executor missing")


def test_reads_use_explicit_factory_and_preserve_real_results(
    tool_database, monkeypatch
):
    import resolv.tools.logistics as logistics_module
    import resolv.tools.order as order_module

    def forbidden(*args, **kwargs):
        pytest.fail("global business database getter called")

    monkeypatch.setattr(order_module, "get_session_factory", forbidden)
    monkeypatch.setattr(logistics_module, "get_session_factory", forbidden)
    policy = policy_api().build_business_policy(
        primary("查订单 MH001 物流", "logistics")
    )
    result = api().execute_business_read(
        call("query_order", order_id="MH001"),
        policy,
        session_factory=tool_database,
        observations=[],
    )
    assert result.output == {
        "found": True,
        "order_id": "MH001",
        "status": "shipped",
        "tracking_no": "SF001",
    }
    logistics = api().execute_business_read(
        call("query_logistics", tracking_no="SF001"),
        policy,
        session_factory=tool_database,
        observations=[result],
    )
    assert logistics.output == {
        "found": True,
        "tracking_no": "SF001",
        "status": "in_transit",
        "location": "上海转运中心",
    }


@pytest.mark.parametrize(
    "tool",
    [
        call(order_id="MH001"),
        call("query_order", order_id="MH002"),
        call("query_refund_status", order_id="MH001"),
    ],
)
def test_invalid_or_write_call_is_rejected_before_opening_business_session(tool):
    calls = []

    def forbidden():
        calls.append(True)
        pytest.fail("unauthorized call opened a business session")

    policy = policy_api().build_business_policy(primary("取消订单 MH001"))
    with pytest.raises(ConversationError):
        api().execute_business_read(
            tool, policy, session_factory=forbidden, observations=[observation()]
        )
    assert calls == []
