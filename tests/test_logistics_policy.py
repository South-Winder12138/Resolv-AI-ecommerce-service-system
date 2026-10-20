import json

import pytest
from pydantic import ValidationError

from resolv.schemas.conversation import ConversationError
from resolv.tools.executor import ToolExecutionResult
from tests.test_business_model import profile as business_profile
from tests.test_business_policy import call, observation, primary
from tests.test_logistics_contracts import api
from tests.test_logistics_mcp_server import profile


def config():
    module = api("resolv.schemas.branch_execution")
    assert hasattr(module, "BusinessRunConfigV2"), "missing versioned MCP business config"
    return module.BusinessRunConfigV2.model_validate({
        **business_profile().model_dump(mode="json"), "config_version": 2,
        "prompt_version": "business-logistics-mcp-v1", "logistics_backend": profile().model_dump(mode="json"),
    })


def policy(observations=(), *, step=1, text="查订单 MOCK001 物流"):
    return api("resolv.services.logistics_policy").build_logistics_tool_policy(primary(text, "logistics"), list(observations), model_step=step)


def test_order_logistics_first_exposes_only_query_order():
    p = policy()
    assert p.allowed_tool_names == ("query_order",) and p.phase == "order"
    assert p.allowed_order_ids == ("MOCK001",) and p.allowed_tracking_nos == ()
    assert p.tool_choice == "query_order"
    with pytest.raises(ConversationError, match="business_scope_violation"):
        api("resolv.services.logistics_policy").validate_logistics_tool_batch([call("query_logistics", tracking_no="MOCKSF001")], p)


def test_current_order_unlocks_matching_tracking():
    p = policy([observation(order="MOCK001", tracking="MOCKSF001")], step=2)
    assert p.allowed_tool_names == ("query_logistics",) and p.phase == "logistics"
    assert p.allowed_tracking_nos == ("MOCKSF001",)
    module = api("resolv.services.logistics_policy")
    module.validate_logistics_tool_batch([call("query_logistics", tracking_no="MOCKSF001")], p)
    with pytest.raises(ConversationError, match="business_scope_violation"):
        module.validate_logistics_tool_batch([call("query_logistics", tracking_no="OTHER")], p)


def test_unrelated_order_observation_cannot_unlock_logistics():
    assert policy([observation(order="OTHER", tracking="MOCKSF001")]).phase == "order"


def test_direct_tracking_skips_order_lookup():
    p = policy(text="查询运单 MOCKSF002 的物流轨迹")
    assert p.allowed_tool_names == ("query_logistics",) and p.allowed_tracking_nos == ("MOCKSF002",)


@pytest.mark.parametrize("observations", [
    [observation(order="MOCK001", tracking=None)],
    [ToolExecutionResult("a", "query_order", {"found": False, "order_id": "MOCK001"})],
    [observation(order="MOCK001", tracking="MOCKSF001"), ToolExecutionResult("b", "query_logistics", {"found": True, "tracking_no": "MOCKSF001"})],
])
def test_no_tracking_or_completed_read_closes_tools(observations):
    p = policy(observations, step=3)
    assert p.phase == "answer" and p.allowed_tool_names == () and p.tool_choice == "none"


def test_sixth_step_has_no_tools():
    p = policy(step=6)
    assert p.phase == "step_limit" and p.allowed_tool_names == () and p.tool_choice == "none"


def test_mixed_batch_is_rejected_without_partial_acceptance():
    p = policy()
    module = api("resolv.services.logistics_policy")
    for calls in [
        [call("query_order", order_id="MOCK001"), {"id": "tc2", "name": "query_logistics", "args": {"tracking_no": "MOCKSF001"}}],
        [call("query_order", order_id="MOCK001", extra=True)],
        [call("cancel_order", order_id="MOCK001")],
    ]:
        with pytest.raises(ConversationError, match="business_scope_violation"):
            module.validate_logistics_tool_batch(calls, p)


def test_saved_policy_is_used_for_replay_not_new_decision():
    module = api("resolv.services.logistics_policy")
    old = policy([observation(order="MOCK001", tracking="MOCKSF001")], step=2)
    closed = policy([observation(order="MOCK001", tracking="MOCKSF001"), ToolExecutionResult("b", "query_logistics", {"found": True, "tracking_no": "MOCKSF001"})], step=3)
    saved = module.parse_logistics_tool_policy(old.model_dump(mode="json"), expected_fingerprint=old.fingerprint)
    calls = [call("query_logistics", tracking_no="MOCKSF001")]
    module.validate_logistics_tool_batch(calls, saved)
    with pytest.raises(ConversationError, match="business_scope_violation"):
        module.validate_logistics_tool_batch(calls, closed)
    tampered = {**old.model_dump(mode="json"), "allowed_tracking_nos": ["OTHER"]}
    with pytest.raises(ConversationError, match="run_state_incompatible"):
        module.parse_logistics_tool_policy(tampered, expected_fingerprint=old.fingerprint)


def test_strict_config_parser_preserves_legacy_and_freezes_mcp_profile():
    module = api("resolv.schemas.branch_execution")
    assert hasattr(module, "parse_business_config"), "missing versioned config parser"
    old = module.parse_business_config(business_profile().model_dump(mode="json"))
    assert type(old) is module.BusinessRunConfig and old.config_version == 1
    new = module.parse_business_config(config().model_dump(mode="json"))
    assert new.config_version == 2 and new.logistics_backend == profile()
    assert new.tool_policy_version == "logistics-phase-tools-v1"
    for invalid in [2.0, True, 3]:
        with pytest.raises(ValueError):
            module.parse_business_config({**new.model_dump(mode="json"), "config_version": invalid})
    with pytest.raises(ValidationError):
        module.parse_business_config({**new.model_dump(mode="json"), "logistics_backend": {**profile().model_dump(mode="json"), "operation_timeout_seconds": 30}})
    assert json.loads(new.model_dump_json())["logistics_backend"]["source"] == "mock-logistics-v1"


def test_omitted_legacy_config_version_remains_local():
    module = api("resolv.schemas.branch_execution")
    value = business_profile().model_dump(mode="json")
    del value["config_version"]
    result = module.parse_business_config(value)
    assert type(result) is module.BusinessRunConfig and result.config_version == 1
    assert not hasattr(result, "logistics_backend")
