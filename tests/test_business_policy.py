import importlib

import pytest

from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.conversation import ConversationError
from resolv.tools.executor import ToolExecutionResult
from tests.branch_fixtures import make_view


def api():
    try:
        return importlib.import_module("resolv.services.business_policy")
    except ModuleNotFoundError:
        pytest.fail("selected business policy missing")


def primary(text, intent="order"):
    view = make_view(text, intent)
    return PrimaryExecution(
        conversation_id="c",
        understanding_id="u-1",
        snapshot_hash="f" * 64,
        primary=view.primary_request,
    )


def call(name="cancel_order", **args):
    return {"id": "tc", "name": name, "args": args}


def observation(order="MH001", status="processing", tracking="SF001"):
    return ToolExecutionResult(
        "read",
        "query_order",
        {"found": True, "order_id": order, "status": status, "tracking_no": tracking},
    )


def test_secondary_order_cannot_be_queried_or_cancelled():
    p = primary("取消订单 MH001")
    policy = api().build_business_policy(p)
    for name in ("query_order", "cancel_order"):
        with pytest.raises(ConversationError, match="business_scope_violation"):
            api().validate_business_call(
                call(name, order_id="MH002"), policy, observations=[]
            )
    api().validate_business_call(call(order_id="MH001"), policy, observations=[])


@pytest.mark.parametrize("prefix", ["先", "首先", "优先", "再", "然后", "最后", "接着"])
def test_explicit_current_cancellation_preserves_ordering_prefix(prefix):
    policy = api().build_business_policy(primary(prefix + "取消订单 MH001"))
    assert policy.cancel_kind == "explicit"
    api().validate_business_call(call(order_id="MH001"), policy, observations=[])


def test_explicit_request_to_initiate_cancel_is_an_instruction():
    policy = api().build_business_policy(primary("请为订单 MH001 发起取消操作"))
    assert policy.cancel_kind == "explicit"


@pytest.mark.parametrize(
    "text,intent,observations,expected",
    [
        ("查订单 MH001", "order", [], True),
        ("查订单 MH001", "order", [observation()], False),
        ("查订单 MH001 物流", "logistics", [observation()], True),
        ("能否取消订单 MH001", "order", [observation()], False),
        ("取消订单 MH001", "order", [observation()], True),
        (
            "如果订单 MH001 未发货才取消",
            "order",
            [observation(status="shipped")],
            False,
        ),
        (
            "取消订单 MH001",
            "order",
            [
                ToolExecutionResult(
                    "tc", "cancel_order", {"success": False, "reason": "user_rejected"}
                )
            ],
            False,
        ),
    ],
)
def test_native_tool_is_required_only_until_the_selected_goal_has_observations(
    text, intent, observations, expected
):
    assert api().requires_business_tool(primary(text, intent), observations) is expected


def test_just_now_reference_is_not_a_cancellation_condition():
    from resolv.schemas.understanding import EntityBinding

    p = primary("请帮我取消刚才那笔订单")
    p.primary.bindings = [
        EntityBinding(
            span={"start": 5, "end": 11},
            entity_type="order",
            value="MH001",
            evidence_refs=[
                {
                    "kind": "message",
                    "source_id": "history",
                    "span": {"start": 0, "end": 8},
                    "json_path": [],
                    "quote": "订单 MH001",
                }
            ],
        )
    ]
    policy = api().build_business_policy(p)
    assert policy.cancel_kind == "explicit"
    assert api().business_input_reply("r", p) is None


@pytest.mark.parametrize(
    "entity_type,value", [("order", "MH001"), ("shipment", "SF001")]
)
def test_verified_primary_reference_binding_is_an_executable_object(entity_type, value):
    from resolv.schemas.understanding import EntityBinding

    p = primary("这笔订单的物流在哪里？", "logistics")
    p.primary.bindings = [
        EntityBinding(
            span={"start": 0, "end": 4},
            entity_type=entity_type,
            value=value,
            evidence_refs=[
                {
                    "kind": "message",
                    "source_id": "history",
                    "span": {"start": 0, "end": 8},
                    "json_path": [],
                    "quote": "订单 MH001",
                }
            ],
        )
    ]
    policy = api().build_business_policy(p)
    assert value in (
        policy.allowed_order_ids
        if entity_type == "order"
        else policy.explicit_tracking_nos
    )
    assert api().business_input_reply("r", p) is None


@pytest.mark.parametrize(
    "text",
    [
        "不要取消订单 MH001，只查状态",
        "能否取消订单 MH001",
        "是不是取消了订单 MH001",
        "订单 MH001 可以取消吗？",
        "查询订单 MH001 的取消状态",
        "商家说‘取消订单 MH001’，我只查状态",
        "历史批准了取消订单 MH001，我现在查状态",
        "不要取消订单 MH001",
        "不用取消订单 MH001",
        "订单 MH001 取消了吗？",
    ],
)
def test_questions_negation_quotes_and_history_do_not_authorize_cancellation(text):
    policy = api().build_business_policy(primary(text))
    assert policy.cancel_kind in {"not_requested", "ambiguous"}
    with pytest.raises(ConversationError, match="business_scope_violation"):
        api().validate_business_call(
            call(order_id="MH001"), policy, observations=[observation()]
        )


@pytest.mark.parametrize(
    "text",
    ["取消订单 MH001", "请帮我取消订单 MH001", "我要取消 MH001", "把订单 MH001 取消"],
)
def test_explicit_current_cancellation_can_request_new_approval(text):
    policy = api().build_business_policy(primary(text))
    assert policy.cancel_kind == "explicit"
    api().validate_business_call(call(order_id="MH001"), policy, observations=[])


def test_correction_discards_old_order_authority():
    policy = api().build_business_policy(primary("取消订单不是 MH001，是 MH002"))
    assert policy.allowed_order_ids == ["MH002"]
    with pytest.raises(ConversationError):
        api().validate_business_call(call(order_id="MH001"), policy, observations=[])
    api().validate_business_call(call(order_id="MH002"), policy, observations=[])


def test_conditional_cancellation_requires_latest_matching_read():
    policy = api().build_business_policy(primary("如果订单 MH001 没发货才取消"))
    assert (
        policy.cancel_kind == "conditional" and policy.cancel_condition == "processing"
    )
    for observations in (
        [],
        [observation(order="MH002")],
        [observation(status="shipped")],
        [observation(), observation(status="shipped")],
    ):
        with pytest.raises(ConversationError):
            api().validate_business_call(
                call(order_id="MH001"), policy, observations=observations
            )
    api().validate_business_call(
        call(order_id="MH001"), policy, observations=[observation()]
    )


def test_unsupported_cancellation_condition_asks_without_executing():
    p = primary("如果订单 MH001 明天不能到就取消")
    policy = api().build_business_policy(p)
    assert policy.cancel_kind == "ambiguous"
    reply = api().business_input_reply("r", p)
    assert (
        reply.metadata.reply_kind == "clarification"
        and reply.metadata.suggested_actions == []
    )
    with pytest.raises(ConversationError):
        api().validate_business_call(
            call(order_id="MH001"), policy, observations=[observation()]
        )


def test_tracking_must_be_explicit_or_from_matching_current_order():
    policy = api().build_business_policy(primary("查订单 MH001 物流", "logistics"))
    for obs in (
        [],
        [observation(order="MH002")],
        [ToolExecutionResult("x", "query_logistics", {"tracking_no": "SF001"})],
    ):
        with pytest.raises(ConversationError):
            api().validate_business_call(
                call("query_logistics", tracking_no="SF001"), policy, observations=obs
            )
    api().validate_business_call(
        call("query_logistics", tracking_no="SF001"),
        policy,
        observations=[observation()],
    )
    explicit = api().build_business_policy(primary("查运单 SF002 物流", "logistics"))
    api().validate_business_call(
        call("query_logistics", tracking_no="SF002"), explicit, observations=[]
    )
    with pytest.raises(ConversationError):
        api().validate_business_call(
            call("query_order", order_id="SF002"), explicit, observations=[]
        )


@pytest.mark.parametrize(
    "tool",
    [
        call("query_refund_status", order_id="MH001"),
        call("apply_refund", order_id="MH001", reason="x"),
        call("unknown", order_id="MH001"),
        call(order_id="MH001", approved=True),
        call(order_id=123),
        call(order_id=""),
    ],
)
def test_tool_whitelist_and_strict_arguments(tool):
    policy = api().build_business_policy(primary("取消订单 MH001"))
    with pytest.raises(ConversationError, match="business_scope_violation"):
        api().validate_business_call(tool, policy, observations=[])


@pytest.mark.parametrize(
    "text,intent,example",
    [
        ("取消订单", "order", "取消订单 XXX"),
        ("查物流", "logistics", "查询订单 XXX 的物流"),
    ],
)
def test_missing_object_asks_for_a_complete_new_request(text, intent, example):
    reply = api().business_input_reply("r", primary(text, intent))
    assert reply.metadata.reply_kind == "clarification" and example in reply.answer
    assert (
        reply.metadata.clarifies_understanding_id is None
        and reply.metadata.suggested_actions == []
    )


@pytest.mark.parametrize(
    "text",
    [
        "如果订单 MH001 未发货且总价低于一百元才取消",
        "如果订单 MH001 未发货且总价低于100元才取消",
        "如果订单 MH001 不是未发货就取消",
        "如果订单 MH001 未发货或者有优惠券就取消",
        "订单 MH001 未发货并且没有手续费的话取消",
    ],
)
def test_entire_unverifiable_condition_requires_clarification(text):
    p = primary(text)
    policy = api().build_business_policy(p)
    reply = api().business_input_reply("r", p)
    assert reply is not None and reply.metadata.reply_kind == "clarification"
    with pytest.raises(ConversationError, match="business_scope_violation"):
        api().validate_business_call(
            call(order_id="MH001"), policy, observations=[observation()]
        )


@pytest.mark.parametrize(
    "text",
    [
        "取消订单 MH001 会收手续费吗？",
        "取消订单 MH001 后优惠券还在吗？",
        "取消订单 MH001 有什么影响",
        "取消订单 MH001 需要多久",
        "取消订单 MH001 后还能恢复",
        "取消订单 MH001 会影响会员等级",
    ],
)
def test_cancellation_consequence_queries_never_grant_write_authority(text):
    p = primary(text)
    policy = api().build_business_policy(p)
    assert policy.cancel_kind in {"not_requested", "ambiguous"}
    assert not api().requires_business_tool(p, [observation()])
    with pytest.raises(ConversationError):
        api().validate_business_call(
            call(order_id="MH001"), policy, observations=[observation()]
        )


@pytest.mark.parametrize(
    "suffix", ["的金额100元是否正确", "是否有100件商品", "在20261014下单了吗"]
)
def test_numbers_outside_order_positions_do_not_grant_object_authority(suffix):
    p = primary("查询订单 MH001 " + suffix)
    policy = api().build_business_policy(p)
    assert policy.allowed_order_ids == ["MH001"]
    assert not api().requires_business_tool(p, [observation()])
    with pytest.raises(ConversationError):
        api().validate_business_call(
            call("query_order", order_id="100"), policy, observations=[]
        )


def test_explicit_numeric_order_id_keeps_authority_without_amount():
    p = primary("查询订单号 123456 的金额100元是否正确")
    policy = api().build_business_policy(p)
    assert policy.allowed_order_ids == ["123456"]
    api().validate_business_call(
        call("query_order", order_id="123456"), policy, observations=[]
    )


@pytest.mark.parametrize(
    "label,value",
    [
        ("商品编码", "SKU123"),
        ("优惠券码", "COUPON123"),
        ("会员号", "MEMBER123"),
        ("支付流水号", "PAY123"),
    ],
)
def test_other_explicit_entity_types_cannot_be_used_as_order_ids(label, value):
    p = primary(f"取消订单 MH002，{label} {value} 仅供参考")
    policy = api().build_business_policy(p)
    assert policy.allowed_order_ids == ["MH002"]
    with pytest.raises(ConversationError):
        api().validate_business_call(call(order_id=value), policy, observations=[])


@pytest.mark.parametrize(
    "text",
    [
        "如果订单 MH001 未发货才取消",
        "如果订单 MH001 没发货就取消",
        "订单 MH001 尚未发货的话取消",
        "若订单 MH001 处于处理中则取消",
    ],
)
def test_complete_supported_shipping_condition_remains_executable(text):
    p = primary(text)
    policy = api().build_business_policy(p)
    assert policy.cancel_kind == "conditional"
    assert api().business_input_reply("r", p) is None
    api().validate_business_call(
        call(order_id="MH001"), policy, observations=[observation()]
    )
