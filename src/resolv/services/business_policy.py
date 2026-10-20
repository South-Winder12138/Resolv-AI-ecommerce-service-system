"""Execution scope from a verified primary request, never from conversation history."""

import re

from pydantic import ValidationError

from resolv.repositories.conversation import fingerprint
from resolv.schemas.branch_execution import BranchReply, BusinessPolicy, ReplyMetadata
from resolv.schemas.conversation import ConversationError
from resolv.services.reference_resolution import active_business_ids
from resolv.tools.registry import get_registered_tool

BUSINESS_TOOL_NAMES = ("query_order", "query_logistics", "cancel_order")
BUSINESS_TOOLS_VERSION = "business-tools-v1"


def business_tool_schemas():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return [
        convert_to_openai_tool(get_registered_tool(name).tool)
        for name in BUSINESS_TOOL_NAMES
    ]


def business_tools_fingerprint():
    return fingerprint(
        {"version": BUSINESS_TOOLS_VERSION, "tools": business_tool_schemas()}
    )


def supported_processing_condition(text, orders):
    """Recognize the whole supported predicate, never just a status substring."""
    normalized = re.sub(r"\s|[，,。！!]", "", text)
    for value in sorted(orders, key=len, reverse=True):
        normalized = re.sub(
            r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])",
            "@",
            normalized,
        )
    obj = r"(?:订单(?:号)?@|@|它|该订单|[这那](?:笔|个)?订单)"
    return (
        re.fullmatch(
            r"(?:如果|若|只要)?" + obj + r"?(?:还|仍)?(?:处于)?"
            r"(?:未发货|没发货|尚未发货|待发货|处理中|processing)(?:状态)?"
            r"(?:的话)?(?:才|就|则)?(?:请帮我|帮我|请|麻烦)?(?:取消|撤销)" + obj + r"?",
            normalized,
        )
        is not None
    )


def build_business_policy(primary):
    p = primary.primary
    text = re.sub(r'“[^”]*”|‘[^’]*’|「[^」]*」|"[^"]*"|\'[^\']*\'', "", p.text)
    corrections = list(re.finditer(r"更正为|改成|正确的是|应该是|实际是", text))
    entity_text = text[corrections[-1].end() :] if corrections else text
    ids = list(
        dict.fromkeys(
            value
            for value in active_business_ids(entity_text)
            if not re.search(
                r"(?:商品(?:编码|编号|号)?|SKU|优惠券(?:码|号)?|会员(?:号|编码)?|支付流水(?:号)?|交易流水(?:号)?)\s*[:：]?\s*"
                + re.escape(value)
                + r"(?![A-Za-z0-9_-])",
                entity_text,
                re.IGNORECASE,
            )
            and (
                not value.isdecimal()
                or re.search(
                    r"(?:订单(?:号)?|运单(?:号)?|物流单(?:号)?|快递单(?:号)?)\s*[:：]?\s*"
                    + re.escape(value)
                    + r"(?![A-Za-z0-9_-])",
                    entity_text,
                )
            )
        )
    )
    tracking = [
        value
        for value in ids
        if re.search(
            r"(?:运单|物流单|快递单)(?:号)?\s*[:：]?\s*" + re.escape(value), entity_text
        )
    ]
    for binding in p.bindings:
        if (
            (binding.value in ids or not ids)
            and binding.entity_type == "shipment"
            and binding.value not in tracking
        ):
            tracking.append(binding.value)
    orders = [value for value in ids if value not in tracking]
    for binding in p.bindings:
        # These bindings have already been checked against the selected clause
        # and frozen source catalog. A pronoun need not contain its resolved ID.
        if (
            binding.entity_type == "order"
            and (not ids or binding.value in ids)
            and binding.value not in orders
        ):
            orders.append(binding.value)
    kind, condition = "not_requested", None
    if re.search(r"取消|撤销", text):
        if re.search(
            r"(?:不要|不用|别|不需要|不想|禁止).{0,8}(?:取消|撤销)|(?:无法|不能)\s*(?:取消|撤销)|历史批准|历史指令|商家说|客服说|工具.*说明",
            text,
        ):
            kind = "not_requested"
        elif re.search(
            r"能否|能不能|是否|是不是|可否|可不可以|会不会|吗|么|[？?]|怎么|如何|多久|什么|多少|取消状态|取消会怎样|(?:取消|撤销).{0,80}(?:后|会|影响|手续费|优惠券)",
            text,
        ):
            kind = "ambiguous"
        elif re.search(r"如果|若|(?<!刚)才|的话|只要|就取消", text):
            kind = (
                "conditional"
                if supported_processing_condition(text, orders)
                else "ambiguous"
            )
            condition = "processing" if kind == "conditional" else None
        elif re.search(
            r"^(?:请|麻烦|帮我|请帮我|我要|我想|现在|直接|立即|请立即|首先|优先|先|再|然后|最后|接着)*\s*(?:取消|撤销)|^把.{0,80}(?:取消|撤销)|(?:请|帮我|我要|我想)(?:取消|撤销)|^(?:请|麻烦|帮我|请帮我)?为订单\s*[A-Za-z0-9_-]+\s*(?:发起|进行|办理)取消(?:操作|申请)?[。！!]*$",
            text,
        ):
            kind = "explicit"
        else:
            kind = "ambiguous"
    return BusinessPolicy(
        allowed_order_ids=orders,
        explicit_tracking_nos=tracking,
        cancel_kind=kind,
        cancel_condition=condition,
    )


def validate_business_call(call, policy, *, observations):
    def reject():
        raise ConversationError("business_scope_violation")

    if not isinstance(call, dict) or call.get("name") not in BUSINESS_TOOL_NAMES:
        reject()
    name, args = call["name"], call.get("args")
    key = "tracking_no" if name == "query_logistics" else "order_id"
    if (
        not isinstance(args, dict)
        or set(args) != {key}
        or not isinstance(args[key], str)
        or not args[key].strip()
    ):
        reject()
    try:
        validated = (
            get_registered_tool(name).tool.args_schema.model_validate(args).model_dump()
        )
    except ValidationError:
        reject()
    if validated != args:
        reject()
    value = args[key]
    if name == "query_logistics":
        allowed = set(policy.explicit_tracking_nos)
        for result in observations:
            data = result.output
            if (
                result.name == "query_order"
                and isinstance(data, dict)
                and data.get("found") is True
                and data.get("order_id") in policy.allowed_order_ids
                and isinstance(data.get("tracking_no"), str)
            ):
                allowed.add(data["tracking_no"])
        if value not in allowed:
            reject()
    elif value not in policy.allowed_order_ids:
        reject()
    if name == "cancel_order":
        if policy.cancel_kind not in {"explicit", "conditional"}:
            reject()
        if policy.cancel_kind == "conditional":
            matching = [
                r.output
                for r in observations
                if r.name == "query_order"
                and isinstance(r.output, dict)
                and r.output.get("order_id") == value
            ]
            if (
                policy.cancel_condition != "processing"
                or not matching
                or matching[-1].get("found") is not True
                or matching[-1].get("status") != "processing"
            ):
                reject()


def requires_business_tool(primary, observations):
    policy = build_business_policy(primary)
    # All observations were revalidated against this run's selected primary.
    if any(result.name == "cancel_order" for result in observations):
        return False
    orders = {
        result.output["order_id"]: result.output
        for result in observations
        if result.name == "query_order"
        and isinstance(result.output, dict)
        and result.output.get("order_id") in policy.allowed_order_ids
    }
    if policy.cancel_kind == "explicit":
        return True
    if policy.cancel_kind == "conditional":
        return any(
            order not in orders or orders[order].get("status") == "processing"
            for order in policy.allowed_order_ids
        )
    if primary.primary.intent == "logistics":
        tracking = set(policy.explicit_tracking_nos)
        for order in policy.allowed_order_ids:
            if order not in orders:
                return True
            if orders[order].get("found") and orders[order].get("tracking_no"):
                tracking.add(orders[order]["tracking_no"])
        observed = {
            result.output.get("tracking_no")
            for result in observations
            if result.name == "query_logistics" and isinstance(result.output, dict)
        }
        return not tracking <= observed
    return not set(policy.allowed_order_ids) <= orders.keys()


def business_input_reply(run_id, primary):
    policy = build_business_policy(primary)
    if not policy.allowed_order_ids and not (
        primary.primary.intent == "logistics" and policy.explicit_tracking_nos
    ):
        example = (
            "取消订单 XXX"
            if "取消" in primary.primary.text
            else (
                "查询订单 XXX 的物流"
                if primary.primary.intent == "logistics"
                else "查询订单 XXX 的状态"
            )
        )
        answer = f"请提供具体对象并完整描述诉求，例如“{example}”。"
    elif policy.cancel_kind == "ambiguous" and re.search(
        r"如果|若|(?<!刚)才|的话|只要", primary.primary.text
    ):
        answer = "暂时无法可靠判断该取消条件，请明确说明需要查询的问题，或重新完整描述取消诉求，例如“取消订单 XXX”。"
    else:
        return None
    return BranchReply(
        answer=answer,
        metadata=ReplyMetadata(
            reply_kind="clarification",
            understanding_id=primary.understanding_id,
            primary_task_id=primary.primary.task_id,
            branch="business",
        ),
    )
