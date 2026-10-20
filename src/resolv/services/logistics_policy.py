"""Freeze the current decision's tools independently of later observations."""

from resolv.repositories.conversation import fingerprint
from resolv.schemas.branch_execution import BusinessPolicy
from resolv.schemas.conversation import ConversationError
from resolv.schemas.logistics import canonical_json_bytes
from resolv.schemas.logistics_execution import BusinessToolPolicy
from resolv.services.business_policy import (
    build_business_policy,
    business_tool_schemas,
    validate_business_call,
)


def logistics_tool_schemas(policy: BusinessToolPolicy):
    return [s for s in business_tool_schemas() if s["function"]["name"] in policy.allowed_tool_names]


def build_logistics_tool_policy(primary, observations, *, model_step):
    if primary.primary.intent != "logistics":
        raise ConversationError("business_scope_violation")
    scope = build_business_policy(primary)
    phase, orders, tracking = "answer", tuple(scope.allowed_order_ids), []
    if model_step == 6:
        phase = "step_limit"
    else:
        observed_orders = {
            r.output.get("order_id"): r.output for r in observations
            if r.name == "query_order" and isinstance(r.output, dict)
            and r.output.get("order_id") in orders
        }
        missing = [order for order in orders if order not in observed_orders]
        if missing:
            phase, orders = "order", tuple(missing)
        else:
            tracking = list(scope.explicit_tracking_nos)
            tracking.extend(
                r["tracking_no"] for r in observed_orders.values()
                if r.get("found") is True and isinstance(r.get("tracking_no"), str) and r["tracking_no"]
            )
            observed_tracking = {
                r.output.get("tracking_no") for r in observations
                if r.name == "query_logistics" and isinstance(r.output, dict)
            }
            tracking = list(dict.fromkeys(t for t in tracking if t not in observed_tracking))
            if tracking:
                phase = "logistics"
    names = {"order": ("query_order",), "logistics": ("query_logistics",)}.get(phase, ())
    schemas = [s for s in business_tool_schemas() if s["function"]["name"] in names]
    return BusinessToolPolicy(
        model_step=model_step, phase=phase, allowed_tool_names=names,
        allowed_order_ids=orders, allowed_tracking_nos=tuple(tracking),
        tool_choice=names[0] if names else "none", schema_fingerprint=fingerprint(schemas),
    )


def parse_logistics_tool_policy(value: dict, *, expected_fingerprint: str):
    try:
        policy = BusinessToolPolicy.model_validate_json(canonical_json_bytes(value))
        if policy.fingerprint != expected_fingerprint or policy.schema_fingerprint != fingerprint(logistics_tool_schemas(policy)):
            raise ValueError("policy fingerprint differs")
        return policy
    except (ValueError, TypeError):
        raise ConversationError("run_state_incompatible") from None


def validate_logistics_tool_batch(calls, policy: BusinessToolPolicy):
    ids = [call.get("id") for call in calls if isinstance(call, dict)]
    if len(ids) != len(calls) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ConversationError("business_scope_violation")
    scope = BusinessPolicy(
        allowed_order_ids=list(policy.allowed_order_ids),
        explicit_tracking_nos=list(policy.allowed_tracking_nos), cancel_kind="not_requested",
    )
    for call in calls:
        if call.get("name") not in policy.allowed_tool_names:
            raise ConversationError("business_scope_violation")
        validate_business_call(call, scope, observations=[])
