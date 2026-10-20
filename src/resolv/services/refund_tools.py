"""Server-bound read tools; model arguments cannot carry any authority."""

from datetime import datetime
from uuid import uuid4

from resolv.schemas.refund import (
    AdapterDescriptor,
    OrderBinding,
    RefundAccessContext,
    RefundFlowError,
    RefundObservation,
)
from resolv.schemas.refund_flow import RefundToolCall
from resolv.services.hybrid_fingerprints import canonical_hash

TOOLS = {
    "query_refund_order": ("query_order", "查询当前已绑定订单及已核验访问权限。"),
    "query_refund_payment": (
        "query_payment",
        "查询当前订单权威支付、整单可退金额和币种。",
    ),
    "query_refund_delivery": ("query_delivery", "查询当前订单所属物流、签收事实。"),
    "query_refund_status": (
        "query_status",
        "查询当前订单已知申请；本地状态不代表资金到账。",
    ),
}


def refund_tool_schemas():
    return tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
        for name, (_, description) in TOOLS.items()
    )


def _check(observation, binding):
    obs = RefundObservation.model_validate_json(observation.model_dump_json())
    if obs.binding_hash != binding.binding_hash or obs.source != binding.order_source:
        raise RefundFlowError("refund_observation_conflict")
    return obs


def _reframe(obs, call, binding, *, fields=None, availability=None):
    return RefundObservation.create(
        observation_id=uuid4().hex,
        tool_call_id=call.id,
        tool_name=call.name,
        binding_hash=binding.binding_hash,
        source=binding.order_source,
        observed_at=datetime.fromisoformat(obs.original_observed_at),
        provider_version=obs.provider_version,
        arguments_hash=canonical_hash({}),
        availability=availability or obs.availability,
        fields=fields if fields is not None else obs.fields,
        error_code=obs.error_code,
    )


def execute_refund_read(call, *, binding, access, adapter, registration, observations):
    call = RefundToolCall.model_validate_json(call.model_dump_json())
    binding = OrderBinding.model_validate_json(binding.model_dump_json())
    if call.name not in TOOLS or call.args.value != {}:
        raise RefundFlowError("refund_tool_scope_violation")
    descriptor = AdapterDescriptor.model_validate_json(
        adapter.descriptor.model_dump_json()
    )
    if descriptor.namespace != binding.order_source:
        raise RefundFlowError("refund_adapter_conflict")
    if access is None:
        raise RefundFlowError("refund_identity_unavailable")
    access = RefundAccessContext.model_validate_json(access.model_dump_json())
    if (
        access.namespace != descriptor.namespace
        or access.authorization_version != descriptor.authorization_version
    ):
        raise RefundFlowError("refund_authorization_conflict")
    for old in observations:
        old = _check(old, binding)
        if old.tool_call_id == call.id:
            if old.tool_name != call.name or old.arguments_hash != canonical_hash({}):
                raise RefundFlowError("refund_observation_conflict")
            return old
    authorization = _check(adapter.authorize(access, binding), binding)
    allowed = authorization.fields.get("authorized")
    if authorization.availability != "available":
        return _reframe(authorization, call, binding)
    if allowed is not None and allowed.availability == "available" and allowed.value is False:
        return _reframe(authorization, call, binding, availability="forbidden", fields={})
    if (
        allowed is None
        or allowed.availability != "available"
        or allowed.value is not True
    ):
        raise RefundFlowError("refund_authorization_conflict")
    local = registration.lookup(binding) if call.name == "query_refund_status" else None
    observation = _check(
        getattr(adapter, TOOLS[call.name][0])(access, binding, call.id), binding
    )
    if (
        observation.tool_call_id != call.id
        or observation.tool_name != call.name
        or observation.arguments_hash != canonical_hash({})
    ):
        raise RefundFlowError("refund_observation_conflict")
    fields = {k: v.model_dump(mode="json") for k, v in observation.fields.items()}
    if observation.availability == "available":
        fields["authorized"] = allowed.model_dump(mode="json")
        if call.name == "query_refund_order":
            fields["subject_ref"] = {
                "availability": "available",
                "value": access.subject_ref,
            }
    if call.name == "query_refund_status":
        fields = {
            "authorized": allowed.model_dump(mode="json"),
            "local_application": {
                "availability": "available",
                "value": local.model_dump(mode="json") if local else None,
            },
            "external_status": {
                "availability": observation.availability,
                "value": fields if observation.availability == "available" else None,
            },
        }
    return _reframe(
        observation,
        call,
        binding,
        fields=fields,
        availability="available" if call.name == "query_refund_status" else None,
    )
