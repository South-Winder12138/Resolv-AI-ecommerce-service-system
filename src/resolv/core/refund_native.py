"""Native terminal return schema; deliberately has no business executor."""

from resolv.core.refund_prompts import NATIVE_DECISION_PROMPT_VERSION
from resolv.core.understanding_model import compact_schema
from resolv.schemas.refund_flow import RefundDecisionOutput
from resolv.services.refund_tools import refund_tool_schemas

RETURN_DECISION = "return_refund_decision"


def uses_native_return(profile):
    return profile.prompt_version == NATIVE_DECISION_PROMPT_VERSION


def native_decision_schema():
    schema = RefundDecisionOutput.model_json_schema()
    schema["properties"]["rules"]["description"] = (
        "只放实际政策规则，每项必须有真实条款引用及对应事实。"
        "身份、权限、金额和已有申请是系统前提，不是政策规则，不得编造引用。"
    )
    schema["properties"]["critical_fields"]["description"] = (
        "实际关键事实引用，包含政策事实及身份、权限、金额、已有申请等系统前提；"
        "系统前提不要求编造政策规则。"
    )
    schema["properties"]["recheck_conditions"]["description"] = (
        "仅记录实际条款明确定义的比较。关键时限口径模糊时不能省略条件而声称符合，"
        "应输出undetermined/human_draft并说明未确定口径。"
    )
    return schema


def return_descriptor(profile):
    return {
        "type": "function",
        "function": {
            "name": RETURN_DECISION,
            "description": "返回当前订单的最终结构化判断；只返回数据，不执行或提交任何业务。",
            "strict": True,
            "parameters": compact_schema(profile.output_schema.value),
        },
    }


def decision_tools(profile, *, require_tool=False, final_round=False):
    reads = refund_tool_schemas()
    if not uses_native_return(profile):
        return reads
    if final_round:
        return (return_descriptor(profile),)
    return reads if require_tool else (*reads, return_descriptor(profile))
