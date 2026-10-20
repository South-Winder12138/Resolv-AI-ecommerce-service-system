"""One offline-capable decoder shared by live responses and failure replay."""

import json

from langchain_core.messages import AIMessage

from resolv.core.refund_native import RETURN_DECISION, uses_native_return
from resolv.core.refund_prompts import (
    COMPLETE_DECISION_PROMPT_VERSION,
    CONTEXT_IDENTITY_PROMPT_VERSION,
)
from resolv.core.understanding_model import public_text
from resolv.schemas.knowledge_branch import FrozenJSON, canonical_json
from resolv.schemas.refund import RefundFlowError, unique_object
from resolv.schemas.refund_flow import (
    RefundDecision,
    RefundDecisionOutput,
    RefundQueryOutput,
    RefundToolCall,
)
from resolv.services.refund_tools import TOOLS


def strict_json(text):
    def invalid(_):
        raise ValueError("Nonfinite JSON")

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid)


class ResponseDecodeError(RefundFlowError):
    def __init__(self, cause, stage, usage, finish):
        error = (
            cause
            if isinstance(cause, RefundFlowError)
            else RefundFlowError("refund_model_output_invalid", True)
        )
        super().__init__(error.code, error.retryable)
        self.cause, self.stage, self.usage, self.finish = cause, stage, usage, finish


def decode_refund_response(
    reply,
    profile,
    *,
    prepare,
    require_tool,
    final_round,
    counter,
    estimate,
    required_tool=None,
):
    usage = (reply.usage_metadata or {}) if isinstance(reply, AIMessage) else {}
    finish, stage = None, "response"
    try:
        if not isinstance(reply, AIMessage) or reply.invalid_tool_calls:
            raise RefundFlowError("refund_model_output_invalid", True)
        stage = "protocol"
        metadata = reply.response_metadata
        finish = metadata.get("finish_reason")
        if (
            profile.model_api_mode == "responses"
            and metadata.get("status") == "completed"
        ):
            finish = "completed"
        calls = reply.tool_calls
        if finish not in (
            {"stop", "completed"}
            if prepare or not calls
            else {"tool_calls", "completed"}
        ) or metadata.get("status") in {"failed", "incomplete"}:
            raise RefundFlowError("refund_model_output_invalid", True)
        stage = "usage"
        if any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("input_tokens", "output_tokens", "total_tokens")
        ) or usage.get("total_tokens") != usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        ):
            raise RefundFlowError("refund_model_usage_unknown", True)
        if usage["input_tokens"] > estimate:
            raise RefundFlowError("refund_model_input_underestimated", True)
        stage = "content"
        content = reply.content
        if isinstance(content, list) and calls:
            content = [
                block
                for block in content
                if not isinstance(block, dict)
                or block.get("type") not in {"function_call", "tool_call"}
            ]
        text = public_text(AIMessage(content=content, response_metadata=metadata))
        stage = "budget"
        output_text = canonical_json({"text": text, "tool_calls": calls})
        if (
            usage["output_tokens"] > profile.reserved_output_tokens
            or counter.count_text(output_text) > profile.reserved_output_tokens
        ):
            raise RefundFlowError("refund_model_output_budget_exceeded", True)
        stage = "tools"
        if prepare:
            if calls or reply.additional_kwargs.get("tool_calls") or not text.strip():
                raise RefundFlowError("refund_model_output_invalid", True)
        else:
            native = uses_native_return(profile)
            if (
                final_round
                and not native
                and (calls or reply.additional_kwargs.get("tool_calls"))
            ):
                raise RefundFlowError("refund_model_output_invalid", True)
            chunks = getattr(reply, "tool_call_chunks", [])
            if len(chunks) != len(calls) or len({c.get("id") for c in calls}) != len(
                calls
            ):
                raise RefundFlowError("refund_model_output_invalid", True)
            terminal = [c for c in calls if c.get("name") == RETURN_DECISION]
            if native and terminal:
                if require_tool or len(calls) != 1 or text.strip():
                    raise RefundFlowError("refund_model_output_invalid", True)
                call, chunk = calls[0], chunks[0]
                if (
                    not call.get("id")
                    or chunk.get("id") != call["id"]
                    or chunk.get("name") != RETURN_DECISION
                ):
                    raise RefundFlowError("refund_model_output_invalid", True)
                stage = "json"
                parsed = strict_json(chunk["args"])
                if parsed != call.get("args"):
                    raise RefundFlowError("refund_model_output_invalid", True)
                stage = "contract"
                value = RefundDecisionOutput.model_validate_json(canonical_json(parsed))
                return (
                    ((), RefundDecision.model_validate_json(value.model_dump_json())),
                    usage,
                    finish,
                )
            if native and (final_round or not calls or text.strip()):
                raise RefundFlowError("refund_model_output_invalid", True)
            for chunk, call in zip(chunks, calls):
                parsed = strict_json(chunk["args"])
                if (
                    parsed != {}
                    or call.get("args") != parsed
                    or not call.get("id")
                    or call.get("name") not in TOOLS
                    or chunk.get("id") != call["id"]
                    or chunk.get("name") != call["name"]
                ):
                    raise RefundFlowError("refund_model_output_invalid", True)
                if native and require_tool and call["name"] != required_tool:
                    raise RefundFlowError("refund_model_output_invalid", True)
            if require_tool and not calls:
                raise RefundFlowError("refund_model_tool_required", True)
            if calls:
                return (
                    (
                        tuple(
                            RefundToolCall(
                                id=c["id"],
                                name=c["name"],
                                args=FrozenJSON.from_value(c["args"]),
                            )
                            for c in calls
                        ),
                        None,
                    ),
                    usage,
                    finish,
                )
        stage = "json"
        parsed = canonical_json(strict_json(text))
        stage = "contract"
        output_type = (
            RefundQueryOutput
            if prepare
            else (
                RefundDecisionOutput
                if profile.prompt_version
                in {COMPLETE_DECISION_PROMPT_VERSION, CONTEXT_IDENTITY_PROMPT_VERSION}
                else RefundDecision
            )
        )
        value = output_type.model_validate_json(parsed)
        if not prepare and output_type is RefundDecisionOutput:
            value = RefundDecision.model_validate_json(value.model_dump_json())
        return (value if prepare else ((), value)), usage, finish
    except Exception as exc:  # noqa: BLE001 -- retain precise cause privately, public code stays stable
        raise ResponseDecodeError(exc, stage, usage, finish) from None
