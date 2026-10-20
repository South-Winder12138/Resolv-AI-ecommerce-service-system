"""One refund-specific preparation call, with bounded model-only fallback."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund import OrderBinding, RefundFlowError
from resolv.schemas.refund_flow import (
    PreparedRefundQuery,
    RefundQueryOutput,
    RefundRequest,
)
from resolv.services.hybrid_fingerprints import canonical_hash

MODEL_FAILURE_CODES = frozenset(
    {
        "refund_model_unavailable",
        "refund_model_output_invalid",
        "refund_model_output_budget_exceeded",
        "refund_model_usage_unknown",
        "refund_model_input_underestimated",
    }
)


def prepare_refund_query(
    request, primary, binding, *, client, profile, query_token_count, audit_hook
):
    request = RefundRequest.model_validate_json(request.model_dump_json())
    p = PrimaryExecution.model_validate_json(primary.model_dump_json())
    binding = OrderBinding.model_validate_json(binding.model_dump_json())
    if (
        p.primary.branch != "refund_flow"
        or p.understanding_id != binding.understanding_id
        or p.primary.task_id != binding.primary_task_id
        or canonical_hash(p.model_dump(mode="json")) != binding.primary_hash
        or profile.adapter.namespace != binding.order_source
    ):
        raise RefundFlowError("refund_identity_conflict")
    data = {
        "primary_text": p.primary.text,
        "request": request.model_dump(mode="json"),
        "binding": binding.model_dump(mode="json"),
        "primary_ref": {
            "task_id": p.primary.task_id,
            "understanding_id": p.understanding_id,
            "snapshot_hash": p.snapshot_hash,
        },
    }
    input_hash = canonical_hash(data)
    messages = [
        SystemMessage(content=profile.query_model.system_prompt),
        HumanMessage(content=canonical_json(data)),
    ]
    fallback_reason = None
    try:
        output = client.prepare(
            messages, profile=profile.query_model, audit_hook=audit_hook
        )
    except RefundFlowError as exc:
        if exc.code not in MODEL_FAILURE_CODES:
            raise
        fallback_reason = exc.code
    except TimeoutError:
        fallback_reason = "refund_model_unavailable"
    else:
        try:
            output = RefundQueryOutput.model_validate_json(output.model_dump_json())
            standard = output.standard.strip()
            if not standard:
                raise ValueError("Empty standard")
            expanded = tuple(
                dict.fromkeys(
                    s.strip()
                    for s in output.expanded
                    if s.strip() and s.strip() != standard
                )
            )
        except (ValidationError, ValueError):
            fallback_reason = "refund_model_output_invalid"
    if fallback_reason:
        standard, expanded = p.primary.text, ()

    def count(text):
        result = query_token_count(text)
        if type(result) is not int or result < 0:
            raise RefundFlowError("refund_query_token_count_invalid")
        return result

    try:
        tokens = count(standard)
    except HybridRetrievalError as exc:
        if exc.code != "query_too_long":
            raise
        if fallback_reason:
            raise RefundFlowError("refund_query_too_long") from exc
        tokens = profile.retrieval_profile.query_max_tokens + 1
    if tokens > profile.retrieval_profile.query_max_tokens and not fallback_reason:
        fallback_reason = "refund_query_model_too_long"
        standard, expanded = p.primary.text, ()
        try:
            tokens = count(standard)
        except HybridRetrievalError as exc:
            if exc.code != "query_too_long":
                raise
            raise RefundFlowError("refund_query_too_long") from exc
    if tokens > profile.retrieval_profile.query_max_tokens:
        raise RefundFlowError("refund_query_too_long")
    return PreparedRefundQuery(
        standard=standard,
        expanded=expanded,
        bm25_text="\n".join((standard, *expanded)),
        status="fallback" if fallback_reason else "prepared",
        fallback_reason=fallback_reason,
        input_hash=input_hash,
    )
