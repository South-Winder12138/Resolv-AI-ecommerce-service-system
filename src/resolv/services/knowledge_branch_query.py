"""Prepare one validated primary; only explicit model/format failures may fall back."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from resolv.core.knowledge_branch_prompts import KnowledgeQueryOutput
from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.schemas.knowledge_branch import (
    KnowledgeBranchError,
    PreparedKnowledgeQuery,
    canonical_json,
)
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.understanding_rules import INTENT_BRANCHES_V1

MODEL_FAILURE_CODES = frozenset(
    {
        "knowledge_model_unavailable",
        "knowledge_model_output_invalid",
        "knowledge_model_output_budget_exceeded",
        "knowledge_model_usage_unknown",
        "knowledge_model_input_underestimated",
    }
)


def validated_knowledge_primary(primary):
    try:
        p = PrimaryExecution.model_validate_json(primary.model_dump_json())
        if (
            p.primary.branch != "knowledge"
            or INTENT_BRANCHES_V1.get(p.primary.intent) != "knowledge"
        ):
            raise ValueError("Knowledge primary required")
        return p
    except (ValidationError, ValueError, TypeError) as exc:
        raise KnowledgeBranchError("knowledge_primary_invalid") from exc


def prepare_knowledge_query(primary, *, client, profile, query_token_count, audit_hook):
    p = validated_knowledge_primary(primary)
    input_hash = canonical_hash(p.model_dump(mode="json"))
    messages = [
        SystemMessage(content=profile.query_model.system_prompt),
        HumanMessage(
            content=canonical_json(
                {
                    "primary_text": p.primary.text,
                    "bindings": [
                        {"entity_type": b.entity_type, "value": b.value}
                        for b in p.primary.bindings
                    ],
                    "primary_ref": {
                        "task_id": p.primary.task_id,
                        "understanding_id": p.understanding_id,
                        "snapshot_hash": p.snapshot_hash,
                    },
                }
            )
        ),
    ]
    fallback_reason = None
    try:
        output = client.invoke_json(
            "knowledge_query",
            messages,
            KnowledgeQueryOutput,
            profile=profile.query_model,
            audit_hook=audit_hook,
        )
    except KnowledgeBranchError as exc:
        if exc.code not in MODEL_FAILURE_CODES:
            raise
        fallback_reason = exc.code
    except TimeoutError:
        fallback_reason = "knowledge_query_model_invalid"
    else:
        try:
            output = KnowledgeQueryOutput.model_validate_json(output.model_dump_json())
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
            fallback_reason = "knowledge_query_model_invalid"
    if fallback_reason:
        standard, expanded = p.primary.text, ()

    def count(text):
        tokens = query_token_count(text)
        if type(tokens) is not int or tokens < 0:
            raise KnowledgeBranchError("query_token_count_invalid")
        return tokens

    try:
        tokens = count(standard)
    except HybridRetrievalError as exc:
        if exc.code != "query_too_long" or fallback_reason:
            raise
        tokens = profile.retrieval_profile.query_max_tokens + 1
    if tokens > profile.retrieval_profile.query_max_tokens and not fallback_reason:
        standard, expanded = p.primary.text, ()
        fallback_reason = "knowledge_query_model_invalid"
        tokens = count(standard)
    if type(tokens) is not int or tokens < 0:
        raise KnowledgeBranchError("query_token_count_invalid")
    if tokens > profile.retrieval_profile.query_max_tokens:
        raise KnowledgeBranchError("query_too_long")
    return PreparedKnowledgeQuery(
        standard=standard,
        expanded=expanded,
        bm25_text="\n".join((standard, *expanded)),
        status="fallback" if fallback_reason else "prepared",
        fallback_reason=fallback_reason,
        input_hash=input_hash,
    )
