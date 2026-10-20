"""Complete evidence context, frozen message selection and citation existence checks."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from resolv.core.config import ContextSettings
from resolv.core.knowledge_branch_model import knowledge_request_parameters
from resolv.schemas.knowledge_branch import (
    FrozenJSON,
    KnowledgeAnswer,
    KnowledgeAnswerInput,
    KnowledgeBranchError,
    KnowledgeEvidenceBundle,
    canonical_json,
)
from resolv.services.context import ContextBudgetExceeded, ContextBuilder, ContextInput
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.knowledge_branch_query import (
    MODEL_FAILURE_CODES,
    validated_knowledge_primary,
)


def knowledge_context_builder(profile, counter):
    model = profile.answer_model
    controls = counter.count_text(canonical_json(knowledge_request_parameters(model)))
    if type(controls) is not int or controls < 0 or controls >= model.input_limit:
        raise KnowledgeBranchError("answer_context_budget_exceeded", True)
    return ContextBuilder(
        ContextSettings(
            _env_file=None,
            tokenizer_profile=model.tokenizer_profile,
            model_context_window_tokens=model.model_context_window_tokens,
            reserved_output_tokens=model.reserved_output_tokens,
            safety_margin_tokens=model.safety_margin_tokens,
            app_max_input_tokens=model.input_limit - controls,
            provider_input_overhead_tokens=model.provider_input_overhead_tokens,
            recent_history_target_tokens=model.recent_history_target_tokens,
            knowledge_context_max_tokens=profile.evidence_max_tokens,
        ),
        counter,
    ), controls


def knowledge_answer_question(primary, query):
    p = validated_knowledge_primary(primary)
    return canonical_json(
        {
            "primary_text": p.primary.text,
            "standard": query.standard,
            "primary_ref": {
                "task_id": p.primary.task_id,
                "understanding_id": p.understanding_id,
            },
            "original_quotes": [
                r.model_dump(mode="json") for r in p.primary.evidence_refs
            ],
        }
    )


def build_knowledge_answer_input(
    run, primary, query, bundle, *, summary, history, profile, counter
):
    p = validated_knowledge_primary(primary)
    bundle = KnowledgeEvidenceBundle.model_validate_json(bundle.model_dump_json())
    if (
        run.conversation_id != p.conversation_id
        or run.understanding_id != p.understanding_id
        or query.input_hash != canonical_hash(p.model_dump(mode="json"))
        or counter.count_text(bundle.serialized) != bundle.input_tokens
    ):
        raise KnowledgeBranchError("knowledge_answer_input_conflict")
    builder, controls = knowledge_context_builder(profile, counter)
    question = knowledge_answer_question(p, query)
    retained = list(history)
    while True:
        try:
            result = builder.build(
                ContextInput(
                    profile.answer_model.system_prompt,
                    summary,
                    retained,
                    [HumanMessage(content=question)],
                    [bundle.payload],
                    [],
                )
            )
            break
        except ContextBudgetExceeded:
            if retained:
                retained.pop(0)
            elif summary is not None:
                summary = None
            else:
                raise KnowledgeBranchError(
                    "answer_context_budget_exceeded", True
                ) from None
    roles = {"system": "system", "human": "user", "ai": "assistant"}
    messages = [{"role": roles[m.type], "content": m.content} for m in result.messages]
    return KnowledgeAnswerInput(
        messages=FrozenJSON.from_value(messages),
        input_estimate=result.input_tokens + controls,
        input_limit=profile.answer_model.input_limit,
        input_hash=canonical_hash(messages),
        evidence_hash=bundle.evidence_hash,
        history_refs=tuple(
            (m.message_id, canonical_hash(m.model_dump(mode="json")))
            for turn in retained
            for m in (turn.user, turn.assistant)
        ),
        summary_ref=(
            f"{summary.conversation_id}:{summary.version}",
            canonical_hash(summary.model_dump(mode="json")),
        )
        if summary
        else None,
    )


def generate_knowledge_answer(input, bundle, *, client, profile, audit_hook):
    input = KnowledgeAnswerInput.model_validate_json(input.model_dump_json())
    bundle = KnowledgeEvidenceBundle.model_validate_json(bundle.model_dump_json())
    if input.evidence_hash != bundle.evidence_hash:
        raise KnowledgeBranchError("knowledge_answer_input_conflict")
    types = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    try:
        messages = [
            types[m["role"]](content=m["content"]) for m in input.messages.value
        ]
    except (KeyError, TypeError) as exc:
        raise KnowledgeBranchError("knowledge_answer_input_conflict") from exc
    try:
        reply = client.invoke_answer(
            messages, profile=profile.answer_model, audit_hook=audit_hook
        )
    except KnowledgeBranchError as exc:
        if exc.code not in MODEL_FAILURE_CODES:
            raise
        raise KnowledgeBranchError("answer_generation_error", True) from exc
    if (
        not isinstance(reply, AIMessage)
        or not isinstance(reply.content, str)
        or not reply.content.strip()
        or reply.tool_calls
        or reply.invalid_tool_calls
        or reply.response_metadata.get("finish_reason") not in {"stop", "completed"}
        or reply.response_metadata.get("status") in {"incomplete", "failed"}
    ):
        raise KnowledgeBranchError("answer_generation_error", True)
    markers = re.findall(r"\[([0-9]+)\]", reply.content)
    if not markers or any(
        m.startswith("0") or not 1 <= int(m) <= len(bundle.citations) for m in markers
    ):
        raise KnowledgeBranchError("answer_citation_invalid", True)
    used = tuple(sorted({int(m) for m in markers}))
    return KnowledgeAnswer(
        answer=reply.content,
        used_citation_numbers=used,
        evidence_hash=bundle.evidence_hash,
        input_hash=input.input_hash,
    )
