"""Apply the code gate, then exactly one evidence-only sufficiency judgment."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from resolv.schemas.knowledge_branch import (
    KnowledgeBranchError,
    KnowledgeDecision,
    KnowledgeEvidenceBundle,
    SufficiencyResult,
    canonical_json,
)
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.knowledge_branch_query import (
    MODEL_FAILURE_CODES,
    validated_knowledge_primary,
)


def check_knowledge_evidence(
    primary, query, bundle, signals, *, client, profile, audit_hook
):
    p = validated_knowledge_primary(primary)
    bundle = KnowledgeEvidenceBundle.model_validate_json(bundle.model_dump_json())
    if (
        query.input_hash != canonical_hash(p.model_dump(mode="json"))
        or signals.raw_logits != tuple(w.rerank_score for w in bundle.windows)
        or signals.confidence_threshold != profile.confidence_threshold
        or signals.effective_score_threshold != profile.effective_score_threshold
    ):
        raise KnowledgeBranchError("knowledge_check_input_conflict")
    if not signals.passed:
        return KnowledgeDecision(
            signals=signals,
            self_check=None,
            evidence_strong=False,
            source="retrieval_low_conf",
            evidence_hash=bundle.evidence_hash,
        )
    messages = [
        SystemMessage(content=profile.check_model.system_prompt),
        HumanMessage(
            content=canonical_json(
                {
                    "primary_text": p.primary.text,
                    "standard": query.standard,
                    "evidence_hash": bundle.evidence_hash,
                    "allowed_supporting_window_ids": [
                        w.window_id for w in bundle.windows
                    ],
                    "evidence": bundle.payload,
                }
            )
        ),
    ]
    try:
        output = client.invoke_json(
            "knowledge_check",
            messages,
            SufficiencyResult,
            profile=profile.check_model,
            audit_hook=audit_hook,
        )
    except KnowledgeBranchError as exc:
        if exc.code not in MODEL_FAILURE_CODES:
            raise
        raise KnowledgeBranchError("self_check_error", True) from exc
    except TimeoutError as exc:
        raise KnowledgeBranchError("self_check_error", True) from exc
    try:
        output = SufficiencyResult.model_validate_json(output.model_dump_json())
        if not set(output.supporting_window_ids).issubset(
            {w.window_id for w in bundle.windows}
        ):
            raise ValueError("Unknown supporting window")
    except (ValidationError, ValueError) as exc:
        raise KnowledgeBranchError("self_check_error", True) from exc
    return KnowledgeDecision(
        signals=signals,
        self_check=output,
        evidence_strong=output.useful,
        source=None if output.useful else "self_check",
        evidence_hash=bundle.evidence_hash,
    )
