"""Select a complete ranked prefix; freeze the exact evidence text used downstream."""

import math

from pydantic import ValidationError

from resolv.schemas.hybrid_knowledge import HybridRetrievalResult
from resolv.schemas.knowledge_branch import (
    KnowledgeBranchError,
    KnowledgeCitation,
    KnowledgeEvidenceBundle,
    evidence_payload,
)
from resolv.services.context import TokenCounter, knowledge_text
from resolv.services.hybrid_fingerprints import canonical_hash


def _bundle(result, windows, counter):
    citations = []
    bodies = {}
    for i, window in enumerate(windows, 1):
        center = result.chunks[window.center_chunk_ids[0]].chunk
        logit = window.rerank_score
        exp = math.exp(-logit) if logit >= 0 else math.exp(logit)
        probability = 1 / (1 + exp) if logit >= 0 else exp / (1 + exp)
        citations.append(
            KnowledgeCitation(
                **window.model_dump(),
                citation_number=i,
                source=center.source,
                title=center.questions,
                section_path=center.section_path,
                index_version=result.index_version,
                index_profile_hash=result.index_profile_hash,
                sigmoid_score=probability,
            )
        )
        for cid in window.chunk_ids:
            bodies.setdefault(cid, result.chunks[cid])
    citations, chunks = tuple(citations), tuple(bodies.values())
    serialized = knowledge_text([evidence_payload(citations, chunks)])
    tokens = counter.count_text(serialized)
    if type(tokens) is not int or tokens < 0:
        raise KnowledgeBranchError("evidence_token_count_invalid")
    return KnowledgeEvidenceBundle(
        windows=tuple(windows),
        citations=citations,
        chunks=chunks,
        serialized=serialized,
        input_tokens=tokens,
        evidence_hash=canonical_hash(
            {"rule": "knowledge-evidence-v1", "text": serialized}
        ),
    )


def select_knowledge_evidence(
    result: HybridRetrievalResult,
    *,
    counter: TokenCounter,
    max_tokens: int = 3000,
) -> KnowledgeEvidenceBundle:
    if type(max_tokens) is not int or max_tokens <= 0:
        raise KnowledgeBranchError("evidence_budget_invalid")
    try:
        # Legacy Hybrid DTOs contain mutable dictionaries: never retain caller objects.
        result = HybridRetrievalResult.model_validate_json(result.model_dump_json())
        selected = _bundle(result, (), counter)
        if selected.input_tokens > max_tokens:
            raise KnowledgeBranchError("evidence_budget_exceeded")
        for i in range(1, len(result.windows) + 1):
            candidate = _bundle(result, result.windows[:i], counter)
            if candidate.input_tokens > max_tokens:
                if i == 1:
                    raise KnowledgeBranchError("evidence_budget_exceeded")
                break
            selected = candidate
        return selected
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        raise KnowledgeBranchError("knowledge_evidence_invalid") from exc


def serialize_knowledge_evidence(bundle: KnowledgeEvidenceBundle) -> str:
    checked = KnowledgeEvidenceBundle.model_validate_json(bundle.model_dump_json())
    return checked.serialized
