"""Versioned heuristic gate. Scores are signals, never correctness probabilities."""

import math
from collections import Counter
from itertools import pairwise

from pydantic import ValidationError

from resolv.schemas.hybrid_knowledge import FrozenFileConfig
from resolv.schemas.knowledge_branch import (
    ConfidenceSignals,
    KeywordHit,
    KnowledgeBranchError,
    KnowledgeBranchProfile,
    KnowledgeEvidenceBundle,
)


def stable_sigmoid(logit: float) -> float:
    if type(logit) not in (int, float) or not math.isfinite(logit):
        raise KnowledgeBranchError("confidence_scores_invalid")
    exp = math.exp(-logit) if logit >= 0 else math.exp(logit)
    return 1 / (1 + exp) if logit >= 0 else exp / (1 + exp)


def confidence_from_probabilities(
    scores: tuple[float, ...],
    *,
    keyword_hits: tuple,
    profile: KnowledgeBranchProfile,
) -> ConfidenceSignals:
    """Pure probability-input arithmetic; no raw logits are invented for this entry."""
    if (
        not isinstance(scores, tuple)
        or len(scores) > 10
        or any(
            type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1
            for s in scores
        )
        or any(a < b for a, b in pairwise(scores))
        or (not scores and keyword_hits)
    ):
        raise KnowledgeBranchError("confidence_scores_invalid")
    count = sum(s >= profile.effective_score_threshold for s in scores)
    quantity = min(count, 3) / 3
    gap = scores[0] - scores[1] if len(scores) >= 2 else 0.0
    weights = profile.score_weights
    contributions = (
        weights[0] * (scores[0] if scores else 0.0),
        weights[1] * quantity,
        weights[2] * gap,
        weights[3] * bool(keyword_hits),
    )
    total = math.fsum(contributions)
    return ConfidenceSignals(
        raw_logits=(),
        probabilities=tuple(float(s) for s in scores),
        effective_count=count,
        quantity=quantity,
        gap=gap,
        keyword_hits=keyword_hits,
        contributions=contributions,
        total=total,
        passed=total >= profile.confidence_threshold,
        effective_score_threshold=profile.effective_score_threshold,
        confidence_threshold=profile.confidence_threshold,
    )


def _positions(text, keyword):
    start = 0
    while (position := text.find(keyword, start)) >= 0:
        yield position
        start = position + 1


def score_knowledge_evidence(bundle, *, file_configs, profile) -> ConfidenceSignals:
    try:
        bundle = KnowledgeEvidenceBundle.model_validate_json(bundle.model_dump_json())
        configs = {}
        for w in bundle.windows:
            if w.document_id not in file_configs:
                raise KnowledgeBranchError("confidence_file_config_missing")
            config = file_configs[w.document_id]
            configs[w.document_id] = FrozenFileConfig.model_validate_json(
                config.model_dump_json()
            )
        bodies = {c.chunk.chunk_id: c.chunk for c in bundle.chunks}
        hits = []
        for w, citation in zip(bundle.windows[:3], bundle.citations[:3]):
            for keyword in configs[w.document_id].key_clause_keywords:
                fields = [("title", None, citation.title)] + [
                    ("body", cid, bodies[cid].text) for cid in w.chunk_ids
                ]
                for field, cid, text in fields:
                    hits.extend(
                        KeywordHit(
                            window_id=w.window_id,
                            document_id=w.document_id,
                            keyword=keyword,
                            field=field,
                            chunk_id=cid,
                            position=position,
                        )
                        for position in _positions(text, keyword)
                    )
        raw = tuple(w.rerank_score for w in bundle.windows)
        scores = tuple(stable_sigmoid(logit) for logit in raw)
        if any(c.sigmoid_score != s for c, s in zip(bundle.citations, scores)):
            raise KnowledgeBranchError("confidence_scores_invalid")
        shared = Counter(cid for w in bundle.windows for cid in w.chunk_ids)
        signals = confidence_from_probabilities(
            scores, keyword_hits=tuple(hits), profile=profile
        )
        return signals.model_copy(
            update={
                "raw_logits": raw,
                "shared_chunk_ids": tuple(
                    sorted(k for k, v in shared.items() if v > 1)
                ),
            }
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise KnowledgeBranchError("confidence_scores_invalid") from exc
