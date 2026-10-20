"""Four-signal arithmetic and literal, document-scoped evidence matching."""

import math

import pytest

from resolv.schemas.knowledge import KnowledgeFileConfig
from resolv.schemas.knowledge_branch import KeywordHit, KnowledgeBranchError
from resolv.services.knowledge_confidence import (
    confidence_from_probabilities,
    score_knowledge_evidence,
    stable_sigmoid,
)
from tests.knowledge_branch_fixtures import branch_profile, bundle


def hit():
    return KeywordHit(
        window_id="a" * 64,
        document_id="d" * 64,
        keyword="退货",
        field="title",
        position=0,
    )


def test_empty_evidence_has_zero_signals():
    s = score_knowledge_evidence(bundle(()), file_configs={}, profile=branch_profile())
    assert s.raw_logits == s.probabilities == ()
    assert (s.effective_count, s.quantity, s.gap, s.total, s.passed) == (
        0,
        0,
        0,
        0,
        False,
    )
    assert s.contributions == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("logit", "expected"), [(-1000.0, 0.0), (1000.0, 1.0), (0.0, 0.5)]
)
def test_stable_sigmoid_extremes(logit, expected):
    assert stable_sigmoid(logit) == expected


def test_four_hand_calculated_contributions_and_quantity_cap():
    s = confidence_from_probabilities(
        (0.8, 0.5, 0.3, 0.1), keyword_hits=(hit(),), profile=branch_profile()
    )
    assert (s.effective_count, s.quantity) == (3, 1.0)
    assert s.gap == pytest.approx(0.3)
    assert s.contributions == pytest.approx((0.4, 0.2, 0.06, 0.1))
    assert s.total == pytest.approx(0.76)
    assert s.passed
    capped = confidence_from_probabilities(
        (0.8, 0.7, 0.6, 0.5), keyword_hits=(), profile=branch_profile()
    )
    assert capped.effective_count == 4 and capped.quantity == 1.0


def test_threshold_includes_exact_boundary_and_excludes_just_below():
    assert confidence_from_probabilities(
        (0.26, 0.11), keyword_hits=(hit(),), profile=branch_profile()
    ).passed
    below = confidence_from_probabilities(
        (0.259999, 0.11), keyword_hits=(hit(),), profile=branch_profile()
    )
    assert below.total < 0.26 and not below.passed
    s = confidence_from_probabilities(
        (0.3, math.nextafter(0.3, 0.0)), keyword_hits=(), profile=branch_profile()
    )
    assert s.effective_count == 1 and s.quantity == pytest.approx(1 / 3)


def test_single_window_gap_is_zero():
    s = confidence_from_probabilities((0.8,), keyword_hits=(), profile=branch_profile())
    assert s.gap == 0.0 and s.contributions[2] == 0.0


@pytest.mark.parametrize(
    "scores", [(0.2, 0.8), (float("nan"),), (float("inf"),), (-0.1,), (1.1,), (True,)]
)
def test_invalid_probability_sequences_are_technical_errors(scores):
    with pytest.raises(KnowledgeBranchError, match="confidence_scores_invalid"):
        confidence_from_probabilities(scores, keyword_hits=(), profile=branch_profile())


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, "1"])
def test_nonfinite_or_non_numeric_logit_is_rejected(value):
    with pytest.raises(KnowledgeBranchError, match="confidence_scores_invalid"):
        stable_sigmoid(value)


def test_literal_unicode_title_and_body_positions_and_shared_windows():
    b = bundle((2.0, 1.0), overlapping=True)
    s = score_knowledge_evidence(
        b,
        file_configs={
            "d" * 64: KnowledgeFileConfig(
                content_type="policy", key_clause_keywords=("退货", "七天", "Policy")
            )
        },
        profile=branch_profile(),
    )
    assert s.raw_logits == (2.0, 1.0)
    assert s.probabilities == pytest.approx((0.880797077978, 0.731058578630))
    assert s.effective_count == 2 and s.quantity == pytest.approx(2 / 3)
    assert len(s.shared_chunk_ids) == 2
    title_hits = [h for h in s.keyword_hits if h.field == "title"]
    assert len(title_hits) == 2 and all(
        h.keyword == "退货" and h.position == 0 for h in title_hits
    )
    body_hits = [h for h in s.keyword_hits if h.keyword == "七天"]
    assert len(body_hits) == 6 and all(
        h.field == "body" and h.position == 4 for h in body_hits
    )
    assert s.contributions[3] == 0.1
    assert not any(h.keyword == "Policy" for h in s.keyword_hits)


def test_other_document_keywords_cannot_contribute():
    s = score_knowledge_evidence(
        bundle((1.0,)),
        file_configs={
            "d" * 64: KnowledgeFileConfig(
                content_type="policy", key_clause_keywords=("POLICY",)
            ),
            "e" * 64: KnowledgeFileConfig(
                content_type="policy", key_clause_keywords=("退货",)
            ),
        },
        profile=branch_profile(),
    )
    assert s.keyword_hits == () and s.contributions[3] == 0.0


def test_missing_file_config_is_failure_not_low_confidence():
    with pytest.raises(KnowledgeBranchError, match="confidence_file_config_missing"):
        score_knowledge_evidence(bundle(), file_configs={}, profile=branch_profile())


def test_only_first_three_windows_can_contribute_keywords():
    b = bundle((4.0, 3.0, 2.0, 1.0))
    from resolv.schemas.hybrid_knowledge import HybridRetrievalResult
    from resolv.services.knowledge_branch_evidence import select_knowledge_evidence
    from tests.knowledge_branch_fixtures import hybrid_result
    from tests.test_knowledge_branch_evidence import Characters

    raw = hybrid_result((4.0, 3.0, 2.0, 1.0))
    for cid in raw.windows[3].chunk_ids:
        body = raw.chunks[cid]
        raw.chunks[cid] = body.model_copy(
            update={"chunk": body.chunk.model_copy(update={"text": "唯一关键字"})}
        )
    raw = HybridRetrievalResult.model_validate_json(raw.model_dump_json())
    b = select_knowledge_evidence(raw, counter=Characters(), max_tokens=100000)
    s = score_knowledge_evidence(
        b,
        file_configs={
            "d" * 64: KnowledgeFileConfig(
                content_type="policy", key_clause_keywords=("唯一关键字",)
            )
        },
        profile=branch_profile(),
    )
    assert s.keyword_hits == () and s.contributions[3] == 0.0
