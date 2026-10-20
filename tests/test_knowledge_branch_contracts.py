"""Contract boundary tests: reject changed identity, scores and mutable snapshots."""

import json

import pytest
from pydantic import ValidationError

from resolv.schemas.hybrid_knowledge import FrozenFileConfig, HybridRetrievalProfile
from resolv.schemas.knowledge_branch import (
    ConfidenceSignals,
    KnowledgeEvidenceBundle,
    KnowledgeRetrievalStage,
    PreparedKnowledgeQuery,
    SufficiencyResult,
)
from resolv.schemas.knowledge_release import (
    GateReceipt,
    ReleaseAcceptance,
    ReleaseSelection,
)
from resolv.services.hybrid_fingerprints import canonical_hash, profile_hash
from tests.hybrid_fixtures import profile
from tests.knowledge_branch_fixtures import branch_profile, bundle, hybrid_result


def test_profiles_and_nested_results_are_immutable():
    p = branch_profile()
    assert (p.evidence_max_tokens, p.queue_timeout_seconds) == (3000, 60)
    assert (p.effective_score_threshold, p.confidence_threshold) == (0.3, 0.26)
    assert p.score_weights == (0.5, 0.2, 0.2, 0.1)
    assert (p.retrieval_profile.device, p.retrieval_profile.batch_size) == ("mps", 4)
    assert (
        p.retrieval_profile.query_max_tokens,
        p.retrieval_profile.pair_max_tokens,
    ) == (512, 3072)
    with pytest.raises(ValidationError):
        p.query_model.model_name = "changed"
    with pytest.raises(ValidationError):
        type(p).model_validate(p.model_dump() | {"api_key": "secret"})
    with pytest.raises(ValidationError):
        p.model_copy(update={"confidence_threshold": float("nan")})
    assert type(p).model_validate_json(p.model_dump_json()) == p


def test_hybrid_snapshot_copies_and_checks_hashes():
    raw = hybrid_result()
    stage = KnowledgeRetrievalStage.from_result(
        raw,
        bundle=bundle(),
        file_configs=(("d" * 64, FrozenFileConfig(content_type="policy")),),
    )
    raw.chunks.clear()
    raw.diagnostics.stage_seconds["mutated"] = 1.0
    assert len(stage.result.chunks) == 3
    copy = stage.result
    copy.chunks.clear()
    assert len(stage.result.chunks) == 3
    assert "mutated" not in stage.result.diagnostics.stage_seconds
    with pytest.raises(ValidationError):
        stage.model_copy(update={"result_hash": "0" * 64})
    assert KnowledgeRetrievalStage.model_validate_json(stage.model_dump_json()) == stage


def test_bundle_payload_is_a_copy_and_score_belongs_to_window():
    b = bundle((1.0, 0.5), overlapping=True)
    public = b.payload
    public["chunks"].clear()
    assert len(b.payload["chunks"]) == 4
    assert b.citations[0].rerank_score == 1.0
    assert all(
        "score" not in row and "rerank_score" not in row
        for row in b.payload["chunks"].values()
    )
    assert KnowledgeEvidenceBundle.model_validate_json(b.model_dump_json()) == b


@pytest.mark.parametrize(
    "change", ["orphan", "section", "duplicate", "score", "hash", "text"]
)
def test_bundle_closure_and_window_score_scope(change):
    b = bundle((1.0, 0.5), overlapping=True)
    data = json.loads(b.model_dump_json())
    if change == "orphan":
        data["chunks"].pop()
    elif change == "section":
        data["chunks"][0]["section_instance_id"] = "0" * 64
    elif change == "duplicate":
        data["citations"][1]["citation_number"] = 1
    elif change == "score":
        data["windows"][0]["rerank_score"] = float("inf")
    elif change == "hash":
        data["evidence_hash"] = "0" * 64
    else:
        data["serialized"] += "changed"
    with pytest.raises(ValidationError):
        KnowledgeEvidenceBundle.model_validate_json(json.dumps(data))


def test_query_fallback_and_text_must_agree():
    q = PreparedKnowledgeQuery(
        standard="退货",
        expanded=("邮费",),
        bm25_text="退货\n邮费",
        status="prepared",
        input_hash="a" * 64,
    )
    with pytest.raises(ValidationError):
        q.model_copy(update={"bm25_text": "别的问题"})
    with pytest.raises(ValidationError):
        q.model_copy(update={"status": "fallback", "fallback_reason": "model_error"})


def test_selfcheck_cannot_claim_sufficient_with_missing_support():
    for values in [
        {"useful": True, "supporting_window_ids": (), "missing_information": ()},
        {
            "useful": True,
            "supporting_window_ids": ("a" * 64,),
            "missing_information": ("运费",),
        },
    ]:
        with pytest.raises(ValidationError):
            SufficiencyResult(**values, reason="缺关键依据")


def test_selection_requires_all_hashes_and_matching_profile():
    p = profile()
    good = {
        "status": "configured",
        "release_id": "knowledge-20261015-v1",
        "index_version": "knowledge_20261015_v1",
        "collection": p.collection_name,
        "profile_hash": profile_hash(p),
        "manifest_hash": "a" * 64,
        "profile": p,
    }
    assert ReleaseSelection(**good).status == "configured"
    for patch in [
        {"profile_hash": None},
        {"collection": "foreign"},
        {"profile_hash": "0" * 64},
        {"status": "disabled"},
    ]:
        with pytest.raises(ValidationError):
            ReleaseSelection(**(good | patch))


def test_acceptance_cannot_choose_required_gates_or_duplicate_receipts():
    assert ReleaseAcceptance.required_gates == frozenset(
        {"K1", "K2", "K3", "snapshot_python", "snapshot_frontend", "review"}
    )
    receipt = GateReceipt(
        name="K1",
        passed=True,
        report_sha256="a" * 64,
        source_fingerprints=(("src/file.py", "b" * 64),),
    )
    data = {
        "candidate_commit": "a" * 40,
        "policy_commit": "b" * 40,
        "source_trees": (("code", "c" * 40),),
        "release_id": "knowledge-20261015-v1",
        "index_version": "knowledge_20261015_v1",
        "profile_hash": "d" * 64,
        "manifest_hash": "e" * 64,
        "gate_receipts": (receipt, receipt),
        "review_important_count": 0,
    }
    with pytest.raises(ValidationError):
        ReleaseAcceptance(**data)
    with pytest.raises(ValidationError):
        GateReceipt(name="K1", passed=True, report_sha256="", source_fingerprints=())


def test_fingerprint_uses_existing_canonical_hash():
    p = branch_profile()
    assert p.fingerprint == canonical_hash(p.model_dump(mode="json"))


def test_unvalidated_legacy_model_copy_cannot_bypass_new_contracts():
    p = branch_profile()
    bad = HybridRetrievalProfile().model_copy(update={"device": "cpu"})
    with pytest.raises(ValidationError):
        type(p)(**(p.model_dump() | {"retrieval_profile": bad}))


def test_acceptance_requires_every_gate_and_no_important_findings():
    data = {
        "candidate_commit": "a" * 40,
        "policy_commit": "b" * 40,
        "source_trees": (("code", "c" * 40),),
        "release_id": "knowledge-20261015-v1",
        "index_version": "knowledge_20261015_v1",
        "profile_hash": "d" * 64,
        "manifest_hash": "e" * 64,
        "gate_receipts": (),
        "review_important_count": 0,
    }
    with pytest.raises(ValueError):
        ReleaseAcceptance(**data).require_passed()
    receipts = tuple(
        GateReceipt(
            name=name,
            passed=True,
            report_sha256="a" * 64,
            source_fingerprints=(("source", "b" * 64),),
        )
        for name in sorted(ReleaseAcceptance.required_gates)
    )
    ready = ReleaseAcceptance(**(data | {"gate_receipts": receipts}))
    ready.require_passed()
    with pytest.raises(ValueError):
        ready.model_copy(update={"review_important_count": 1}).require_passed()


def test_signals_reject_nonfinite_numbers():
    with pytest.raises(ValidationError):
        ConfidenceSignals(
            raw_logits=(float("nan"),),
            probabilities=(0.5,),
            effective_count=1,
            quantity=1 / 3,
            gap=0.0,
            keyword_hits=(),
            contributions=(0.25, 0.0, 0.0, 0.0),
            total=0.25,
            passed=False,
        )
