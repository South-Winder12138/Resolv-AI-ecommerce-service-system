"""A citation or a fake forward cannot replace the approved semantic evidence."""

import copy
import importlib
import json
from pathlib import Path

import pytest

from tests.refund_submission_fixtures import flow_harness, submission_harness

__all__ = ["flow_harness", "submission_harness"]


def test_C1_semantic_round_stops_before_consuming_later_cases():
    m = runner()
    invoked = []

    def failed(case):
        invoked.append(case["case_id"])
        return {"case_id": case["case_id"], "machine_passed": False}

    assert hasattr(m, "run_semantic_round"), "missing fail-fast C1 round"
    result = m.run_semantic_round(m.load_cases()["cases"], invoke=failed)
    assert invoked == ["F1"]
    assert len(result) == 8 and all(r["status"] == "not_executed" for r in result[1:])


def test_semantic_button_request_identity_is_scoped_to_original_offer():
    m = runner()
    assert hasattr(m, "semantic_submission_request_id"), (
        "missing per-offer command alias"
    )
    assert m.semantic_submission_request_id(
        "F1", "offer-a"
    ) == m.semantic_submission_request_id("F1", "offer-a")
    assert m.semantic_submission_request_id(
        "F1", "offer-a"
    ) != m.semantic_submission_request_id("F1", "offer-b")


def test_prepare_assembly_runs_actual_ingestion_and_indexer_with_real_descriptor(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from hashlib import sha256
    from types import SimpleNamespace

    from scripts import refund_gate_support as support

    from resolv.db import knowledge_hybrid, milvus
    from resolv.db.engine import create_db_engine
    from resolv.db.knowledge_base import KnowledgeBase
    from resolv.repositories import knowledge_hybrid_milvus
    from resolv.schemas.knowledge_release import ReleaseManifest
    from resolv.services import hybrid_embedding, knowledge_release

    m = runner()
    formal = ReleaseManifest.model_validate_json(
        Path("tests/fixtures/refund_flow/formal-policy-manifest.json").read_bytes()
    )
    blobs = {s.source: Path(s.original_path).read_bytes() for s in formal.documents}
    assert all(
        sha256(blobs[s.source]).hexdigest() == s.raw_sha256 for s in formal.documents
    )
    monkeypatch.setattr(
        knowledge_release, "_committed_manifest", lambda _: (formal, blobs, b"")
    )
    obj = {"owner": "a" * 32, "collection": "mhrefund_k_" + "a" * 16}
    policy = m.owned_policy(obj)
    monkeypatch.setattr(m, "owned_policy", lambda _: policy)
    monkeypatch.setattr(support, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    manifest = w / "resource-ownership.json"
    obj = support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    support.save_json(manifest, obj)
    engine = create_db_engine("sqlite:///" + str(tmp_path / "knowledge.db"))
    KnowledgeBase.metadata.create_all(engine)
    resources = support.RefundOwnedResources(manifest, obj, engine, None, None)

    @contextmanager
    def owned(_):
        yield resources

    monkeypatch.setattr(support, "owned_refund_resources", owned)
    client = SimpleNamespace(
        exists=False,
        has_collection=lambda **kw: client.exists,
        describe_collection=lambda **kw: {"collection_id": 123},
    )

    @contextmanager
    def connected(_):
        yield client

    monkeypatch.setattr(milvus, "milvus_client", connected)

    @contextmanager
    def locked(*a):
        yield SimpleNamespace(assert_owned=lambda: None)

    monkeypatch.setattr(knowledge_hybrid, "hybrid_index_lock", locked)
    encoded = []

    class Dense:
        dimension = 1024

        def __init__(self, profile):
            self.profile = profile

        def embed_documents(self, texts):
            encoded.extend(texts)
            return [[1.0] + [0.0] * 1023 for _ in texts]

        def close(self):
            pass

    class Store:
        def __init__(self, client, profile):
            self.profile = profile

        def create_or_validate(self):
            client.exists = True

        def validate_existing(self):
            pass

        def upsert(self, entries, vectors):
            return len(entries)

    monkeypatch.setattr(hybrid_embedding, "HybridEmbeddingAdapter", Dense)
    monkeypatch.setattr(knowledge_hybrid_milvus, "HybridMilvusStore", Store)
    try:
        result = m.prepare_owned(manifest, w / "unused")
        assert result["passed"] and result["build"]["indexed_count"] == 51
        assert (
            len(encoded) == 51
            and support.validate_manifest(manifest)["collection_id"] == "123"
        )
    finally:
        engine.dispose()


def test_owned_policy_rebinds_real_manifest_sources_without_git(monkeypatch):
    from hashlib import sha256

    from resolv.schemas.knowledge_release import ReleaseManifest
    from resolv.services import knowledge_release

    m = runner()
    manifest = ReleaseManifest.model_validate_json(
        Path("tests/fixtures/refund_flow/formal-policy-manifest.json").read_bytes()
    )
    blobs = {s.source: Path(s.original_path).read_bytes() for s in manifest.documents}
    assert all(
        sha256(blobs[s.source]).hexdigest() == s.raw_sha256 for s in manifest.documents
    )
    monkeypatch.setattr(
        knowledge_release, "_committed_manifest", lambda _: (manifest, blobs, b"")
    )
    owned, documents, corpus = m.owned_policy(
        {"owner": "a" * 32, "collection": "mhrefund_k_aaaaaaaaaaaaaaaa"}
    )
    assert owned.release_id == "refund_20261016_" + "a" * 16
    assert all(
        d.source.startswith("knowledge/releases/" + owned.release_id + "/")
        for d in documents
    )
    assert sum(len(d.chunks) for d in corpus.documents) == 51


def test_case_loader_is_offline_without_git_metadata(monkeypatch):
    from resolv.services import knowledge_release

    monkeypatch.setattr(
        knowledge_release,
        "_blob",
        lambda *a: (_ for _ in ()).throw(AssertionError("export has no .git")),
    )
    assert len(runner().load_cases()["cases"]) == 8


def test_fixed_rejected_seed_keeps_canonical_receipt_and_foreign_keys(tmp_path):
    from types import SimpleNamespace

    from resolv.db.engine import create_db_engine, create_session_factory
    from resolv.db.init_db import init_db
    from resolv.repositories.refund_application import RefundApplicationRepository
    from resolv.services.refund_registration import RefundRegistrationService
    from tests.refund_fixtures import binding

    m = runner()
    engine = create_db_engine("sqlite:///" + str(tmp_path / "business.db"))
    init_db(engine)
    try:
        factory = create_session_factory(engine)
        case = m.load_cases()["cases"][3]
        source = "refund-gate-owned"
        m.seed_rejected(SimpleNamespace(business_factory=factory), case, source)
        m.seed_rejected(SimpleNamespace(business_factory=factory), case, source)
        with factory() as session:
            record = RefundRegistrationService(session).lookup(
                binding(case["order_id"], source)
            )
            assert (
                record.application_id == "refund-F4-existing"
                and record.status == "rejected"
            )
            repo = RefundApplicationRepository(session)
            assert (
                repo.validated_receipt(repo.receipt("seed-F4")).application_id
                == record.application_id
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize("case_index", range(8))
def test_real_runtime_assembly_is_owned_offline_and_provider_lazy(
    tmp_path, monkeypatch, case_index
):
    from contextlib import ExitStack
    from types import SimpleNamespace

    from resolv.core import config
    from tests.refund_flow_fixtures import flow_case, selected

    m = runner()
    settings = config.Settings(
        _env_file=None,
        openai_api_key="offline",
        openai_base_url="https://example.aliyuncs.com/compatible-mode/v1",
        openai_model="qwen3.8-flash",
    )
    context = config.ContextSettings(
        _env_file=None,
        model_context_window_tokens=256000,
        tokenizer_profile="o200k_base",
    )
    summary = config.SummaryModelSettings(
        _env_file=None,
        model_name="gpt-6-sol",
        api_key="offline",
        base_url="https://summary.invalid/v1",
        model_context_window_tokens=256000,
    )
    monkeypatch.setattr(
        config, "Settings", lambda **kw: type(settings)(**kw) if kw else settings
    )
    monkeypatch.setattr(
        config, "ContextSettings", lambda **kw: type(context)(**kw) if kw else context
    )

    class OfflineSummary(type(summary)):
        def __init__(self, **kw):
            super().__init__(**(kw or {**summary.model_dump(), "_env_file": None}))

    monkeypatch.setattr(config, "SummaryModelSettings", OfflineSummary)
    monkeypatch.setattr(m, "owned_selection", lambda _: selected())
    monkeypatch.setattr(m, "verify_collection", lambda *a: None)
    monkeypatch.setattr(m, "make_pool", lambda _: SimpleNamespace(close=lambda: None))
    from scripts import refund_gate_support
    from sqlalchemy.engine import make_url

    urls = (
        make_url("mysql+pymysql://u:p@localhost/resolv_knowledge?charset=utf8mb4"),
        make_url("mysql+pymysql://u:p@localhost/resolv_conversations?charset=utf8mb4"),
    )
    monkeypatch.setattr(refund_gate_support, "configured_urls", lambda: urls)
    with flow_case(tmp_path) as h, ExitStack() as cleanup:
        resources = SimpleNamespace(
            identity={
                "owner": "a" * 32,
                "knowledge_database": "owned_k",
                "conversation_database": "owned_c",
            },
            conversation_factory=h.service.session_factory,
            business_factory=h.business,
        )
        budget = m.BudgetedRefundClient(None, tmp_path / "unused", "F1", 1, "b" * 64)
        service, facts, denied = m.make_semantic_service(
            resources, m.load_cases()["cases"][case_index], budget, tmp_path, cleanup
        )
        assert service.orchestration_version == "routed-v4"
        assert service.refund_runtime.profile.config_version == 7
        assert service.refund_runtime.profile.evidence_max_tokens == 15000
        assert service.refund_runtime.profile.decision_model.input_limit == 60000
        assert service.understanding.context.config.input_limit == 60000
        assert service.refund_runtime.client._clients == {}
        assert service.refund_runtime.flow.client is budget
        assert facts.descriptor.namespace == "refund-gate-" + "a" * 16
        access = service.refund_runtime.flow.identity.resolve(
            service.refund_identity_context
        )
        assert access.trusted and access.namespace == facts.descriptor.namespace
        assert denied.attempts == [] and budget.audits == []
        conversation = service.create()
        run, created = service.accept(
            conversation.conversation_id,
            "offline-accept",
            "订单 RF1501A 申请整单退款，原因是不想要了。",
            "agent",
        )
        assert created and run.model_name == settings.openai_model
        assert not budget.audits, "accepting a run cannot call a provider"
        memory = service.summary_service.for_run(run, SimpleNamespace())
        assert callable(memory.create_candidate)
        assert not denied.attempts, (
            "constructing summary metadata cannot invoke a model"
        )


def runner():
    assert importlib.util.find_spec("scripts.smoke_refund_flow"), (
        "missing bounded R2 runner"
    )
    return importlib.import_module("scripts.smoke_refund_flow")


def test_C2_model_and_retrieval_reservations_require_explicit_propagation(monkeypatch):
    from types import SimpleNamespace

    m = runner()
    calls = []
    monkeypatch.setattr(
        m,
        "reserve_refund_call",
        lambda *a, **kw: (calls.append(kw), SimpleNamespace(invocation_id="c2"))[1],
    )
    monkeypatch.setattr(m, "finish_refund_call", lambda *a, **kw: None)
    client = m.BudgetedRefundClient(None, "owned", "F1", 3, "f" * 64, continuation="C2")
    assert (
        client.invoke(purpose="resolution", invoke=lambda audit_hook: "output")
        == "output"
    )
    assert calls[0]["continuation"] == "C2"
    from scripts import refund_gate_support

    retrievals = []
    monkeypatch.setattr(
        refund_gate_support,
        "reserve_refund_retrieval",
        lambda *a, **kw: retrievals.append(kw),
    )
    monkeypatch.setattr(
        refund_gate_support, "finish_refund_retrieval", lambda *a, **kw: None
    )
    assert (
        m.budgeted_retrieve(
            "owned", "F1:3", "f" * 64, lambda: "evidence", continuation="C2"
        )
        == "evidence"
    )
    assert retrievals[0]["continuation"] == "C2"


@pytest.mark.parametrize("continuation", ["C2", "C3"])
def test_semantic_runner_defers_only_R1_and_stops_after_first_failed_case(
    tmp_path, monkeypatch, continuation
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scripts import refund_continuation as old
    from scripts import refund_gate_support as support
    from scripts import refund_semantic_continuation as c2

    m = runner()
    target = tmp_path / "cases"
    target.mkdir()
    monkeypatch.setattr(support, "new_report_dir", lambda *a: target)
    monkeypatch.setattr(c2, "fingerprints", lambda: ("a" * 40, {"source": "b" * 64}))
    monkeypatch.setattr(old, "fingerprints", lambda: ("a" * 40, {"source": "b" * 64}))
    authorization = []
    monkeypatch.setattr(
        c2,
        "authorize_" + continuation.lower(),
        lambda *a, **kw: authorization.append(kw),
    )
    monkeypatch.setattr(
        c2, "grant", lambda *a, **kw: {"r1_prerequisite_deferred": True}
    )
    finished = []
    monkeypatch.setattr(c2, "finish_case", lambda *a, **kw: finished.append(kw))

    @contextmanager
    def resources(_):
        yield SimpleNamespace(identity={})

    monkeypatch.setattr(support, "owned_refund_resources", resources)
    monkeypatch.setattr(m, "owned_selection", lambda _: None)
    invoked = []

    def semantic(resources, case, ownership, source_hash, target, **kw):
        invoked.append((case["case_id"], kw))
        return {"case_id": case["case_id"], "machine_passed": case["case_id"] == "F1"}

    monkeypatch.setattr(m, "semantic_case", semantic)
    result = m.run_refund_gate(
        ownership=tmp_path / "owner",
        case_ids=tuple(f"F{i}" for i in range(1, 9)),
        report_dir=target,
        prepare_report=m.ROOT
        / "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-prepare-continuation.json",
        continuation=continuation,
        approval_commit="a" * 40,
    )
    assert authorization and [case for case, _ in invoked] == ["F1", "F2"]
    assert all(kw["continuation"] == continuation for _, kw in invoked)
    assert [item["passed"] for item in finished] == [True, False]
    assert [kw["attempt"] for _, kw in invoked] == (
        [4, 2] if continuation == "C3" else [3, 2]
    )
    assert all(item["continuation"] == continuation for item in finished)
    assert (
        result["gate"] == "R2-" + continuation
        and result["r1_prerequisite_deferred"] is True
    )
    assert result["formal_migration_eligible"] is False
    assert len(result["cases"]) == 8 and all(
        c["status"] == "not_executed" for c in result["cases"][2:]
    )


@pytest.mark.parametrize("continuation", ["C2", "C3"])
def test_semantic_review_keeps_deferred_prerequisite_even_when_semantics_pass(
    monkeypatch, continuation
):
    # Projection mechanics only; the real semantic validator is tested separately.
    m = runner()
    from resolv.services.hybrid_fingerprints import canonical_hash

    cases = [{"case_id": f"F{i}"} for i in range(1, 9)]
    original = {
        "gate": "R2-" + continuation,
        "cases": cases,
        "candidate_commit": "a" * 40,
        "source_fingerprints": {"source": "b" * 64},
        "formal_preserved": True,
        "r1_prerequisite_deferred": True,
        "formal_migration_eligible": False,
    }
    reviews = {
        "reviewer": "offline-validator-mechanics",
        "observed_report_hash": canonical_hash(original),
        "case_reviews": [
            {
                "case_id": case["case_id"],
                "result_hash": canonical_hash(case),
                "notes": "mechanics-only",
                "checks": dict.fromkeys(
                    (
                        "scope",
                        "facts",
                        "policy",
                        "exceptions",
                        "time",
                        "reason",
                        "citations",
                        "status",
                        "submission",
                        "no_funds_promise",
                        "unverified",
                    ),
                    True,
                ),
            }
            for case in cases
        ],
    }
    monkeypatch.setattr(m, "check_semantic_result", lambda *a, **kw: {"passed": True})
    result = m.review_semantics(original, reviews)
    assert result["passed"] and result["formal_migration_eligible"] is False
    assert (
        result["r1_prerequisite_deferred"] is True
        and result["gate"] == "R2-" + continuation + "-reviewed"
    )


@pytest.mark.parametrize("continuation", ["C2", "C3"])
def test_semantic_startup_failure_still_records_the_deferred_prerequisite(
    tmp_path, monkeypatch, continuation
):
    import json

    from scripts import refund_gate_support as support
    from scripts import refund_semantic_continuation as c2

    m = runner()
    fixed = m.load_cases()
    monkeypatch.setattr(m, "load_cases", lambda: fixed)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    target = tmp_path / "report"
    target.mkdir()
    monkeypatch.setattr(support, "new_report_dir", lambda *a: target)
    monkeypatch.setattr(support, "validate_manifest", lambda *a: {})
    monkeypatch.setattr(support, "budget_ledger", lambda *a: [])

    def failed_source():
        raise ValueError("offline-startup-control")

    monkeypatch.setattr(c2, "fingerprints", failed_source)
    assert (
        m.main(
            [
                "--mode",
                "semantic",
                "--ownership",
                str(tmp_path / "owner"),
                "--report-dir",
                str(target),
                "--continuation",
                continuation,
                "--approval-plan-commit",
                "a" * 40,
            ]
        )
        == 1
    )
    receipt = json.loads(
        (
            tmp_path
            / f"docs/superpowers/acceptance/2026-10-16-refund-flow/R2-{continuation}.json"
        ).read_text()
    )
    assert receipt["gate"] == "R2-" + continuation and receipt["passed"] is False
    assert receipt["r1_prerequisite_deferred"] is True
    assert receipt["formal_migration_eligible"] is False


def test_retest_selection_is_fixed_distinct_and_at_most_two():
    m = runner()
    assert hasattr(m, "select_case_attempts"), "missing bounded corrected-case selector"
    assert [c["case_id"] for c in m.select_case_attempts(("F1", "F3"), attempt=2)] == [
        "F1",
        "F3",
    ]
    for ids in [("F1", "F2", "F3"), ("F1", "F1"), ("new",)]:
        with pytest.raises(ValueError):
            m.select_case_attempts(ids, attempt=2)
    with pytest.raises(ValueError):
        m.select_case_attempts(("F1",), attempt=1)


def test_empty_intent_continuation_never_clears_marker_or_allows_partial_data(
    tmp_path, monkeypatch
):
    from scripts import refund_gate_support as support

    m = runner()
    assert hasattr(m, "validate_empty_intent_continuation"), (
        "missing proof-bound empty-intent continuation"
    )
    monkeypatch.setattr(support, "ROOT", tmp_path)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    p = w / "resource-ownership.json"
    obj = support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    obj["collection_build"] = "started"
    support.save_json(p, obj)
    public = tmp_path / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    prior = public / "R2-prepare-correction.json"
    support.save_json(
        prior,
        {
            "gate": "R2-prepare",
            "passed": False,
            "error_type": "AttributeError",
            "source_fingerprints": {"fixture": "c" * 64},
            "resource_identity": obj,
        },
    )
    counts = {
        t: 0
        for t in [
            "knowledge_documents",
            "knowledge_chunks",
            "knowledge_index_versions",
            "knowledge_chunk_indexes",
        ]
    }
    assert m.validate_empty_intent_continuation(
        p, prior, {"fixture": "d" * 64}, counts=counts, collection_exists=False
    )
    assert support.validate_manifest(p)["collection_build"] == "started"
    for changed, exists in [
        (dict(counts, knowledge_documents=1), False),
        (counts, True),
    ]:
        with pytest.raises(ValueError):
            m.validate_empty_intent_continuation(
                p,
                prior,
                {"fixture": "d" * 64},
                counts=changed,
                collection_exists=exists,
            )
    obj["prepare_continuation"] = {"attempted": True}
    support.save_json(p, obj)
    with pytest.raises(ValueError):
        m.validate_empty_intent_continuation(
            p, prior, {"fixture": "d" * 64}, counts=counts, collection_exists=False
        )


def test_only_zero_work_preflight_can_receive_one_source_bound_correction(
    tmp_path, monkeypatch
):
    from scripts import refund_gate_support as support

    m = runner()
    assert hasattr(m, "validate_preflight_correction"), (
        "missing finite preflight correction guard"
    )
    monkeypatch.setattr(support, "ROOT", tmp_path)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    ownership = w / "resource-ownership.json"
    obj = support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    support.save_json(ownership, obj)
    prior = {
        "gate": "R2-prepare",
        "passed": False,
        "error_type": "TypeError",
        "source_fingerprints": {"fixture": "c" * 64},
        "resource_identity": obj,
    }
    public = tmp_path / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    p = public / "R2-prepare.json"
    support.save_json(p, prior)
    assert m.validate_preflight_correction(ownership, p, {"fixture": "d" * 64})
    with pytest.raises(ValueError):
        m.validate_preflight_correction(ownership, p, prior["source_fingerprints"])
    obj["collection_build"] = "started"
    support.save_json(ownership, obj)
    with pytest.raises(ValueError):
        m.validate_preflight_correction(ownership, p, {"fixture": "d" * 64})
    del obj["collection_build"]
    support.save_json(ownership, obj)
    support.save_json(public / "R2-prepare-correction.json", {"passed": False})
    with pytest.raises(ValueError):
        m.validate_preflight_correction(ownership, p, {"fixture": "d" * 64})


def test_gate_cli_requires_owned_new_reports_and_mode_before_io(tmp_path):
    m = runner()
    assert hasattr(m, "main"), "missing explicit R2 CLI"
    with pytest.raises(SystemExit):
        m.main([])
    with pytest.raises(ValueError):
        m.main(
            [
                "--mode",
                "prepare",
                "--ownership",
                str(tmp_path / "foreign"),
                "--report-dir",
                str(tmp_path / "out"),
            ]
        )


def test_case_adapter_never_reads_or_authorizes_another_order():
    m = runner()
    assert hasattr(m, "IsolatedRefundFacts"), (
        "missing explicitly isolated facts adapter"
    )
    from tests.refund_fixtures import binding

    case = m.load_cases()["cases"][0]
    adapter = m.IsolatedRefundFacts(case, "a" * 32)
    wrong = binding(namespace=adapter.descriptor.namespace)
    with pytest.raises(ValueError):
        adapter.query_order(None, wrong, "x")


def test_retrieval_budget_is_reserved_before_actual_search(tmp_path, monkeypatch):
    from scripts import refund_gate_support as support

    monkeypatch.setattr(support, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    p = w / "resource-ownership.json"
    support.save_json(
        p, support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    )
    assert hasattr(support, "reserve_refund_retrieval"), (
        "missing retrieval budget fence"
    )
    support.reserve_refund_retrieval(p, label="formal1", source_hash="c" * 64)
    with pytest.raises(ValueError):
        support.reserve_refund_retrieval(p, label="formal1", source_hash="c" * 64)
    with pytest.raises(ValueError):
        support.reserve_refund_retrieval(p, label="extra", source_hash="c" * 64)


def test_fixed_cases_are_complete_and_policy_quotes_are_committed():
    m = runner()
    fixed = m.load_cases()
    assert [c["case_id"] for c in fixed["cases"]] == [f"F{i}" for i in range(1, 9)]
    assert fixed["clock"] == "2026-10-16T04:00:00Z"
    bad = copy.deepcopy(fixed)
    del bad["cases"][0]["expected"]["required_fact_fields"]
    with pytest.raises(ValueError):
        m.validate_cases(bad)
    bad = copy.deepcopy(fixed)
    bad["policy_quotes"]["ordinary"] = "编造政策"
    with pytest.raises(ValueError):
        m.validate_cases(bad)


def test_manual_review_cannot_approve_a_different_observed_report():
    m = runner()
    with pytest.raises(ValueError):
        m.review_semantics(
            {"cases": []},
            {"observed_report_hash": "a" * 64, "reviewer": "test", "case_reviews": []},
        )


def test_prepare_failure_is_safe_retained_and_never_repeated(tmp_path, monkeypatch):
    from scripts import refund_gate_support as support

    m = runner()
    monkeypatch.setattr(support, "ROOT", tmp_path)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    ownership = w / "resource-ownership.json"
    support.save_json(
        ownership, support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    )
    monkeypatch.setattr(
        support, "committed_sources", lambda _: ("c" * 40, {"fixture": "d" * 64})
    )
    monkeypatch.setattr(support, "snapshot_formal_resources", lambda **kwargs: {})
    monkeypatch.setattr(m, "load_cases", lambda: {"cases": []})
    monkeypatch.setattr(m, "ensure_services", lambda *a: None)
    calls = []

    def fail(*args):
        calls.append("prepare")
        raise ValueError("private credential")

    monkeypatch.setattr(m, "prepare_owned", fail)
    args = [
        "--mode",
        "prepare",
        "--ownership",
        str(ownership),
        "--report-dir",
        str(w / "prepare"),
    ]
    assert m.main(args) == 1
    report = json.loads((w / "prepare/report.json").read_text())
    assert report["passed"] is False and report["formal_preserved"] is True
    assert "credential" not in str(report)
    with pytest.raises(ValueError):
        m.main(args)
    assert calls == ["prepare"]


def test_semantic_gate_rejects_incomplete_or_wrong_source_answer():
    m = runner()
    case = m.load_cases()["cases"][0]
    meta = json.loads(Path("frontend/tests/fixtures/refundReply.json").read_text())
    # The actual, internally valid software fixture is not ordinary-goods proof.
    result = {
        "status": "completed",
        "metadata": meta,
        "answer": "可以退款[1]",
        "observations": [],
        "decision": None,
        "binding": {"order_id": case["order_id"]},
        "application_count_before": 0,
        "application_count_after": 0,
        "legacy_tool_calls": 0,
    }
    checked = m.check_semantic_result(result, case)
    assert checked["passed"] is False
    assert checked["machine_passed"] is False
    assert checked["failures"]
    result["metadata"]["refund"]["verdict"] = "undetermined"
    assert m.check_semantic_result(result, case)["passed"] is False


def test_complete_isolated_proof_passes_structure_but_mutated_observation_fails(
    submission_harness,
):
    from resolv.repositories.refund_flow import RefundFlowRepository
    from tests.test_refund_submission import counts, submit

    h, m = submission_harness, runner()
    with h.factory() as session:
        record = RefundFlowRepository(session, "local").get(h.run.run_id)
    meta = h.service.get_run(h.run.run_id).result_json
    before = counts(h)[0]
    view = submit(h)
    again = submit(h)
    result = {
        "status": "completed",
        "metadata": meta,
        "answer": record.decision.reason,
        "binding": record.binding.model_dump(mode="json"),
        "observations": [o.model_dump(mode="json") for o in record.observations],
        "decision": record.decision.model_dump(mode="json"),
        "refund_record": record.model_dump(mode="json"),
        "application_count_before": before,
        "application_count_after": counts(h)[0],
        "submission": view.model_dump(mode="json"),
        "duplicate_receipt": again.receipt.model_dump(mode="json"),
        "legacy_tool_calls": 0,
    }
    # This is a validator mechanics fixture, never an F1 accuracy acceptance.
    case = {
        "case_id": "fixture",
        "order_id": h.offer.binding.order_id,
        "expected": {
            "verdict": "eligible",
            "next_actions": ["show_offer"],
            "offer_count": 1,
            "policy_quotes": [record.decision.policy_citations[0].quote],
            "required_fact_fields": ["activated"],
            "reason_groups": [],
            "forbidden_patterns": [],
        },
    }
    checked = m.check_semantic_result(result, case)
    assert checked["machine_passed"] is True and checked["passed"] is False
    changed = copy.deepcopy(result)
    changed["observations"][0]["fields"]["authorized"]["value"] = False
    assert m.check_semantic_result(changed, case)["machine_passed"] is False


def test_true_forward_and_source_receipts_required():
    m = runner()
    with pytest.raises(ValueError):
        m.validate_retrieval_gate({"passed": True, "real_services": True})
    fake = {
        "model_class": "FakeForward",
        "device": "mps:0",
        "dtype": "torch.float32",
        "batch_size": 4,
        "input_hash": "a" * 64,
        "output_hash": "b" * 64,
        "shape": [4, 1],
        "finite": True,
    }
    with pytest.raises(ValueError):
        m.validate_forward_receipt(fake)


def test_gate_failures_and_unknown_calls_consume_budget(tmp_path, monkeypatch):
    from scripts import refund_gate_support as support

    monkeypatch.setattr(support, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    manifest = w / "resource-ownership.json"
    support.save_json(
        manifest, support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    )
    m = runner()
    calls = []

    def fail(**kwargs):
        calls.append("sent")
        raise TimeoutError("private key must not escape")

    wrapped = m.BudgetedRefundClient(fail, manifest, "F1", 1, "c" * 64)
    with pytest.raises(TimeoutError):
        wrapped.invoke(purpose="refund_query", invoke=fail)
    ledger = support.budget_ledger(manifest)
    assert calls == ["sent"] and len(ledger) == 1
    assert ledger[0]["outcome"] == "unknown" and ledger[0]["input_tokens"] is None
    with pytest.raises(ValueError):
        wrapped.invoke(purpose="refund_query", invoke=fail)
    assert calls == ["sent"]


def test_case_attempt_cannot_be_replayed_before_a_provider_call(tmp_path, monkeypatch):
    from scripts import refund_gate_support as support

    monkeypatch.setattr(support, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    p = w / "resource-ownership.json"
    support.save_json(
        p, support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    )
    assert hasattr(support, "reserve_refund_case_attempt"), (
        "missing pre-provider case fence"
    )
    support.reserve_refund_case_attempt(
        p, case_id="F1", attempt=1, source_hash="c" * 64
    )
    with pytest.raises(ValueError):
        support.reserve_refund_case_attempt(
            p, case_id="F1", attempt=1, source_hash="c" * 64
        )
