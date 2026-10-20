"""Offline factories for the knowledge contracts; no model or server imports."""

import math
from types import SimpleNamespace

import pytest

from resolv.core.knowledge_branch_config import default_knowledge_branch_profile
from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.hybrid_knowledge import (
    FusedHybridCandidate,
    HybridDiagnostics,
    HybridEvidenceWindow,
    HybridRetrievalProfile,
    HybridRetrievalResult,
    VerifiedHybridChunk,
)
from resolv.schemas.knowledge_branch import (
    KnowledgeCitation,
    KnowledgeEvidenceBundle,
    KnowledgeModelProfile,
    evidence_payload,
)
from resolv.services.context import knowledge_text
from resolv.services.hybrid_fingerprints import canonical_hash, profile_hash
from tests.branch_fixtures import make_view
from tests.hybrid_fixtures import candidate, chunk, profile


def model_profile(purpose):
    from resolv.core.knowledge_branch_prompts import PROMPTS

    return KnowledgeModelProfile(
        purpose=purpose,
        model_name="offline-fixture",
        api_address_fingerprint="e" * 64,
        model_api_mode="chat_completions",
        tokenizer_profile="o200k_base",
        model_context_window_tokens=256000,
        app_max_input_tokens=12000,
        provider_input_overhead_tokens=6500,
        reserved_output_tokens=2000,
        safety_margin_tokens=1000,
        system_prompt=PROMPTS.get(purpose, "只根据证据回答。"),
        prompt_version="fixture-v1",
    )


def branch_profile():
    return default_knowledge_branch_profile(
        query_model=model_profile("knowledge_query"),
        check_model=model_profile("knowledge_check"),
        answer_model=model_profile("knowledge_answer"),
        summary_profile=model_profile("summary"),
    )


def primary(text="退货运费由谁承担？", **ids):
    from resolv.understanding_rules import INTENT_BRANCHES_V1

    intent = next(k for k, v in INTENT_BRANCHES_V1.items() if v == "knowledge")
    request = make_view(text, intent).primary_request
    request = request.model_copy(
        update={"intent": intent, "branch": "knowledge", **ids}
    )
    return PrimaryExecution(
        conversation_id=request.conversation_id,
        understanding_id=request.understanding_id,
        snapshot_hash="f" * 64,
        primary=request,
    )


def hybrid_result(logits=(1.0,), *, overlapping=False):
    windows, bodies = [], {}
    for i, logit in enumerate(logits):
        start = (i if i < 2 else 3 * i - 2) if overlapping else 3 * i
        indices = range(start, start + 3)
        ids = tuple(f"{j + 1:064x}" for j in indices)
        for j, cid in zip(indices, ids):
            bodies[cid] = VerifiedHybridChunk(
                chunk=chunk(chunk_id=cid, chunk_index=j),
                content_hash="c" * 64,
                section_instance_id="f" * 64,
                expandable=True,
                document_hash="b" * 64,
                index_hash="c" * 64,
            )
        center = candidate(chunk_id=ids[1], rank=i + 1)
        fused = FusedHybridCandidate(
            index_version=center.index_version,
            chunk_id=center.chunk_id,
            document_id=center.document_id,
            index_hash=center.index_hash,
            dense=center,
            rrf_score=1.0 / (61 + i),
        )
        windows.append(
            HybridEvidenceWindow(
                window_id=f"{i + 100:064x}",
                document_id="d" * 64,
                section_instance_id="f" * 64,
                chunk_ids=ids,
                center_chunk_ids=(ids[1],),
                center_candidates=(fused,),
                best_rrf=fused.rrf_score,
                rerank_score=float(logit),
            )
        )
    return HybridRetrievalResult(
        index_version="fixture-v3",
        index_profile_hash=profile_hash(profile()),
        retrieval_profile=HybridRetrievalProfile(),
        windows=tuple(windows),
        chunks=bodies,
        diagnostics=HybridDiagnostics(selected_count=len(windows)),
    )


def bundle(logits=(1.0,), **changes):
    result = hybrid_result(logits, overlapping=changes.pop("overlapping", False))
    citations = tuple(
        KnowledgeCitation(
            **w.model_dump(),
            citation_number=i + 1,
            source="synthetic.md",
            title="退货",
            section_path="退货",
            index_version=result.index_version,
            index_profile_hash=result.index_profile_hash,
            sigmoid_score=(
                1 / (1 + math.exp(-w.rerank_score))
                if w.rerank_score >= 0
                else math.exp(w.rerank_score) / (1 + math.exp(w.rerank_score))
            ),
        )
        for i, w in enumerate(result.windows)
    )
    chunks = tuple(result.chunks.values())
    serialized = knowledge_text([evidence_payload(citations, chunks)])
    data = {
        "windows": result.windows,
        "citations": citations,
        "chunks": chunks,
        "serialized": serialized,
        "input_tokens": len(serialized),
        "evidence_hash": canonical_hash(
            {"rule": "knowledge-evidence-v1", "text": serialized}
        ),
    }
    return KnowledgeEvidenceBundle(**(data | changes))


@pytest.fixture
def knowledge_factory(tmp_path):
    from resolv.db.engine import create_db_engine, create_session_factory
    from resolv.models.conversation import Conversation
    from resolv.models.knowledge_branch import ConversationKnowledgeRun  # noqa: F401
    from resolv.models.understanding import ConversationUnderstanding  # noqa: F401

    engine = create_db_engine(f"sqlite:///{tmp_path / 'knowledge-branch.db'}")
    Conversation.metadata.create_all(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def knowledge_owned(knowledge_factory):
    from resolv.models.conversation import ConversationRun
    from resolv.services.conversation_routing import load_primary, run_understanding
    from tests.branch_fixtures import make_routed_service
    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.understanding_fixtures import make_classification, make_resolution

    text = "退货运费由谁承担？"
    service = make_routed_service(
        knowledge_factory,
        MemoryCheckpoints(),
        stage_outputs=[
            make_resolution(resolved_text=text),
            make_classification([text], ["product_consultation"]),
        ],
    )
    c = service.create()
    run, _ = service.accept(c.conversation_id, "knowledge-request", text, "agent")
    with service.execution(run.run_id) as execution:
        run = service.get_run(run.run_id)
        view = run_understanding(run, service.understanding)
        u = service.understanding.record(c.conversation_id, run.understanding_id)
        p = load_primary(run, view, snapshot_hash=u.snapshot.snapshot_hash)
        profile = branch_profile()
        with knowledge_factory.begin() as session:
            row = session.get(ConversationRun, run.run_id)
            row.orchestration_version = "routed-v3"
            row.state_schema_version = 3
            row.knowledge_config = profile.model_dump(mode="json")
        yield SimpleNamespace(
            factory=knowledge_factory,
            service=service,
            run=run,
            execution=execution,
            primary=p,
            profile=profile,
        )


def make_knowledge_service(
    factory,
    checkpoints,
    *,
    stage_outputs=None,
    hybrid_result=None,
    query_reply=None,
    check_reply=None,
    answer_reply=None,
    selection=None,
):
    """Real SQLite stages with deterministic audited model/transport substitutes."""
    import json
    from collections import Counter
    from datetime import UTC, datetime
    from uuid import uuid4

    from langchain_core.messages import AIMessage

    from resolv.models.conversation import ConversationRun
    from resolv.schemas.hybrid_knowledge import HybridVersionSnapshot
    from resolv.schemas.knowledge_branch import (
        KnowledgeModelAudit,
    )
    from resolv.schemas.knowledge_release import ReleaseSelection
    from resolv.services.conversation_routing import run_understanding
    from resolv.services.hybrid_fingerprints import manifest_hash
    from resolv.services.knowledge_branch import KnowledgeBranchService
    from tests.branch_fixtures import make_routed_service
    from tests.hybrid_fixtures import manifest
    from tests.hybrid_fixtures import profile as index_profile
    from tests.understanding_fixtures import make_classification, make_resolution

    result_factory = globals()["hybrid_result"]
    result = hybrid_result if hybrid_result is not None else result_factory((2.0, 1.0))
    selected = selection or ReleaseSelection(
        status="configured",
        release_id="fixture-release",
        index_version="fixture-v3",
        collection=index_profile().collection_name,
        profile=index_profile(),
        profile_hash=profile_hash(index_profile()),
        manifest_hash=manifest_hash(manifest()),
    )
    prof = branch_profile().model_copy(
        update={"selection": selected, "index_profile": selected.profile}
    )
    text = "退货运费由谁承担？"
    conversation = make_routed_service(
        factory,
        checkpoints,
        stage_outputs=stage_outputs
        or [
            make_resolution(resolved_text=text),
            make_classification([text], ["product_consultation"]),
        ],
    )
    c = conversation.create()
    run, _ = conversation.accept(c.conversation_id, "knowledge-service", text, "agent")
    with conversation.execution(run.run_id):
        run_understanding(conversation.get_run(run.run_id), conversation.understanding)
        with factory.begin() as session:
            row = session.get(ConversationRun, run.run_id)
            row.orchestration_version, row.state_schema_version = "routed-v3", 3
            row.knowledge_config = prof.model_dump(mode="json")
    counts = Counter()
    responses = {
        "knowledge_query": query_reply
        if query_reply is not None
        else {"standard": text, "expanded": ["邮费"]},
        "knowledge_check": check_reply
        if check_reply is not None
        else {
            "useful": True,
            "supporting_window_ids": [result.windows[0].window_id]
            if result.windows
            else [],
            "missing_information": [],
            "reason": "确定性测试条款",
        },
        "knowledge_answer": answer_reply
        if answer_reply is not None
        else AIMessage(
            content="质量问题由商家承担[1]。",
            response_metadata={"finish_reason": "stop"},
        ),
    }

    class TestCounter:
        def count_text(self, text):
            return max(1, len(text) // 4)

        def count(self, messages, schemas):
            return sum(self.count_text(m.content) for m in messages) + 10

    class AuditedClient:
        counter_factory = staticmethod(lambda _: TestCounter())

        def invoke_json(self, purpose, messages, output_type, *, profile, audit_hook):
            output = self.call(purpose, messages, profile, audit_hook)
            return output_type.model_validate_json(json.dumps(output))

        def invoke_answer(self, messages, *, profile, audit_hook):
            return self.call("knowledge_answer", messages, profile, audit_hook)

        def call(self, purpose, messages, profile, hook):
            counts[purpose] += 1
            a = KnowledgeModelAudit(
                invocation_id=uuid4().hex,
                purpose=purpose,
                profile_fingerprint=profile.fingerprint,
                input_hash=canonical_hash(
                    [m.model_dump(mode="json") for m in messages]
                ),
                input_estimate=1000,
                started_at=datetime.now(UTC),
                outcome="started",
            )
            hook(a)
            reply = responses[purpose]
            if isinstance(reply, Exception):
                hook(
                    a.model_copy(
                        update={
                            "outcome": "unknown",
                            "error_code": "knowledge_model_unavailable",
                            "finished_at": datetime.now(UTC),
                        }
                    )
                )
                raise reply
            hook(
                a.model_copy(
                    update={
                        "outcome": "succeeded",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "finish_reason": "stop",
                        "finished_at": datetime.now(UTC),
                    }
                )
            )
            return reply

    class Retrievers:
        error = None

        def count_query_tokens(self, text):
            return len(text)

        def validate_selection(self, selection):
            return HybridVersionSnapshot(
                version_id=selection.index_version,
                profile=selection.profile,
                manifest=manifest(),
                state="ready",
            )

        def retrieve(self, selection, query, *, bm25_text, timeout_seconds):
            counts["retrieval"] += 1
            if self.error:
                raise self.error
            assert query == text and bm25_text.startswith(query)
            return result

    client, retrievers = AuditedClient(), Retrievers()
    service = KnowledgeBranchService(
        conversation, clients={p: client for p in responses}, retrievers=retrievers
    )
    h = SimpleNamespace(
        conversation=conversation,
        service=service,
        run=conversation.get_run(run.run_id),
        counts=counts,
        responses=responses,
        retrievers=retrievers,
        profile=prof,
        fault_hook=None,
    )
    conversation.fault_hook = lambda point: (
        h.fault_hook(point) if h.fault_hook else None
    )
    return h
