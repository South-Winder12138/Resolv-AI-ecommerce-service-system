"""Recoverable knowledge stages. External work never runs in a SQL transaction."""

from typing import TYPE_CHECKING

from resolv.models.conversation import ConversationMessage
from resolv.models.knowledge_branch import ConversationKnowledgeReviewItem
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.knowledge_branch import KnowledgeBranchRepository
from resolv.schemas.knowledge_branch import (
    BranchReplyV2,
    FrozenActionDraft,
    FrozenSuggestedAction,
    KnowledgeBranchError,
    KnowledgeBranchProfile,
    KnowledgeEvidenceBundle,
    KnowledgePublicResult,
    KnowledgeRetrievalStage,
    KnowledgeReviewItem,
    ReplyMetadataV2,
    canonical_json,
    evidence_payload,
)
from resolv.services.branch_replies import draft_content
from resolv.services.context import knowledge_text
from resolv.services.conversation_routing import load_primary
from resolv.services.hybrid_fingerprints import canonical_hash, manifest_hash
from resolv.services.knowledge_branch_answer import (
    build_knowledge_answer_input,
    generate_knowledge_answer,
    knowledge_answer_question,
    knowledge_context_builder,
)
from resolv.services.knowledge_branch_check import check_knowledge_evidence
from resolv.services.knowledge_branch_evidence import select_knowledge_evidence
from resolv.services.knowledge_branch_query import prepare_knowledge_query
from resolv.services.knowledge_confidence import score_knowledge_evidence

if TYPE_CHECKING:
    from resolv.services.conversation import ConversationService

FALLBACK_TEXT = "目前的知识资料还不足以可靠回答这个问题。您可以补充具体情况，或通过下方入口整理转人工说明。"


def knowledge_answer_reply(record, answer):
    if (
        answer is None
        or record.check is None
        or not record.check.evidence_strong
        or record.retrieval is None
        or answer.evidence_hash != record.retrieval.bundle.evidence_hash
    ):
        raise KnowledgeBranchError("knowledge_answer_input_conflict")
    b, s = record.retrieval.bundle, record.profile.selection
    return BranchReplyV2(
        answer=answer.answer,
        metadata=ReplyMetadataV2(
            reply_kind="answer",
            understanding_id=record.understanding_id,
            primary_task_id=record.primary_task_id,
            branch="knowledge",
            knowledge=KnowledgePublicResult(
                release_id=s.release_id,
                index_version=s.index_version,
                evidence_hash=b.evidence_hash,
                decision="sufficient",
                citations=b.citations,
                chunks=b.chunks,
                used_citation_numbers=answer.used_citation_numbers,
            ),
        ),
    )


class KnowledgeBranchService:
    def __init__(self, conversation: "ConversationService", *, clients, retrievers):
        self.conversation, self.clients, self.retrievers = (
            conversation,
            clients,
            retrievers,
        )

    def _load(self, run_id, execution):
        execution.assert_owned()
        run = self.conversation.get_run(run_id)
        u = self.conversation.understanding.record(
            run.conversation_id, run.understanding_id
        )
        self.conversation.understanding.compatible(u)
        primary = load_primary(
            run,
            self.conversation.understanding.view(u),
            snapshot_hash=u.snapshot.snapshot_hash,
        )
        try:
            profile = KnowledgeBranchProfile.model_validate_json(
                canonical_json(run.knowledge_config)
            )
        except ValueError as exc:
            raise KnowledgeBranchError("knowledge_profile_hash_conflict") from exc
        with self.conversation.session_factory.begin() as session:
            record = KnowledgeBranchRepository(
                session, self.conversation.owner_id
            ).create(run_id, execution.epoch, primary, profile)
        if record.check and (
            record.retrieval is None
            or record.check.evidence_hash != record.retrieval.bundle.evidence_hash
        ):
            raise KnowledgeBranchError("knowledge_stage_hash_conflict")
        if (
            record.answer_input
            and record.answer_input.evidence_hash
            != record.retrieval.bundle.evidence_hash
        ):
            raise KnowledgeBranchError("knowledge_stage_hash_conflict")
        if record.answer and (
            record.answer.evidence_hash != record.retrieval.bundle.evidence_hash
            or record.answer.input_hash != record.answer_input.input_hash
        ):
            raise KnowledgeBranchError("knowledge_stage_hash_conflict")
        return run, primary, record

    def _counter(self, profile):
        return self.clients["knowledge_answer"].counter_factory(
            profile.answer_model.tokenizer_profile
        )

    def _audit(self, run_id, execution):
        def save(audit):
            execution.assert_owned()
            with self.conversation.session_factory.begin() as session:
                KnowledgeBranchRepository(
                    session, self.conversation.owner_id
                ).record_audit(run_id, execution.epoch, audit)

        return save

    def _phase(self, run_id, execution, stage, operation):
        run, primary, record = self._load(run_id, execution)
        if record.saved(stage) is not None:
            return record
        try:
            payload = operation(run, primary, record)
            execution.assert_owned()
            self.conversation.fault(f"knowledge_before_{stage}_saved")
            expected = (
                record.stages[-1].output_hash if record.stages else record.primary_hash
            )
            with self.conversation.session_factory.begin() as session:
                saved = KnowledgeBranchRepository(
                    session, self.conversation.owner_id
                ).save_stage(
                    run_id,
                    execution.epoch,
                    stage,
                    payload,
                    expected_input_hash=expected,
                )
            self.conversation.fault(f"knowledge_after_{stage}_saved")
            return saved
        except Exception as exc:
            execution.assert_owned()
            try:
                with self.conversation.session_factory.begin() as session:
                    KnowledgeBranchRepository(
                        session, self.conversation.owner_id
                    ).record_error(
                        run_id,
                        execution.epoch,
                        getattr(exc, "code", "knowledge_stage_failed"),
                        getattr(exc, "retryable", True),
                    )
            except Exception as store_error:
                raise KnowledgeBranchError(
                    "knowledge_store_failed", True
                ) from store_error
            raise

    def prepare_query(self, run_id, execution):
        def prepare(run, primary, record):
            if record.profile.selection.status != "configured":
                raise KnowledgeBranchError(
                    "knowledge_release_disabled"
                    if record.profile.selection.status == "disabled"
                    else "knowledge_release_selector_invalid"
                )
            return prepare_knowledge_query(
                primary,
                client=self.clients["knowledge_query"],
                profile=record.profile,
                query_token_count=self.retrievers.count_query_tokens,
                audit_hook=self._audit(run_id, execution),
            )

        return self._phase(run_id, execution, "query", prepare)

    def retrieve(self, run_id, execution):
        def retrieve(run, primary, record):
            if record.query is None:
                raise KnowledgeBranchError("knowledge_stage_predecessor_missing")
            profile = record.profile
            version = self.retrievers.validate_selection(profile.selection)
            if (
                manifest_hash(version.manifest) != profile.selection.manifest_hash
                or version.profile != profile.index_profile
            ):
                raise KnowledgeBranchError("knowledge_release_not_ready")
            result = self.retrievers.retrieve(
                profile.selection,
                record.query.standard,
                bm25_text=record.query.bm25_text,
                timeout_seconds=profile.queue_timeout_seconds,
            )
            if (
                result.index_version != profile.selection.index_version
                or result.index_profile_hash != profile.selection.profile_hash
                or result.retrieval_profile != profile.retrieval_profile
            ):
                raise KnowledgeBranchError("knowledge_release_not_ready")
            bundle = select_knowledge_evidence(
                result,
                counter=self._counter(profile),
                max_tokens=profile.evidence_max_tokens,
            )
            return KnowledgeRetrievalStage.from_result(
                result,
                bundle=bundle,
                file_configs=tuple(
                    (d.document_id, d.file_config) for d in version.manifest.documents
                ),
            )

        return self._phase(run_id, execution, "retrieval", retrieve)

    def check(self, run_id, execution):
        def check(run, primary, record):
            if record.retrieval is None:
                raise KnowledgeBranchError("knowledge_stage_predecessor_missing")
            signals = score_knowledge_evidence(
                record.retrieval.bundle,
                file_configs=dict(record.retrieval.file_configs),
                profile=record.profile,
            )
            return check_knowledge_evidence(
                primary,
                record.query,
                record.retrieval.bundle,
                signals,
                client=self.clients["knowledge_check"],
                profile=record.profile,
                audit_hook=self._audit(run_id, execution),
            )

        return self._phase(run_id, execution, "check", check)

    def answer(self, run_id, execution):
        def context(run, primary, record):
            if record.check is None or not record.check.evidence_strong:
                raise KnowledgeBranchError("knowledge_evidence_not_sufficient")
            b, p = record.retrieval.bundle, record.profile
            counter = self._counter(p)
            # Fixed evidence/question cannot be made smaller by summarizing history.
            # Use the exact answer builder before any memory/summary side effects.
            build_knowledge_answer_input(
                run, primary, record.query, b,
                summary=None, history=(), profile=p, counter=counter,
            )
            builder, _ = knowledge_context_builder(p, counter)
            from resolv.services.knowledge_branch_summary import (
                KnowledgeSummaryAdapter,
            )
            summary_service = KnowledgeSummaryAdapter(self.conversation.summary_service, p, self._audit(run_id, execution)) if self.conversation.summary_service else None
            pinned = self.conversation.prepare_memory(
                run_id,
                execution,
                working_question=knowledge_answer_question(primary, record.query),
                context_builder=builder,
                system_prompt=p.answer_model.system_prompt,
                knowledge=[b.payload],
                tool_schemas=[],
                summary_service_override=summary_service,
            )
            with self.conversation.session_factory() as session:
                repo = ConversationRepository(session, self.conversation.owner_id)
                summary = repo.summary(run.conversation_id, pinned.summary_version)
                history = repo.completed_turns(
                    run.conversation_id,
                    after_seq=summary.covered_until_seq if summary else 0,
                    cutoff_seq=pinned.history_cutoff_seq,
                )
            return build_knowledge_answer_input(
                pinned,
                primary,
                record.query,
                b,
                summary=summary,
                history=history,
                profile=p,
                counter=counter,
            )

        self._phase(run_id, execution, "answer_input", context)

        def generate(run, primary, record):
            if record.check is None or not record.check.evidence_strong:
                raise KnowledgeBranchError("knowledge_evidence_not_sufficient")
            return generate_knowledge_answer(
                record.answer_input,
                record.retrieval.bundle,
                client=self.clients["knowledge_answer"],
                profile=record.profile,
                audit_hook=self._audit(run_id, execution),
            )

        return self._phase(run_id, execution, "answer", generate).answer

    def fallback(self, run_id, execution):
        _, primary, record = self._load(run_id, execution)
        if record.check is None or record.check.evidence_strong:
            raise KnowledgeBranchError("knowledge_fallback_not_authorized")
        s = record.profile.selection
        return BranchReplyV2(
            answer=FALLBACK_TEXT,
            metadata=ReplyMetadataV2(
                reply_kind="knowledge_fallback",
                understanding_id=record.understanding_id,
                primary_task_id=record.primary_task_id,
                branch="knowledge",
                suggested_actions=(
                    FrozenSuggestedAction(
                        suggestion_id=canonical_hash(run_id + "transfer_human"),
                        type="transfer_human",
                        label="转人工",
                        draft=FrozenActionDraft(
                            title="转人工内容确认",
                            content=draft_content(primary.primary),
                        ),
                    ),
                ),
                knowledge=KnowledgePublicResult(
                    release_id=s.release_id,
                    index_version=s.index_version,
                    evidence_hash=record.retrieval.bundle.evidence_hash,
                    decision="insufficient",
                    citations=(),
                    chunks=(),
                    used_citation_numbers=(),
                ),
            ),
        )

    def _review(self, run, primary, record):
        b = record.retrieval.bundle
        windows, citations = b.windows[:3], b.citations[:3]
        ids = {cid for w in windows for cid in w.chunk_ids}
        chunks = tuple(c for c in b.chunks if c.chunk.chunk_id in ids)
        serialized = knowledge_text([evidence_payload(citations, chunks)])
        evidence = KnowledgeEvidenceBundle(
            windows=windows,
            citations=citations,
            chunks=chunks,
            serialized=serialized,
            input_tokens=self._counter(record.profile).count_text(serialized),
            evidence_hash=canonical_hash(
                {"rule": "knowledge-evidence-v1", "text": serialized}
            ),
        )
        return KnowledgeReviewItem(
            run_id=run.run_id,
            conversation_id=run.conversation_id,
            understanding_id=record.understanding_id,
            primary_task_id=record.primary_task_id,
            original_question=run.question,
            primary_text=primary.primary.text,
            standard=record.query.standard,
            source=record.check.source,
            total=record.check.signals.total,
            self_check_reason=record.check.self_check.reason
            if record.check.self_check
            else None,
            evidence=evidence,
            created_at=run.created_at,
        )

    def finalize(self, run_id, execution, reply):
        run, primary, record = self._load(run_id, execution)
        if record.check is None:
            raise KnowledgeBranchError("knowledge_stage_predecessor_missing")
        expected = (
            knowledge_answer_reply(record, record.answer)
            if record.check.evidence_strong
            else self.fallback(run_id, execution)
        )
        if BranchReplyV2.model_validate_json(reply.model_dump_json()) != expected:
            raise KnowledgeBranchError("knowledge_reply_conflict")
        persisted_metadata = reply.metadata
        if run.orchestration_version == "routed-v4":
            from resolv.services.refund_runtime import promote_reply_v3
            persisted_metadata = promote_reply_v3(persisted_metadata)
        review = (
            None if record.check.evidence_strong else self._review(run, primary, record)
        )
        try:
            execution.assert_owned()
            with self.conversation.session_factory.begin() as session:
                ConversationRepository(
                    session, self.conversation.owner_id
                ).commit_answer(
                    run_id, execution.epoch, reply.answer, metadata=reply.metadata
                )
                if review:
                    KnowledgeBranchRepository(
                        session, self.conversation.owner_id
                    ).add_review(run_id, execution.epoch, review)
                self.conversation.fault("knowledge_before_finalize_commit")
            self.conversation.fault("knowledge_after_finalize_commit")
        except Exception as exc:
            execution.assert_owned()
            try:
                with self.conversation.session_factory() as session:
                    repo = ConversationRepository(session, self.conversation.owner_id)
                    saved_run = repo.run(run_id)
                    answer = session.get(ConversationMessage, run.answer_message_id)
                    item = session.get(ConversationKnowledgeReviewItem, run_id)
                    confirmed = (
                        answer is not None
                        and answer.text == reply.answer
                        and saved_run.result_json
                        == persisted_metadata.model_dump(mode="json")
                        and (
                            (review is None and item is None)
                            or (
                                review is not None
                                and item is not None
                                and item.item_json == review.model_dump(mode="json")
                                and item.item_hash == review.fingerprint
                            )
                        )
                    )
                if confirmed:
                    return
            except Exception as store_error:
                raise KnowledgeBranchError(
                    "knowledge_finalize_store_failed", True
                ) from store_error
            try:
                with self.conversation.session_factory.begin() as session:
                    KnowledgeBranchRepository(
                        session, self.conversation.owner_id
                    ).record_error(
                        run_id, execution.epoch, "knowledge_finalize_store_failed", True
                    )
            except Exception as store_error:
                raise KnowledgeBranchError(
                    "knowledge_finalize_store_failed", True
                ) from store_error
            raise KnowledgeBranchError("knowledge_finalize_store_failed", True) from exc
