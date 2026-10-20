"""Persisted v2 reply whitelist; internal execution records never become API data."""

from resolv.schemas.branch_execution import parse_reply_metadata
from resolv.schemas.conversation import ConversationError
from resolv.schemas.knowledge_branch import (
    KnowledgeExecutionRecord,
    KnowledgePublicResult,
    ReplyMetadataV2,
    canonical_json,
)
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.knowledge_branch import knowledge_answer_reply


def public_knowledge_reply(run, execution_record) -> dict:
    def conflict():
        raise ConversationError("run_state_conflict", run_id=run.run_id)

    try:
        if (
            run.orchestration_version not in {"routed-v3", "routed-v4"}
            or run.status != "completed"
            or run.result_json is None
        ):
            conflict()
        meta = parse_reply_metadata(run.result_json)
        if (
            not isinstance(meta, ReplyMetadataV2)
            or meta.understanding_id != run.understanding_id
        ):
            conflict()
        public = meta.model_dump(mode="json")
        if run.orchestration_version == "routed-v4":
            if meta.schema_version != 3 or meta.refund is not None:
                conflict()
            meta = ReplyMetadataV2.model_validate_json(
                canonical_json(
                    {k: v for k, v in public.items() if k != "refund"}
                    | {"schema_version": 2}
                )
            )
        elif meta.schema_version != 2:
            conflict()
        if meta.branch != "knowledge":
            if meta.knowledge is not None or meta.reply_kind == "knowledge_fallback":
                conflict()
            return public
        if execution_record is None:
            conflict()
        record = KnowledgeExecutionRecord.model_validate_json(
            execution_record.model_dump_json()
        )
        if (
            record.run_id != run.run_id
            or record.conversation_id != run.conversation_id
            or record.understanding_id != run.understanding_id
            or meta.primary_task_id != record.primary_task_id
            or canonical_hash(run.knowledge_config) != record.profile.fingerprint
            or record.retrieval is None
            or record.check is None
            or record.check.evidence_hash != record.retrieval.bundle.evidence_hash
        ):
            conflict()
        previous = record.primary_hash
        for stage in record.stages:
            if stage.input_hash != previous:
                conflict()
            previous = stage.output_hash
        selection = record.profile.selection
        if selection.status != "configured" or meta.knowledge is None:
            conflict()
        if record.check.evidence_strong:
            if (
                record.answer is None
                or record.answer_input is None
                or record.answer.input_hash != record.answer_input.input_hash
                or record.answer.evidence_hash != record.retrieval.bundle.evidence_hash
                or any(
                    n > len(record.retrieval.bundle.citations)
                    for n in record.answer.used_citation_numbers
                )
            ):
                conflict()
            expected = knowledge_answer_reply(record, record.answer).metadata
            if meta != expected:
                conflict()
        else:
            expected = KnowledgePublicResult(
                release_id=selection.release_id,
                index_version=selection.index_version,
                evidence_hash=record.retrieval.bundle.evidence_hash,
                decision="insufficient",
                citations=(),
                chunks=(),
                used_citation_numbers=(),
            )
            if meta.reply_kind != "knowledge_fallback" or meta.knowledge != expected:
                conflict()
        return public
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise ConversationError("run_state_conflict", run_id=run.run_id) from exc
