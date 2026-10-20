"""Fenced write-once stages inside caller-owned short conversation transactions."""

from datetime import timedelta

from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from resolv.models.conversation import Conversation
from resolv.models.knowledge_branch import (
    ConversationKnowledgeReviewItem,
    ConversationKnowledgeRun,
)
from resolv.models.understanding import (
    ConversationUnderstanding,
    ConversationUnderstandingRequest,
)
from resolv.repositories.conversation import ConversationRepository
from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.knowledge_branch import (
    _STAGE_TYPES,
    FrozenJSON,
    KnowledgeBranchError,
    KnowledgeBranchProfile,
    KnowledgeExecutionRecord,
    KnowledgeModelAudit,
    KnowledgeReviewItem,
    SavedKnowledgeStage,
    canonical_json,
)
from resolv.schemas.understanding import UnderstandingRequestRecord
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.understanding_rules import INTENT_BRANCHES_V1

STAGES = ("query", "retrieval", "check", "answer_input", "answer")


def same_task_identity(value, frozen):
    if value.model_dump(exclude={"created_at"}) != frozen.model_dump(
        exclude={"created_at"}
    ):
        return False
    if value.created_at == frozen.created_at:
        return True
    # MySQL DATETIME(0) can truncate or round the Python creation timestamp.
    # All executable/provenance fields still match exactly; frozen JSON is unchanged.
    if value.created_at.microsecond:
        return False
    floor = frozen.created_at.replace(microsecond=0)
    allowed = {floor}
    if frozen.created_at.microsecond >= 500000:
        allowed.add(floor + timedelta(seconds=1))
    return value.created_at in allowed


class KnowledgeBranchRepository:
    def __init__(self, session, owner_id):
        self.session, self.owner_id = session, owner_id

    def _row(self, run_id):
        return self.session.scalar(
            select(ConversationKnowledgeRun)
            .join(
                Conversation,
                Conversation.conversation_id
                == ConversationKnowledgeRun.conversation_id,
            )
            .where(
                ConversationKnowledgeRun.run_id == run_id,
                Conversation.owner_id == self.owner_id,
            )
            .execution_options(populate_existing=True)
        )

    def _primary(self, run, primary):
        p = PrimaryExecution.model_validate_json(primary.model_dump_json())
        task = self.session.get(
            ConversationUnderstandingRequest, p.primary.task_id, populate_existing=True
        )
        u = (
            self.session.get(
                ConversationUnderstanding, run.understanding_id, populate_existing=True
            )
            if run.understanding_id
            else None
        )
        if (
            run.orchestration_version not in {"routed-v3", "routed-v4"}
            or p.conversation_id != run.conversation_id
            or p.understanding_id != run.understanding_id
            or p.primary.branch != "knowledge"
            or INTENT_BRANCHES_V1.get(p.primary.intent) != "knowledge"
            or task is None
            or u is None
            or u.conversation_id != run.conversation_id
            or u.status != "ready"
            or u.snapshot.get("snapshot_hash") != p.snapshot_hash
        ):
            raise KnowledgeBranchError("knowledge_identity_conflict")
        value = UnderstandingRequestRecord.model_validate(
            {
                name: getattr(task, name)
                for name in UnderstandingRequestRecord.model_fields
            }
        )
        if not same_task_identity(value, p.primary):
            raise KnowledgeBranchError("knowledge_identity_conflict")
        return p

    def _owned(self, run_id, epoch):
        run = ConversationRepository(self.session, self.owner_id).owned(run_id, epoch)
        row = self._row(run_id)
        if row is None:
            raise KnowledgeBranchError("knowledge_run_missing")
        p = self._primary(
            run, PrimaryExecution.model_validate_json(canonical_json(row.primary_json))
        )
        if (
            row.owner_id != self.owner_id
            or row.primary_hash != canonical_hash(p.model_dump(mode="json"))
            or row.profile_json != run.knowledge_config
            or row.profile_hash != canonical_hash(row.profile_json)
            or row.conversation_id != run.conversation_id
            or row.understanding_id != run.understanding_id
            or row.primary_task_id != p.primary.task_id
        ):
            raise KnowledgeBranchError("knowledge_identity_conflict")
        return row

    def _record(self, row):
        try:
            if row.owner_id != self.owner_id or row.primary_hash != canonical_hash(
                row.primary_json
            ):
                raise KnowledgeBranchError("knowledge_identity_conflict")
            stages = []
            previous = row.primary_hash
            for stage in STAGES:
                value = getattr(row, f"{stage}_payload")
                if value is None and (
                    getattr(row, f"{stage}_input_hash") is not None
                    or getattr(row, f"{stage}_output_hash") is not None
                ):
                    raise KnowledgeBranchError("knowledge_stage_hash_conflict")
                if value is not None:
                    saved = SavedKnowledgeStage(
                        stage=stage,
                        payload=FrozenJSON.from_value(value),
                        input_hash=getattr(row, f"{stage}_input_hash"),
                        output_hash=getattr(row, f"{stage}_output_hash"),
                    )
                    if saved.input_hash != previous:
                        raise KnowledgeBranchError("knowledge_stage_hash_conflict")
                    previous = saved.output_hash
                    stages.append(saved)
            if row.profile_hash != canonical_hash(row.profile_json):
                raise KnowledgeBranchError("knowledge_profile_hash_conflict")
            return KnowledgeExecutionRecord(
                run_id=row.run_id,
                owner_id=row.owner_id,
                conversation_id=row.conversation_id,
                understanding_id=row.understanding_id,
                primary_task_id=row.primary_task_id,
                primary_hash=row.primary_hash,
                profile=KnowledgeBranchProfile.model_validate_json(
                    canonical_json(row.profile_json)
                ),
                stages=tuple(stages),
                audits=tuple(
                    KnowledgeModelAudit.model_validate_json(canonical_json(a))
                    for a in row.audits
                ),
                error_code=row.error_code,
                retryable=row.retryable,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise KnowledgeBranchError("knowledge_snapshot_invalid") from exc

    def get(self, run_id) -> KnowledgeExecutionRecord | None:
        row = self._row(run_id)
        return self._record(row) if row is not None else None

    def create(self, run_id, epoch, primary, profile):
        run = ConversationRepository(self.session, self.owner_id).owned(run_id, epoch)
        p = self._primary(run, primary)
        profile = KnowledgeBranchProfile.model_validate_json(profile.model_dump_json())
        value = profile.model_dump(mode="json")
        if run.knowledge_config != value:
            raise KnowledgeBranchError("knowledge_profile_hash_conflict")
        row = self._row(run_id)
        if row is None:
            try:
                with self.session.begin_nested():
                    row = ConversationKnowledgeRun(
                        run_id=run_id,
                        owner_id=self.owner_id,
                        conversation_id=run.conversation_id,
                        understanding_id=run.understanding_id,
                        primary_task_id=p.primary.task_id,
                        primary_hash=canonical_hash(p.model_dump(mode="json")),
                        primary_json=p.model_dump(mode="json"),
                        profile_json=value,
                        profile_hash=profile.fingerprint,
                    )
                    self.session.add(row)
                    self.session.flush()
            except IntegrityError:
                row = self._row(run_id)
                if row is None:
                    raise
        if row.primary_json != p.model_dump(mode="json") or row.profile_json != value:
            raise KnowledgeBranchError("knowledge_identity_conflict")
        return self._record(self._owned(run_id, epoch))

    def save_stage(self, run_id, epoch, stage, payload, *, expected_input_hash):
        if stage not in STAGES:
            raise KnowledgeBranchError("knowledge_stage_invalid")
        row = self._owned(run_id, epoch)
        record = self._record(row)
        position = STAGES.index(stage)
        if len(record.stages) < position:
            raise KnowledgeBranchError("knowledge_stage_predecessor_missing")
        actual_input = (
            record.primary_hash
            if position == 0
            else record.stages[position - 1].output_hash
        )
        if expected_input_hash != actual_input:
            raise KnowledgeBranchError("knowledge_stage_hash_conflict")
        checked = _STAGE_TYPES[stage].model_validate_json(payload.model_dump_json())
        value = checked.model_dump(mode="json")
        if stage == "query" and checked.input_hash != row.primary_hash:
            raise KnowledgeBranchError("knowledge_stage_hash_conflict")
        column = getattr(ConversationKnowledgeRun, f"{stage}_payload")
        self.session.execute(
            update(ConversationKnowledgeRun)
            .where(ConversationKnowledgeRun.run_id == run_id, column.is_(None))
            .values(
                **{
                    f"{stage}_payload": value,
                    f"{stage}_input_hash": actual_input,
                    f"{stage}_output_hash": canonical_hash(value),
                    "revision": ConversationKnowledgeRun.revision + 1,
                }
            )
            .execution_options(synchronize_session=False)
        )
        self.session.expire(row)
        saved = self._record(row)
        if (
            getattr(row, f"{stage}_payload") != value
            or getattr(row, f"{stage}_input_hash") != actual_input
        ):
            raise KnowledgeBranchError("knowledge_stage_conflict")
        return saved

    def record_audit(self, run_id, epoch, audit):
        row = self._owned(run_id, epoch)
        audit = KnowledgeModelAudit.model_validate_json(audit.model_dump_json())
        profile = self._record(row).profile
        field = {
            "knowledge_query": "query_model",
            "knowledge_check": "check_model",
            "knowledge_answer": "answer_model",
            "summary": "summary_profile",
        }[audit.purpose]
        if audit.profile_fingerprint != getattr(profile, field).fingerprint:
            raise KnowledgeBranchError("knowledge_audit_profile_conflict")
        value = audit.model_dump(mode="json")
        old = next(
            (
                a
                for a in row.audits
                if (a["invocation_id"], a["outcome"])
                == (audit.invocation_id, audit.outcome)
            ),
            None,
        )
        if old is not None:
            if old != value:
                raise KnowledgeBranchError("knowledge_audit_conflict")
            return
        result = self.session.execute(
            update(ConversationKnowledgeRun)
            .where(
                ConversationKnowledgeRun.run_id == run_id,
                ConversationKnowledgeRun.revision == row.revision,
            )
            .values(audits=[*row.audits, value], revision=row.revision + 1)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise KnowledgeBranchError("knowledge_audit_conflict", True)
        self.session.expire(row)

    def add_review(self, run_id, epoch, item):
        row = self._owned(run_id, epoch)
        item = KnowledgeReviewItem.model_validate_json(item.model_dump_json())
        if (
            item.run_id != run_id
            or item.conversation_id != row.conversation_id
            or item.understanding_id != row.understanding_id
            or item.primary_task_id != row.primary_task_id
        ):
            raise KnowledgeBranchError("knowledge_review_conflict")
        value = item.model_dump(mode="json")
        old = self.session.get(ConversationKnowledgeReviewItem, run_id)
        if old is not None:
            if old.item_json != value or old.item_hash != canonical_hash(value):
                raise KnowledgeBranchError("knowledge_review_conflict")
            return
        self.session.add(
            ConversationKnowledgeReviewItem(
                run_id=run_id,
                item_json=value,
                item_hash=canonical_hash(value),
                created_at=item.created_at,
            )
        )
        self.session.flush()

    def record_error(self, run_id, epoch, code, retryable):
        row = self._owned(run_id, epoch)
        row.error_code, row.retryable = code, retryable
        self.session.flush()
