"""Caller-owned short transactions; writes fenced by owner and epoch."""

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select

from resolv.db.conversation_base import utc_now
from resolv.models.conversation import (
    ConversationAction,
    ConversationMessage,
    ConversationRun,
)
from resolv.models.understanding import ConversationUnderstanding as U
from resolv.models.understanding import ConversationUnderstandingRequest as R
from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.schemas.conversation import ConversationError, ConversationRecord
from resolv.schemas.understanding import (
    EvidenceSource,
    IntentStageResult,
    ResolutionStageResult,
    SnapshotSources,
    UnderstandingError,
    UnderstandingRecord,
    UnderstandingRequestRecord,
    UnderstandingView,
)
from resolv.understanding_rules import INTENT_BRANCHES_V1, referenced_task

TERMINAL = {"ready", "needs_clarification"}


def record(type_, row):
    value = {name: getattr(row, name) for name in type_.model_fields}
    return type_.model_validate_json(
        json.dumps(
            value, default=lambda v: v.isoformat() if isinstance(v, datetime) else v
        )
    )


def source(kind, source_id, *, text=None, json_value=None, observed_at=None):
    value = {
        "kind": kind,
        "source_id": source_id,
        "text": text,
        "json_value": json_value,
        "observed_at": observed_at.isoformat() if observed_at else None,
    }
    return EvidenceSource.model_validate_json(
        json.dumps(value | {"source_hash": fingerprint(value)}, ensure_ascii=False)
    )


class UnderstandingRepository:
    def __init__(self, session, owner_id="local"):
        self.session, self.owner_id = session, owner_id
        self.chat = ConversationRepository(session, owner_id)

    def conversation(self, conversation_id, *, active=False, lock=False):
        try:
            c = self.chat.conversation(conversation_id, lock=lock)
        except ConversationError:
            raise UnderstandingError("conversation_not_found") from None
        if active and c.status != "active":
            raise UnderstandingError("conversation_archived")
        return c

    def row(self, id, conversation_id=None):
        row = self.session.get(U, id)
        if row is None or (
            conversation_id is not None and row.conversation_id != conversation_id
        ):
            raise UnderstandingError("understanding_not_found")
        self.conversation(row.conversation_id)
        return row

    def get(self, conversation_id, id):
        self.conversation(conversation_id)
        return record(UnderstandingRecord, self.row(id, conversation_id))

    def task(self, conversation_id, id, *, selectable=False):
        self.conversation(conversation_id)
        row = self.session.scalar(
            select(R)
            .where(R.task_id == id, R.conversation_id == conversation_id)
            .with_for_update()
        )
        if row is None:
            raise UnderstandingError("task_not_found")
        parent = self.row(row.understanding_id, conversation_id)
        if selectable and (
            row.queue_status != "pending" or parent.status not in TERMINAL
        ):
            raise UnderstandingError(
                "task_conflict", related_id=row.selected_by_understanding_id
            )
        return row

    def list_tasks(self, conversation_id, status, cursor=None, limit=50):
        self.conversation(conversation_id)
        if status not in {"pending", "selected"} or not 1 <= limit <= 100:
            raise UnderstandingError("invalid_request")
        statement = (
            select(R)
            .join(U, U.understanding_id == R.understanding_id)
            .where(
                R.conversation_id == conversation_id,
                R.queue_status == status,
                U.status.in_(TERMINAL),
            )
        )
        if cursor:
            statement = statement.where(R.task_id > cursor)
        values = []
        for row in self.session.scalars(
            statement.order_by(R.task_id).limit(limit)
        ).all():
            item = record(UnderstandingRequestRecord, row)
            if status == "selected":
                selected = self.row(row.selected_by_understanding_id, conversation_id)
                if selected.status not in TERMINAL:
                    continue
                item = UnderstandingRequestRecord.model_validate_json(
                    json.dumps(selected.final_json["primary_request"])
                )
            values.append(item)
        return values

    def create(self, conversation_id, request, freeze):
        c = self.conversation(conversation_id, active=True, lock=True)
        digest = fingerprint(request.model_dump(mode="json"))
        existing = self.session.scalar(
            select(U).where(
                U.conversation_id == conversation_id, U.request_id == request.request_id
            )
        )
        if existing:
            if existing.request_fingerprint != digest:
                raise UnderstandingError(
                    "request_conflict", understanding_id=existing.understanding_id
                )
            return record(UnderstandingRecord, existing), False
        parent = None
        parent_row = None
        if request.clarifies_understanding_id:
            parent_row = self.row(request.clarifies_understanding_id, conversation_id)
            if parent_row.status != "needs_clarification":
                raise UnderstandingError("clarification_conflict")
            if parent_row.clarification_child_id:
                raise UnderstandingError(
                    "clarification_conflict",
                    related_id=parent_row.clarification_child_id,
                )
            parent = record(UnderstandingRecord, parent_row).model_dump(mode="json")
        if request.continue_task_id:
            task = self.task(conversation_id, request.continue_task_id, selectable=True)
            pending = [record(UnderstandingRequestRecord, task)]
        else:
            # Freeze ALL candidates, never a pagination subset.
            pending = [
                record(UnderstandingRequestRecord, r)
                for r in self.session.scalars(
                    select(R)
                    .join(U, U.understanding_id == R.understanding_id)
                    .where(
                        R.conversation_id == conversation_id,
                        R.queue_status == "pending",
                        U.status.in_(TERMINAL),
                    )
                    .order_by(R.task_id)
                )
            ]
        turns = self.chat.completed_turns(conversation_id)
        if any(
            self.session.get(ConversationRun, turn.user.run_id).conversation_id
            != conversation_id
            for turn in turns
        ):
            raise UnderstandingError("snapshot_invalid")
        summary = self.chat.summary(conversation_id)
        sources = SnapshotSources(
            conversation=ConversationRecord.model_validate(c),
            turns=turns,
            summary=summary,
            pending_requests=pending,
            parent=parent,
            linked_inputs={
                t.understanding_id: self.row(
                    t.understanding_id, conversation_id
                ).original_text
                for t in pending
            },
        )
        snapshot = freeze(sources)
        if (
            snapshot.conversation_id != conversation_id
            or snapshot.owner_id != self.owner_id
            or snapshot.original_text != request.text
        ):
            raise UnderstandingError("snapshot_invalid")
        row = U(
            understanding_id=uuid4().hex,
            conversation_id=conversation_id,
            request_id=request.request_id,
            request_fingerprint=digest,
            original_text=request.text,
            clarifies_understanding_id=request.clarifies_understanding_id,
            continue_task_id=request.continue_task_id,
            snapshot=snapshot.model_dump(mode="json"),
        )
        self.session.add(row)
        self.session.flush()
        if parent_row:
            parent_row.clarification_child_id = row.understanding_id
        self.session.flush()
        return record(UnderstandingRecord, row), True

    def owned(self, id, epoch, execution_owner_id, *, allow_archive=False):
        initial = self.row(id)
        self.conversation(initial.conversation_id, active=not allow_archive, lock=True)
        row = self.session.scalar(
            select(U)
            .where(U.understanding_id == id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if (
            row.execution_epoch != epoch
            or row.execution_owner_id != execution_owner_id
            or execution_owner_id is None
        ):
            raise UnderstandingError("understanding_busy", True, id)
        return row

    def claim(self, id, execution_owner_id):
        initial = self.row(id)
        self.conversation(initial.conversation_id, active=True, lock=True)
        row = self.session.scalar(
            select(U)
            .where(U.understanding_id == id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row.status in TERMINAL:
            return record(UnderstandingRecord, row)
        # Only the holder of the real named lock calls claim. A crashed owner can be replaced.
        row.execution_epoch += 1
        row.attempt += 1
        row.execution_owner_id = execution_owner_id
        if row.status != "finalizing":
            row.status = "running"
        row.error_code, row.retryable = None, False
        row.invocations = [
            a | {"outcome": "unknown"} if a["outcome"] == "started" else a
            for a in row.invocations
        ]
        self.session.flush()
        return record(UnderstandingRecord, row)

    def record_invocation(self, id, epoch, execution_owner_id, audit):
        row = self.owned(id, epoch, execution_owner_id)
        values = [
            a for a in row.invocations if a["invocation_id"] != audit.invocation_id
        ]
        row.invocations = [*values, audit.model_dump(mode="json")]
        self.session.flush()

    def save_stage(self, id, epoch, execution_owner_id, stage, output, audit):
        row = self.owned(id, epoch, execution_owner_id)
        field, type_ = (
            ("resolution", ResolutionStageResult)
            if stage == "resolution"
            else ("classification", IntentStageResult)
        )
        if stage not in {"resolution", "intents"}:
            raise UnderstandingError("invalid_stage")
        if row.status in TERMINAL or row.final_json:
            raise UnderstandingError("understanding_state_conflict")
        if getattr(row, field) is None:
            setattr(
                row, field, type_(output=output, audit=audit).model_dump(mode="json")
            )
        self.session.flush()
        return record(UnderstandingRecord, row)

    def publish(self, id, epoch, execution_owner_id, selection):
        row = self.owned(id, epoch, execution_owner_id)
        if row.final_json is not None:
            return record(UnderstandingRecord, row)
        current = record(UnderstandingRecord, row)
        if current.resolution is None or current.classification is None:
            raise UnderstandingError("understanding_state_conflict")
        classified = current.classification.output.requests
        primary, secondary = None, []
        for index, request in enumerate(classified):
            is_primary = index == selection.primary_index
            unresolved = [
                u
                for u in current.resolution.output.unresolved_references
                if any(
                    ref.kind == "input"
                    and ref.source_id == "current"
                    and ref.span
                    and ref.span.start < u.span.end
                    and u.span.start < ref.span.end
                    for ref in request.evidence_refs
                )
            ]
            old_id = referenced_task(request, current.snapshot)
            if old_id:
                old = self.task(row.conversation_id, old_id, selectable=is_primary)
                if is_primary:
                    if selection.continuation_task_id != old_id:
                        raise UnderstandingError("continuation_conflict")
                    old.queue_status, old.selected_by_understanding_id = "selected", id
                item = record(UnderstandingRequestRecord, old)
                if is_primary:
                    # Preserve the original queue record as task provenance.
                    # The child's durable result carries its current meaning,
                    # with the same stable task ID and no inherited approval.
                    item = item.model_copy(
                        update={
                            "understanding_id": id,
                            "ordinal": index,
                            "role": "primary",
                            "text": request.text,
                            "intent": request.intent,
                            "branch": INTENT_BRANCHES_V1.get(request.intent),
                            "evidence_refs": request.evidence_refs,
                            "bindings": [
                                current.resolution.output.bindings[i]
                                for i in request.binding_indexes
                            ],
                            "ambiguities": request.ambiguities,
                            "unresolved_references": unresolved,
                        }
                    )
            elif is_primary or index in selection.secondary_indexes:
                item = UnderstandingRequestRecord(
                    task_id=uuid4().hex,
                    understanding_id=id,
                    conversation_id=row.conversation_id,
                    ordinal=index,
                    role="primary" if is_primary else "secondary",
                    text=request.text,
                    intent=request.intent,
                    branch=INTENT_BRANCHES_V1.get(request.intent),
                    evidence_refs=request.evidence_refs,
                    bindings=[
                        current.resolution.output.bindings[i]
                        for i in request.binding_indexes
                    ],
                    ambiguities=request.ambiguities,
                    unresolved_references=unresolved,
                    queue_status="not_queued" if is_primary else "pending",
                    selected_by_understanding_id=None,
                    created_at=utc_now(),
                )
                data = item.model_dump(mode="json")
                data["created_at"] = item.created_at
                self.session.add(R(**data))
            else:
                continue
            if is_primary:
                primary = item
            else:
                secondary.append(item)
        status = "ready" if selection.route_ready else "needs_clarification"
        view = UnderstandingView(
            schema_version=1,
            understanding_id=id,
            conversation_id=row.conversation_id,
            status=status,
            original_text=row.original_text,
            resolution=current.resolution.output,
            primary_request=primary,
            branch=selection.branch,
            route_ready=selection.route_ready,
            pending_requests=secondary,
            clarification_question=selection.clarification_question,
            context_refs=current.snapshot.source_fingerprints,
            versions={
                "resolution": current.snapshot.config.resolution_version,
                "intents": current.snapshot.config.intent_version,
                "mapping": current.snapshot.config.mapping_version,
                "schema": 1,
            },
            stage_results={
                "resolution": current.resolution.model_dump(mode="json"),
                "intents": current.classification.model_dump(mode="json"),
            },
            followup_ids={
                "parent": row.clarifies_understanding_id,
                "clarification_child": row.clarification_child_id,
                "continued_task": selection.continuation_task_id,
            },
            error_code=None,
            retryable=False,
        )
        row.final_json, row.status = view.model_dump(mode="json"), "finalizing"
        self.session.flush()
        return record(UnderstandingRecord, row)

    def finish(self, id, epoch, execution_owner_id, terminal_status):
        row = self.owned(id, epoch, execution_owner_id)
        if (
            terminal_status not in TERMINAL
            or not row.final_json
            or row.final_json["status"] != terminal_status
        ):
            raise UnderstandingError("understanding_state_conflict")
        row.status, row.error_code, row.retryable = terminal_status, None, False
        self.session.flush()
        return record(UnderstandingRecord, row)

    def record_error(self, id, epoch, execution_owner_id, error):
        row = self.owned(id, epoch, execution_owner_id, allow_archive=True)
        if row.status in TERMINAL:
            return
        row.status = (
            "finalizing"
            if row.final_json and error.code != "conversation_archived"
            else ("interrupted" if error.retryable else "failed")
        )
        row.error_code, row.retryable = error.code, error.retryable
        self.session.flush()

    def release(self, id, epoch, execution_owner_id):
        row = self.owned(id, epoch, execution_owner_id, allow_archive=True)
        row.execution_owner_id = None
        self.session.flush()

    def load_source(self, conversation_id, ref, cutoff_seq):
        self.conversation(conversation_id)
        if ref.kind == "message":
            message = self.session.get(ConversationMessage, ref.source_id)
            run = self.session.get(ConversationRun, message.run_id) if message else None
            if (
                not message
                or message.conversation_id != conversation_id
                or message.seq > cutoff_seq
                or not run
                or run.conversation_id != conversation_id
                or run.status != "completed"
            ):
                raise UnderstandingError("source_invalid")
            return source(
                "message",
                ref.source_id,
                text=message.text,
                observed_at=message.created_at,
            )
        if ref.kind == "action":
            action = self.session.get(ConversationAction, ref.source_id)
            run = self.session.get(ConversationRun, action.run_id) if action else None
            answer = (
                self.session.get(ConversationMessage, run.answer_message_id)
                if run
                else None
            )
            if (
                not action
                or not run
                or run.conversation_id != conversation_id
                or run.status != "completed"
                or not answer
                or answer.seq > cutoff_seq
                or action.execution_status != "completed"
                or action.result is None
            ):
                raise UnderstandingError("source_invalid")
            return source(
                "action",
                ref.source_id,
                json_value=action.result,
                observed_at=action.completed_at,
            )
        raise UnderstandingError("source_invalid")
