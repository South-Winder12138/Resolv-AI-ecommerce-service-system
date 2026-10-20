"""Conversation operations inside caller-owned short transactions."""

import json
from hashlib import sha256
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import select

from resolv.db.conversation_base import utc_now
from resolv.models.conversation import (
    Conversation,
    ConversationAction,
    ConversationMessage,
    ConversationRun,
    ConversationSummaryRecord,
)
from resolv.schemas.conversation import (
    ActionRecord,
    CompletedTurn,
    ConversationError,
    MessageRecord,
    RunRecord,
    SummaryRecord,
)
from resolv.tools.registry import ToolExecutionError, get_registered_tool


def canonical_json(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ConversationError("invalid_request") from None


def fingerprint(value):
    return sha256(canonical_json(value).encode()).hexdigest()


class ConversationRepository:
    def __init__(self, session, owner_id="local"):
        self.session, self.owner_id = session, owner_id

    def conversation(self, conversation_id, *, lock=False):
        statement = select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.owner_id == self.owner_id,
        )
        if lock:
            statement = statement.with_for_update()
        row = self.session.scalar(statement)
        if row is None:
            raise ConversationError("conversation_not_found")
        return row

    def run_row(self, run_id):
        row = self.session.get(ConversationRun, run_id)
        if row is None:
            raise ConversationError("run_state_conflict", run_id=run_id)
        self.conversation(row.conversation_id)
        return row

    def run(self, run_id):
        row = self.run_row(run_id)
        record = RunRecord.model_validate(row)
        message = self.session.get(ConversationMessage, row.question_message_id)
        if message is None or message.run_id != run_id or message.role != "user":
            raise ConversationError("run_state_conflict", run_id=run_id)
        return record.model_copy(update={"question": message.text})

    def owned(self, run_id, epoch, owner_id=None):
        initial = self.run_row(run_id)
        conversation = self.conversation(initial.conversation_id, lock=True)
        row = self.session.scalar(
            select(ConversationRun)
            .where(ConversationRun.run_id == run_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if (
            conversation.current_run_id != run_id
            or row.execution_epoch != epoch
            or not row.execution_owner_id
            or (owner_id is not None and row.execution_owner_id != owner_id)
        ):
            raise ConversationError("conversation_busy", True, run_id)
        return row

    def action(self, action_id):
        row = self.session.get(ConversationAction, action_id)
        if row is None:
            raise ConversationError("action_conflict")
        self.run_row(row.run_id)
        return ActionRecord.model_validate(row)

    def stage_action(
        self, run_id, epoch, model_step, tool_call_id, tool_name, effect, arguments
    ):
        self.owned(run_id, epoch)
        existing = self.session.scalar(
            select(ConversationAction).where(
                ConversationAction.run_id == run_id,
                ConversationAction.model_step == model_step,
                ConversationAction.tool_call_id == tool_call_id,
            )
        )
        try:
            registered = get_registered_tool(tool_name)
            if (
                registered.effect.value != effect
                or not isinstance(arguments, dict)
                or set(arguments) - set(registered.tool.args_schema.model_fields)
            ):
                raise ValueError("invalid tool arguments or effect")
            canonical_json(arguments)
            validated = registered.tool.args_schema.model_validate(
                arguments
            ).model_dump()
        except (
            ValidationError,
            ToolExecutionError,
            ConversationError,
            ValueError,
            TypeError,
        ):
            raise ConversationError(
                "action_conflict" if existing else "invalid_request", run_id=run_id
            ) from None
        digest = fingerprint(validated)
        if existing:
            if (
                existing.tool_name != tool_name
                or existing.effect != effect
                or existing.arguments_fingerprint != digest
            ):
                raise ConversationError("action_conflict", run_id=run_id)
            return ActionRecord.model_validate(existing)
        if effect == "write":
            pending = self.session.scalar(
                select(ConversationAction).where(
                    ConversationAction.run_id == run_id,
                    ConversationAction.approval_status == "pending",
                )
            )
            if pending:
                raise ConversationError("pending_approval_exists", run_id=run_id)
        row = ConversationAction(
            action_id=uuid4().hex,
            run_id=run_id,
            model_step=model_step,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            effect=effect,
            arguments=validated,
            arguments_fingerprint=digest,
            approval_id=uuid4().hex if effect == "write" else None,
            approval_status="pending" if effect == "write" else "not_required",
        )
        self.session.add(row)
        self.session.flush()
        return ActionRecord.model_validate(row)

    def save_action_result(self, action_id, epoch, result):
        record = self.action(action_id)
        self.owned(record.run_id, epoch)
        canonical_json(result)
        row = self.session.get(ConversationAction, action_id)
        if row.execution_status == "completed":
            if canonical_json(row.result) != canonical_json(result):
                raise ConversationError("action_conflict", run_id=row.run_id)
        else:
            row.result, row.execution_status, row.completed_at = (
                result,
                "completed",
                utc_now(),
            )
        self.session.flush()
        return ActionRecord.model_validate(row)

    def summary(self, conversation_id, version=None):
        if version == 0:
            return None
        statement = select(ConversationSummaryRecord).where(
            ConversationSummaryRecord.conversation_id == conversation_id
        )
        statement = (
            statement.where(ConversationSummaryRecord.version == version)
            if version is not None
            else statement.order_by(ConversationSummaryRecord.version.desc()).limit(1)
        )
        row = self.session.scalar(statement)
        if row is None and version is not None:
            raise ConversationError("run_state_conflict")
        return SummaryRecord.model_validate(row) if row else None

    def completed_turns(self, conversation_id, *, after_seq=0, cutoff_seq=None):
        statement = (
            select(ConversationMessage)
            .join(ConversationRun, ConversationMessage.run_id == ConversationRun.run_id)
            .where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationRun.status == "completed",
                ConversationMessage.seq > after_seq,
            )
        )
        if cutoff_seq is not None:
            statement = statement.where(ConversationMessage.seq <= cutoff_seq)
        rows = self.session.scalars(statement.order_by(ConversationMessage.seq)).all()
        users = {}
        turns = []
        for row in rows:
            if row.role == "user":
                users[row.run_id] = row
            elif row.run_id in users:
                actions = self.session.scalars(
                    select(ConversationAction).where(
                        ConversationAction.run_id == row.run_id
                    )
                ).all()
                turns.append(
                    CompletedTurn(
                        user=MessageRecord.model_validate(users[row.run_id]),
                        assistant=MessageRecord.model_validate(row),
                        actions=[ActionRecord.model_validate(a) for a in actions],
                    )
                )
        return turns

    def publish_memory(
        self, run_id, epoch, candidate, summary_version, history_cutoff_seq
    ):
        row = self.owned(run_id, epoch)
        if row.memory_prepared:
            return self.run(run_id)
        if candidate is not None:
            latest = self.summary(row.conversation_id)
            previous = latest.version if latest else 0
            if candidate.previous_version != previous:
                raise ConversationError("run_state_conflict", run_id=run_id)
            summary_version = previous + 1
            self.session.add(
                ConversationSummaryRecord(
                    conversation_id=row.conversation_id,
                    version=summary_version,
                    content=candidate.content.model_dump(),
                    covered_until_seq=candidate.covered_until_seq,
                    model_name=candidate.model_name,
                    source_run_id=run_id,
                )
            )
        else:
            self.summary(row.conversation_id, summary_version)
        row.summary_version, row.history_cutoff_seq, row.memory_prepared = (
            summary_version,
            history_cutoff_seq,
            True,
        )
        self.session.flush()
        return self.run(run_id)

    def save_knowledge(self, run_id, epoch, hits):
        row = self.owned(run_id, epoch)
        values = [
            hit.model_dump() if hasattr(hit, "model_dump") else hit for hit in hits
        ]
        canonical_json(values)
        if row.knowledge_prepared and canonical_json(
            row.knowledge_snapshot
        ) != canonical_json(values):
            raise ConversationError("run_state_conflict", run_id=run_id)
        row.knowledge_snapshot, row.knowledge_prepared = values, True
        self.session.flush()
        return self.run(run_id)

    def commit_answer(self, run_id, epoch, text, *, metadata=None):
        run = self.owned(run_id, epoch)
        from resolv.schemas.branch_execution import ReplyMetadata, parse_reply_metadata
        from resolv.schemas.knowledge_branch import ReplyMetadataV2
        from resolv.schemas.refund_submission import ReplyMetadataV3

        if (
            run.orchestration_version in {"routed-v2", "routed-v3", "routed-v4"}
            and metadata is None
        ):
            raise ConversationError("run_state_conflict", run_id=run_id)
        if metadata is not None and not isinstance(
            metadata, (ReplyMetadata, ReplyMetadataV2, ReplyMetadataV3)
        ):
            raise ConversationError("run_state_conflict", run_id=run_id)
        if metadata is not None:
            try:
                data = metadata.model_dump(mode="json")
                if run.orchestration_version == "routed-v3" and isinstance(
                    metadata, ReplyMetadata
                ):
                    data.update(schema_version=2, knowledge=None)
                if run.orchestration_version == "routed-v4" and not isinstance(
                    metadata, ReplyMetadataV3
                ):
                    data.update(
                        schema_version=3, knowledge=data.get("knowledge"), refund=None
                    )
                metadata = parse_reply_metadata(data)
                expected = {"routed-v2": 1, "routed-v3": 2, "routed-v4": 3}.get(
                    run.orchestration_version, 1
                )
                if metadata.schema_version != expected:
                    raise ValueError("Metadata version does not match run")
            except ValueError as exc:
                raise ConversationError("run_state_conflict", run_id=run_id) from exc
        result = metadata.model_dump(mode="json") if metadata else None
        old = self.session.get(ConversationMessage, run.answer_message_id)
        if old:
            if (
                old.run_id != run_id
                or old.role != "assistant"
                or old.text != text
                or canonical_json(run.result_json) != canonical_json(result)
            ):
                raise ConversationError("run_state_conflict", run_id=run_id)
            return MessageRecord.model_validate(old)
        conversation = self.conversation(run.conversation_id, lock=True)
        answer = ConversationMessage(
            message_id=run.answer_message_id,
            conversation_id=run.conversation_id,
            run_id=run_id,
            seq=conversation.next_message_seq,
            role="assistant",
            text=text,
        )
        self.session.add(answer)
        run.result_json = result
        conversation.next_message_seq += 1
        run.status = "finalizing"
        self.session.flush()
        return MessageRecord.model_validate(answer)

    def record_model_audit(self, run_id, epoch, audit):
        from resolv.schemas.branch_execution import BusinessAudit

        run = self.owned(run_id, epoch)
        if not isinstance(audit, BusinessAudit):
            raise ConversationError("run_state_conflict", run_id=run_id)
        data = audit.model_dump(mode="json")
        old = next(
            (a for a in run.model_audits if a["invocation_id"] == audit.invocation_id),
            None,
        )
        if old is not None:
            if canonical_json(old) != canonical_json(data):
                raise ConversationError("run_state_conflict", run_id=run_id)
            return
        run.model_audits = [*run.model_audits, data]
        self.session.flush()

    def mark_completed(self, run_id, epoch):
        run = self.owned(run_id, epoch)
        answer = self.session.get(ConversationMessage, run.answer_message_id)
        if run.status != "finalizing" or answer is None or answer.run_id != run_id:
            raise ConversationError("run_state_conflict", run_id=run_id)
        run.status, run.error_code, run.retryable = "completed", None, False
        self.conversation(run.conversation_id, lock=True).current_run_id = None
        self.session.flush()
        return self.run(run_id)
