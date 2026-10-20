"""Caller-owned short transactions; this repository never dispatches HTTP."""

from uuid import uuid4

from pydantic import TypeAdapter
from sqlalchemy import select

from resolv.db.conversation_base import utc_now
from resolv.models.conversation import ConversationAction
from resolv.models.logistics import ConversationLogisticsAttempt as Attempt
from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.schemas.conversation import ConversationError
from resolv.schemas.logistics import (
    Digest,
    LogisticsFailure,
    ShipmentEnvelope,
    TrackingNo,
)
from resolv.schemas.logistics_execution import LogisticsAttemptRecord


class LogisticsAttemptRepository:
    def __init__(self, session, owner_id):
        self.session, self.owner_id = session, owner_id
        self.conversations = ConversationRepository(session, owner_id)

    def _owned_action(self, action_id, epoch):
        action = self.session.get(ConversationAction, action_id)
        if action is None:
            raise ConversationError("action_conflict")
        run = self.conversations.owned(action.run_id, epoch)
        action = self.session.scalar(select(ConversationAction).where(
            ConversationAction.action_id == action_id,
        ).with_for_update().execution_options(populate_existing=True))
        if run.lease_expires_at is None or run.lease_expires_at <= utc_now() or run.status != "running":
            raise ConversationError("conversation_busy", True, run.run_id)
        if action.tool_name != "query_logistics" or action.effect != "read":
            raise ConversationError("business_scope_violation", run_id=run.run_id)
        return action, run

    def prepare(self, action_id, epoch, *, config_hash, tracking_no):
        TypeAdapter(Digest).validate_python(config_hash, strict=True)
        TypeAdapter(TrackingNo).validate_python(tracking_no, strict=True)
        action, run = self._owned_action(action_id, epoch)
        if action.arguments != {"tracking_no": tracking_no} or action.arguments_fingerprint != fingerprint(action.arguments):
            raise LogisticsFailure("logistics_result_invalid")
        if not run.business_config or config_hash != fingerprint(run.business_config):
            raise LogisticsFailure("logistics_configuration_conflict")
        if action.execution_status != "pending":
            raise LogisticsFailure("logistics_store_failed")
        existing = self.session.scalar(select(Attempt).where(
            Attempt.action_id == action_id, Attempt.state == "prepared",
        ).with_for_update())
        if existing:
            raise LogisticsFailure("logistics_result_unknown", True)
        latest = self.session.scalar(select(Attempt.attempt_no).where(
            Attempt.action_id == action_id,
        ).order_by(Attempt.attempt_no.desc()).limit(1).with_for_update()) or 0
        row = Attempt(
            attempt_id=uuid4().hex, action_id=action_id, run_id=run.run_id,
            conversation_id=run.conversation_id, owner_id=self.owner_id,
            execution_epoch=epoch, attempt_no=latest + 1, config_hash=config_hash,
            arguments_hash=fingerprint(action.arguments), tracking_no=tracking_no,
        )
        self.session.add(row)
        self.session.flush()
        return LogisticsAttemptRecord.model_validate(row)

    def _pending(self, attempt_id, epoch, *, allow_previous_epoch=False):
        row = self.session.get(Attempt, attempt_id)
        if row is None or row.owner_id != self.owner_id:
            raise LogisticsFailure("logistics_store_failed")
        action, run = self._owned_action(row.action_id, epoch)
        row = self.session.scalar(select(Attempt).where(
            Attempt.attempt_id == attempt_id,
        ).with_for_update().execution_options(populate_existing=True))
        if (
            row.run_id != run.run_id or row.conversation_id != run.conversation_id
            or (not allow_previous_epoch and row.execution_epoch != epoch)
            or row.state != "prepared" or action.execution_status != "pending"
        ):
            raise LogisticsFailure("logistics_store_failed")
        return row, action

    @staticmethod
    def _finish(row, state, error_code=None):
        row.state, row.error_code, row.finished_at = state, error_code, utc_now()
        row.elapsed_ms = max(0, int((row.finished_at - row.started_at).total_seconds() * 1000))

    def complete(self, attempt_id, epoch, envelope: ShipmentEnvelope, projection: dict):
        row, action = self._pending(attempt_id, epoch)
        if envelope.request_id != attempt_id or envelope.tracking_no != row.tracking_no:
            raise LogisticsFailure("logistics_result_invalid")
        # Canonicalization also rejects non-JSON/non-finite projection data.
        fingerprint(projection)
        payload = envelope.model_dump(mode="json")
        row.envelope, row.projection, row.result_hash = payload, projection, fingerprint(payload)
        self._finish(row, "completed")
        action.result, action.execution_status, action.completed_at = projection, "completed", row.finished_at
        self.session.flush()

    def fail(self, attempt_id, epoch, error: LogisticsFailure):
        row, _ = self._pending(attempt_id, epoch)
        self._finish(row, "failed", error.code)
        self.session.flush()

    def mark_unknown(self, attempt_id, epoch):
        row, _ = self._pending(attempt_id, epoch, allow_previous_epoch=True)
        self._finish(row, "unknown", "logistics_result_unknown")
        self.session.flush()

    def find_completed(self, run_id, config_hash, tracking_no):
        self.conversations.run_row(run_id)
        row = self.session.scalar(select(Attempt).where(
            Attempt.run_id == run_id, Attempt.owner_id == self.owner_id,
            Attempt.config_hash == config_hash, Attempt.tracking_no == tracking_no,
            Attempt.state == "completed",
        ).order_by(Attempt.started_at, Attempt.attempt_id).limit(1).with_for_update())
        return LogisticsAttemptRecord.model_validate(row) if row else None

