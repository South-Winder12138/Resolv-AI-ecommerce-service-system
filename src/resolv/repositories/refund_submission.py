"""Independent submission commands and aliases; never claim a chat run lease."""

from datetime import timedelta
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from resolv.db.conversation_base import utc_now
from resolv.models.conversation import Conversation
from resolv.models.refund_flow import (
    ConversationRefundOffer,
    ConversationRefundRequestAlias,
    ConversationRefundSubmission,
)
from resolv.repositories.refund_flow import RefundFlowRepository, checked
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_submission import (
    RefundReceipt,
    RefundRevalidation,
    RefundRevalidationAudit,
    RefundSubmissionRecord,
    SubmissionLease,
)
from resolv.services.hybrid_fingerprints import canonical_hash


class RefundSubmissionRepository:
    def __init__(self, session, owner_id, *, clock=utc_now):
        self.session, self.owner_id, self.clock = session, owner_id, clock

    def _row(self, command_id, *, lock=False):
        query = (
            select(ConversationRefundSubmission)
            .join(Conversation)
            .where(
                ConversationRefundSubmission.command_id == command_id,
                Conversation.owner_id == self.owner_id,
                ConversationRefundSubmission.owner_id == self.owner_id,
            )
            .execution_options(populate_existing=True)
        )
        return self.session.scalar(query.with_for_update() if lock else query)

    def _record(self, row):
        offer = RefundFlowRepository(self.session, self.owner_id).get_offer(
            row.offer_id
        )
        if offer is None or offer.conversation_id != row.conversation_id:
            raise RefundFlowError("refund_submission_conflict")
        if (row.receipt is None) != (row.receipt_hash is None) or (
            row.receipt is not None and canonical_hash(row.receipt) != row.receipt_hash
        ):
            raise RefundFlowError("refund_receipt_conflict")
        results, audits = [], []
        for entry in row.revalidations:
            if "audit" in entry:
                if canonical_hash(entry["audit"]) != entry.get("hash"):
                    raise RefundFlowError("refund_revalidation_conflict")
                audits.append(checked(RefundRevalidationAudit, entry["audit"]))
            else:
                results.append(checked(RefundRevalidation, entry))
        return RefundSubmissionRecord(
            submission_id=row.command_id,
            offer_id=row.offer_id,
            owner_id=row.owner_id,
            conversation_id=row.conversation_id,
            request_hash=row.request_hash,
            status=row.status,
            execution_epoch=row.execution_epoch,
            execution_owner_id=row.execution_owner_id,
            lease_expires_at=row.lease_expires_at,
            revalidations=tuple(results),
            revalidation_audits=tuple(audits),
            receipt=checked(RefundReceipt, row.receipt) if row.receipt else None,
            error_code=row.error_code,
            retryable=row.retryable,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def get(self, command_id):
        row = self._row(command_id)
        return self._record(row) if row else None

    def get_by_offer(self, offer_id):
        row = self.session.scalar(
            select(ConversationRefundSubmission).where(
                ConversationRefundSubmission.offer_id == offer_id,
                ConversationRefundSubmission.owner_id == self.owner_id,
            )
        )
        return self.get(row.command_id) if row else None

    def accept(self, offer_id, request_id, request_hash):
        if (
            not isinstance(request_id, str)
            or not request_id.strip()
            or len(request_id) > 128
        ):
            raise RefundFlowError("refund_request_invalid")
        offer = RefundFlowRepository(self.session, self.owner_id).get_offer(offer_id)
        if offer is None:
            raise RefundFlowError("refund_not_found")
        # Serialize offers in MySQL. SQLite callers use short transactions; unique
        # keys remain the final arbiter for both command and request aliases.
        self.session.scalar(
            select(ConversationRefundOffer)
            .where(ConversationRefundOffer.offer_id == offer_id)
            .with_for_update()
        )
        alias = self.session.get(
            ConversationRefundRequestAlias,
            (self.owner_id, request_id),
            populate_existing=True,
        )
        if alias:
            old = self.get(alias.command_id)
            if (
                old is None
                or old.offer_id != offer_id
                or alias.request_hash != request_hash
            ):
                raise RefundFlowError("refund_request_conflict")
            return old
        row = self.session.scalar(
            select(ConversationRefundSubmission).where(
                ConversationRefundSubmission.offer_id == offer_id
            )
        )
        if row is None:
            try:
                with self.session.begin_nested():
                    row = ConversationRefundSubmission(
                        command_id=uuid4().hex,
                        offer_id=offer_id,
                        owner_id=self.owner_id,
                        conversation_id=offer.conversation_id,
                        request_hash=request_hash,
                    )
                    self.session.add(row)
                    self.session.flush()
            except IntegrityError:
                row = self.session.scalar(
                    select(ConversationRefundSubmission).where(
                        ConversationRefundSubmission.offer_id == offer_id
                    )
                )
                if row is None:
                    raise
        if row.owner_id != self.owner_id or row.request_hash != request_hash:
            raise RefundFlowError("refund_request_conflict")
        try:
            with self.session.begin_nested():
                self.session.add(
                    ConversationRefundRequestAlias(
                        owner_id=self.owner_id,
                        request_id=request_id,
                        command_id=row.command_id,
                        request_hash=request_hash,
                    )
                )
                self.session.flush()
        except IntegrityError:
            alias = self.session.get(
                ConversationRefundRequestAlias,
                (self.owner_id, request_id),
                populate_existing=True,
            )
            if (
                alias is None
                or alias.command_id != row.command_id
                or alias.request_hash != request_hash
            ):
                raise RefundFlowError("refund_request_conflict") from None
        return self._record(row)

    def claim(self, command_id, worker_id, now, lease_seconds):
        row = self._row(command_id, lock=True)
        if row is None:
            raise RefundFlowError("refund_not_found")
        if not worker_id or lease_seconds <= 0:
            raise RefundFlowError("refund_submission_conflict")
        result = self.session.execute(
            update(ConversationRefundSubmission)
            .where(
                ConversationRefundSubmission.command_id == command_id,
                ConversationRefundSubmission.owner_id == self.owner_id,
                ConversationRefundSubmission.execution_epoch == row.execution_epoch,
                ConversationRefundSubmission.status != "completed",
                or_(
                    ConversationRefundSubmission.status.in_(["created", "interrupted"]),
                    (ConversationRefundSubmission.status == "failed")
                    & ConversationRefundSubmission.retryable.is_(True),
                    (ConversationRefundSubmission.status == "running")
                    & (ConversationRefundSubmission.lease_expires_at <= now),
                ),
            )
            .values(
                status="running",
                execution_epoch=row.execution_epoch + 1,
                execution_owner_id=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise RefundFlowError("refund_submission_busy", True)
        self.session.expire(row)
        return SubmissionLease(
            command_id=command_id,
            owner_id=self.owner_id,
            worker_id=worker_id,
            epoch=row.execution_epoch,
        )

    def _owned_row(self, lease):
        row = self._row(lease.command_id, lock=True)
        if (
            row is None
            or lease.owner_id != self.owner_id
            or row.execution_epoch != lease.epoch
            or row.execution_owner_id != lease.worker_id
            or row.status != "running"
            or row.lease_expires_at is None
            or row.lease_expires_at <= self.clock()
        ):
            raise RefundFlowError("refund_submission_conflict", True)
        return row

    def owned(self, lease):
        return self._record(self._owned_row(lease))

    def _fenced_update(self, lease, values):
        row = self._owned_row(lease)
        result = self.session.execute(
            update(ConversationRefundSubmission)
            .where(
                ConversationRefundSubmission.command_id == lease.command_id,
                ConversationRefundSubmission.owner_id == self.owner_id,
                ConversationRefundSubmission.execution_owner_id == lease.worker_id,
                ConversationRefundSubmission.execution_epoch == lease.epoch,
                ConversationRefundSubmission.status == "running",
                ConversationRefundSubmission.lease_expires_at > self.clock(),
            )
            .values(**values, updated_at=self.clock())
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise RefundFlowError("refund_submission_conflict", True)
        self.session.expire(row)
        return row

    def heartbeat(self, lease, seconds):
        row = self._fenced_update(
            lease, {"lease_expires_at": self.clock() + timedelta(seconds=seconds)}
        )
        return row.lease_expires_at

    def audit(self, lease, audit, result=None):
        row = self._owned_row(lease)
        audit = checked(RefundRevalidationAudit, audit.model_dump(mode="json"))
        record = self._record(row)
        previous = [
            a for a in record.revalidation_audits if a.attempt_id == audit.attempt_id
        ]
        if audit.outcome == "started":
            if previous or result is not None:
                raise RefundFlowError("refund_revalidation_conflict")
        elif len(previous) != 1 or previous[0].outcome != "started":
            raise RefundFlowError("refund_revalidation_conflict")
        additions = []
        if result is not None:
            result = checked(RefundRevalidation, result.model_dump(mode="json"))
            if (
                audit.outcome != "succeeded"
                or result.revalidation_hash != audit.result_hash
            ):
                raise RefundFlowError("refund_revalidation_conflict")
            additions.append(result.model_dump(mode="json"))
        elif audit.result_hash is not None or audit.outcome == "succeeded":
            raise RefundFlowError("refund_revalidation_conflict")
        value = audit.model_dump(mode="json")
        row = self._fenced_update(
            lease,
            {
                "revalidations": [
                    *row.revalidations,
                    *additions,
                    {"audit": value, "hash": canonical_hash(value)},
                ]
            },
        )
        return self._record(row)

    def finish_error(self, lease, code, retryable, *, reassess=False):
        row = self._fenced_update(
            lease,
            {
                "status": "interrupted" if retryable else "failed",
                "error_code": code,
                "retryable": retryable,
                "execution_owner_id": None,
                "lease_expires_at": None,
            },
        )
        if reassess:
            self.session.execute(
                update(ConversationRefundOffer)
                .where(
                    ConversationRefundOffer.offer_id == row.offer_id,
                    ConversationRefundOffer.owner_id == self.owner_id,
                    ConversationRefundOffer.status != "submitted",
                )
                .values(status="needs_reassessment", invalid_reason=code)
            )
        self.session.flush()
        return self._record(row)

    def release(self, lease):
        # Cleanup cannot release or overwrite a newer worker's state.
        self.session.execute(
            update(ConversationRefundSubmission)
            .where(
                ConversationRefundSubmission.command_id == lease.command_id,
                ConversationRefundSubmission.owner_id == self.owner_id,
                ConversationRefundSubmission.execution_epoch == lease.epoch,
                ConversationRefundSubmission.execution_owner_id == lease.worker_id,
                ConversationRefundSubmission.status == "running",
            )
            .values(
                status="interrupted",
                retryable=True,
                execution_owner_id=None,
                lease_expires_at=None,
            )
            .execution_options(synchronize_session=False)
        )

    def publish(self, lease, receipt):
        row = self._owned_row(lease)
        receipt = checked(RefundReceipt, receipt.model_dump(mode="json"))
        result = self.session.execute(
            update(ConversationRefundSubmission)
            .where(
                ConversationRefundSubmission.command_id == lease.command_id,
                ConversationRefundSubmission.owner_id == self.owner_id,
                ConversationRefundSubmission.execution_owner_id == lease.worker_id,
                ConversationRefundSubmission.execution_epoch == lease.epoch,
                ConversationRefundSubmission.status == "running",
                ConversationRefundSubmission.receipt.is_(None),
                ConversationRefundSubmission.lease_expires_at > self.clock(),
            )
            .values(
                status="completed",
                receipt=receipt.model_dump(mode="json"),
                receipt_hash=receipt.fingerprint,
                execution_owner_id=None,
                lease_expires_at=None,
                error_code=None,
                retryable=False,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise RefundFlowError("refund_submission_conflict", True)
        self.session.execute(
            update(ConversationRefundOffer)
            .where(
                ConversationRefundOffer.offer_id == row.offer_id,
                ConversationRefundOffer.owner_id == self.owner_id,
            )
            .values(status="submitted")
        )
        self.session.expire(row)
        return self._record(row)
