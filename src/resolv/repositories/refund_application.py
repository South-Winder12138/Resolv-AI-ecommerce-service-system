"""Canonical refund rows; caller owns locking and transaction lifetime."""

import json

from sqlalchemy import select

from resolv.models.refund_application import RefundApplication, RefundSubmissionReceipt
from resolv.schemas.refund import RefundApplicationRecord, RefundMoney
from resolv.services.hybrid_fingerprints import canonical_hash


class RefundApplicationRepository:
    def __init__(self, session):
        self.session = session

    def get(self, source, order_id):
        return self.session.scalar(select(RefundApplication).where(
            RefundApplication.order_source == source, RefundApplication.order_id == order_id))

    def receipt(self, command_id):
        return self.session.get(RefundSubmissionReceipt, command_id)

    def add_receipt(self, command, application, result):
        self.session.add(RefundSubmissionReceipt(
            command_id=command.command_id, application_id=application.application_id,
            request_hash=command.request_hash, command_hash=command.fingerprint,
            result=result.model_dump(mode="json"), result_hash=result.fingerprint,
            created_at=command.created_at,
        ))
        self.session.flush()

    def record(self, row):
        return RefundApplicationRecord(
            application_id=row.application_id, order_source=row.order_source, order_id=row.order_id,
            subject_ref=row.subject_ref, status=row.status, reason=row.reason,
            money=RefundMoney.model_validate_json(json.dumps(row.money)) if row.money is not None else None,
            created_at=row.created_at, legacy_refund_id=row.legacy_refund_id, origin=row.origin,
        )

    def validated_receipt(self, row):
        from resolv.schemas.refund import RefundFlowError
        from resolv.schemas.refund_submission import RefundReceipt

        if row.result_hash != canonical_hash(row.result):
            raise RefundFlowError("refund_store_inconsistent")
        result = RefundReceipt.model_validate_json(json.dumps(row.result))
        app = self.session.get(RefundApplication, row.application_id)
        if app is None or result.application_id != app.application_id or (
            result.order_source, result.order_id, result.reason, result.money, result.created_at
        ) != (app.order_source, app.order_id, app.reason,
              self.record(app).money.public() if app.money else None, app.created_at):
            raise RefundFlowError("refund_store_inconsistent")
        return result
