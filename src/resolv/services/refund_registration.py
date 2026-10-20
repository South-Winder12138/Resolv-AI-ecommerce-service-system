"""One registration boundary shared by old tools and new submission commands."""

from datetime import UTC
from uuid import uuid4

from sqlalchemy import select, text

from resolv.models.order import Order
from resolv.models.refund import Refund
from resolv.models.refund_application import RefundApplication
from resolv.repositories.refund import RefundRepository
from resolv.repositories.refund_application import RefundApplicationRepository
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_submission import RefundReceipt, RefundRegistrationCommand

LOCAL_SOURCE = "local-business-v1"


def begin_refund_write(session):
    if session.get_bind().dialect.name != "sqlite":
        raise RefundFlowError("refund_store_failed", True)
    # A caller may already hold BEGIN IMMEDIATE (old approval execution).
    driver = session.connection().connection.driver_connection
    if not driver.in_transaction:
        session.execute(text("BEGIN IMMEDIATE"))


class RefundRegistrationService:
    def __init__(self, session):
        self.session = session
        self.repo = RefundApplicationRepository(session)

    def _consistent(self, source, order_id):
        app = self.repo.get(source, order_id)
        if source == LOCAL_SOURCE:
            order = self.session.scalar(select(Order).where(Order.order_no == order_id))
            old = self.session.scalar(select(Refund).where(Refund.order_id == order.id)) if order else None
            if bool(app) != bool(old) or (app and (
                app.legacy_refund_id != old.id or app.status != old.status or app.reason != old.reason
                or app.created_at != old.created_at.replace(tzinfo=UTC)
            )):
                raise RefundFlowError("refund_store_inconsistent")
        elif app is not None and app.legacy_refund_id is not None:
            raise RefundFlowError("refund_store_inconsistent")
        return app

    def lookup(self, binding):
        app = self._consistent(binding.order_source, binding.order_id)
        return self.repo.record(app) if app else None

    def _application(self, *, source, order_id, reason, created_at, command=None, legacy=None):
        row = RefundApplication(
            application_id=uuid4().hex if command else "legacy-" + str(legacy.id),
            order_source=source, order_id=order_id, reason=reason,
            created_at=created_at, status="pending" if command else legacy.status,
            origin="offer" if command else "legacy", legacy_refund_id=legacy.id if legacy else None,
            subject_ref=command.subject_ref if command else None,
            money=command.money.model_dump(mode="json") if command else None,
            offer_id=command.offer_id if command else None, command_id=command.command_id if command else None,
            decision_hash=command.decision_hash if command else None, evidence_hash=command.evidence_hash if command else None,
            revalidation_hash=command.revalidation_hash if command else None,
            snapshots=command.snapshots.value if command else None,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def register(self, command: RefundRegistrationCommand) -> RefundReceipt:
        command = RefundRegistrationCommand.model_validate_json(command.model_dump_json())
        begin_refund_write(self.session)
        saved = self.repo.receipt(command.command_id)
        if saved:
            if saved.command_hash != command.fingerprint or saved.request_hash != command.request_hash:
                raise RefundFlowError("refund_request_conflict")
            self._consistent(command.binding.order_source, command.binding.order_id)
            return self.repo.validated_receipt(saved)
        app = self._consistent(command.binding.order_source, command.binding.order_id)
        created = app is None
        if app is None:
            old = None
            if command.binding.order_source == LOCAL_SOURCE:
                order = self.session.scalar(select(Order).where(Order.order_no == command.binding.order_id))
                if order is None:
                    raise RefundFlowError("refund_not_found")
                old = Refund(order_id=order.id, status="pending", reason=command.reason,
                             created_at=command.created_at.replace(tzinfo=None), updated_at=command.created_at.replace(tzinfo=None))
                RefundRepository(self.session).add(old)
                self.session.flush()
            app = self._application(source=command.binding.order_source, order_id=command.binding.order_id,
                                    reason=command.reason, created_at=command.created_at, command=command, legacy=old)
        record = self.repo.record(app)
        result = RefundReceipt(application_id=app.application_id, order_source=app.order_source, order_id=app.order_id,
                               status=app.status, created=created, money=record.money.public() if record.money else None,
                               reason=app.reason, created_at=app.created_at)
        self.repo.add_receipt(command, app, result)
        return result

    def register_legacy(self, order, reason):
        begin_refund_write(self.session)
        existing = self._consistent(LOCAL_SOURCE, order.order_no)
        if existing:
            return {"success": True, "order_id": order.order_no, "status": existing.status, "changed": False}
        if order.status not in {"shipped", "delivered"}:
            return {"success": False, "order_id": order.order_no, "reason": "refund_not_available"}
        legacy = Refund(order_id=order.id, status="pending", reason=reason)
        RefundRepository(self.session).add(legacy)
        self.session.flush()
        self._application(source=LOCAL_SOURCE, order_id=order.order_no, reason=reason,
                          created_at=legacy.created_at.replace(tzinfo=UTC), legacy=legacy)
        return {"success": True, "order_id": order.order_no, "status": "pending", "changed": True}

    def backfill_legacy(self):
        """Explicit migration only; never called from reads or registration."""
        for old, order in self.session.execute(select(Refund, Order).join(Order, Order.id == Refund.order_id)):
            app = self.repo.get(LOCAL_SOURCE, order.order_no)
            if app is not None:
                self._consistent(LOCAL_SOURCE, order.order_no)
            else:
                self._application(source=LOCAL_SOURCE, order_id=order.order_no, reason=old.reason,
                                  created_at=old.created_at.replace(tzinfo=UTC), legacy=old)
