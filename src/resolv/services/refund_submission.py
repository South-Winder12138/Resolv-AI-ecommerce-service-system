"""Standalone, fenced commands. Only the SQLite receipt proves a local application."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from uuid import uuid4

from sqlalchemy import text

from resolv.db.checkpoints import CheckpointError, CheckpointSession
from resolv.db.conversation_base import utc_now
from resolv.models.conversation import Conversation, ConversationRun
from resolv.repositories.refund_application import RefundApplicationRepository
from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.repositories.refund_submission import RefundSubmissionRepository
from resolv.schemas.knowledge_branch import FrozenJSON
from resolv.schemas.refund import RefundAccessContext, RefundFlowError
from resolv.schemas.refund_submission import (
    RefundRegistrationCommand,
    RefundRevalidationAudit,
    RefundSubmissionView,
    RefundSubmitInput,
)
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.refund_registration import (
    RefundRegistrationService,
    begin_refund_write,
)
from resolv.services.refund_reply_projection import public_refund_offer
from resolv.services.refund_revalidation import (
    check_refund_commit_time,
    revalidate_refund_offer,
)


@contextmanager
def mysql_submission_lock(engine, command_id):
    """A non-reconnecting dedicated physical connection, with no row transaction."""
    if engine.dialect.name != "mysql" or len("refund-submit:" + command_id) > 64:
        raise RefundFlowError("refund_store_failed", True)
    raw = engine.raw_connection()
    connection = raw.driver_connection
    previous_autocommit = connection.get_autocommit()
    owned = CheckpointSession(connection, "refund-submit:" + command_id)
    acquired = False
    try:
        connection.autocommit(True)
        with connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, 0)", (owned._lock_name,))
            result = cursor.fetchone()[0]
        if result == 0:
            raise RefundFlowError("refund_submission_busy", True)
        if result != 1:
            raise RefundFlowError("refund_store_failed", True)
        acquired = True
        owned.assert_owned()
        yield owned
    except CheckpointError:
        raise RefundFlowError("refund_submission_conflict", True) from None
    finally:
        try:
            with owned._lock:
                owned._active = False
                if acquired and not owned._lost:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT RELEASE_LOCK(%s)", (owned._lock_name,))
                        if cursor.fetchone()[0] != 1:
                            raw.invalidate()
                if not owned._lost:
                    connection.autocommit(previous_autocommit)
                else:
                    raw.invalidate()
        except Exception:  # noqa: BLE001 — uncertain lock cleanup invalidates the connection
            raw.invalidate()
        finally:
            raw.close()


@dataclass
class SubmissionExecution:
    service: object
    lease: object
    lock: object
    deadline: object
    failed: Event = field(default_factory=Event)
    gate: RLock = field(default_factory=RLock)

    def assert_local(self):
        if self.failed.is_set() or self.service.clock() >= self.deadline:
            self.failed.set()
            raise RefundFlowError("refund_submission_conflict", True)

    def assert_owned(self):
        with self.gate:
            self.assert_local()
            try:
                self.lock.assert_owned()
                with self.service.session_factory.begin() as session:
                    record = self.service._repo(session).owned(self.lease)
                self.deadline = record.lease_expires_at
            except Exception:  # noqa: BLE001 — any uncertain ownership fences the worker
                self.failed.set()
                raise RefundFlowError("refund_submission_conflict", True) from None

    def heartbeat(self):
        with self.gate:
            self.assert_owned()
            with self.service.session_factory.begin() as session:
                self.deadline = self.service._repo(session).heartbeat(self.lease, 120)


class RefundSubmissionService:
    def __init__(
        self,
        session_factory,
        business_factory,
        *,
        owner_id,
        identity,
        facts,
        selection_reader,
        profile,
        clock=utc_now,
        fault_hook=None,
        execution_lock_factory=None,
    ):
        self.session_factory, self.business_factory = session_factory, business_factory
        self.owner_id, self.identity, self.facts = owner_id, identity, facts
        self.selection_reader, self.profile, self.clock = (
            selection_reader,
            profile,
            clock,
        )
        self.fault_hook, self.execution_lock_factory = (
            fault_hook,
            execution_lock_factory,
        )

    def _repo(self, session):
        return RefundSubmissionRepository(session, self.owner_id, clock=self.clock)

    def _fault(self, point):
        if self.fault_hook:
            self.fault_hook(point)

    def _offer(self, conversation_id, offer_id):
        with self.session_factory() as session:
            offer = RefundFlowRepository(session, self.owner_id).get_offer(offer_id)
            parent = session.get(ConversationRun, offer.run_id) if offer else None
        if (
            offer is None
            or offer.conversation_id != conversation_id
            or parent is None
            or parent.status != "completed"
        ):
            raise RefundFlowError("refund_not_found")
        return offer

    def _command(self, conversation_id, command_id):
        with self.session_factory() as session:
            record = self._repo(session).get(command_id)
        if record is None or record.conversation_id != conversation_id:
            raise RefundFlowError("refund_not_found")
        return record, self._offer(conversation_id, record.offer_id)

    def _access(self, offer, context):
        access = self.identity.resolve(context)
        access = RefundAccessContext.model_validate_json(access.model_dump_json())
        if (
            access.subject_ref != offer.subject_ref
            or access.namespace != offer.binding.order_source
        ):
            raise RefundFlowError("refund_not_found")
        return access

    def _usable_conversation(self, conversation_id):
        with self.session_factory() as s:
            row = s.get(Conversation, conversation_id)
            if row is None or row.owner_id != self.owner_id:
                raise RefundFlowError("refund_not_found")
            if row.status != "active":
                raise RefundFlowError("refund_conversation_archived")

    def lookup(self, binding):
        """Short read facade used by fresh external status reads."""
        with self.business_factory() as s:
            return RefundRegistrationService(s).lookup(binding)

    def _registration_command(self, record, offer, revalidation, created_at):
        return RefundRegistrationCommand(
            command_id=record.submission_id,
            offer_id=offer.offer_id,
            request_hash=record.request_hash,
            binding=offer.binding,
            subject_ref=offer.subject_ref,
            money=offer.money,
            reason=offer.reason,
            decision_hash=offer.decision_hash,
            evidence_hash=offer.evidence_hash,
            revalidation_hash=revalidation.revalidation_hash,
            snapshots=FrozenJSON.from_value(
                {
                    "critical_facts": offer.critical_facts.value,
                    "revalidation": revalidation.model_dump(mode="json"),
                }
            ),
            created_at=created_at,
        )

    def _business_receipt(self, record, offer):
        with self.business_factory() as s:
            # SQLite's driver does not begin on SELECT by default. Pin these
            # receipt/application reads to one snapshot, including WAL mode.
            s.execute(text("BEGIN"))
            repo = RefundApplicationRepository(s)
            row = repo.receipt(record.submission_id)
            if row is None:
                application = repo.get(
                    offer.binding.order_source, offer.binding.order_id
                )
                if (
                    (
                        application is not None
                        and application.command_id == record.submission_id
                    )
                    or record.receipt is not None
                    or record.status == "completed"
                    or offer.status == "submitted"
                ):
                    raise RefundFlowError("refund_store_inconsistent")
                return None
            result = repo.validated_receipt(row)
            request_hash, command_hash, created_at = (
                row.request_hash,
                row.command_hash,
                row.created_at,
            )
            RefundRegistrationService(s)._consistent(
                result.order_source, result.order_id
            )
        # A worker can commit between the first command read and the receipt
        # read. Its successful recheck was saved before that SQLite commit.
        # Close SQLite first, then read the current command once; GET still
        # neither claims execution nor performs revalidation.
        with self.session_factory() as s:
            latest = self._repo(s).get(record.submission_id)
        if latest is None:
            raise RefundFlowError("refund_not_found")
        expected = (
            self._registration_command(
                latest, offer, latest.revalidations[-1], created_at
            )
            if latest.revalidations
            else None
        )
        if (
            latest.offer_id != record.offer_id
            or latest.request_hash != record.request_hash
            or expected is None
            or request_hash != latest.request_hash
            or command_hash != expected.fingerprint
            or (result.order_source, result.order_id)
            != (offer.binding.order_source, offer.binding.order_id)
            or (record.receipt is not None and record.receipt != result)
            or (latest.receipt is not None and latest.receipt != result)
        ):
            raise RefundFlowError(
                "refund_store_inconsistent", submission_id=record.submission_id
            )
        return result

    def business_receipt(self, command_id):
        with self.session_factory() as s:
            record = self._repo(s).get(command_id)
        if record is None:
            raise RefundFlowError("refund_not_found")
        offer = self._offer(record.conversation_id, record.offer_id)
        return self._business_receipt(record, offer)

    @staticmethod
    def _view(record, receipt=None):
        return RefundSubmissionView(
            submission_id=record.submission_id,
            status=record.status,
            receipt=receipt or record.receipt,
            error_code=record.error_code,
            retryable=record.retryable,
        )

    def get_submission(self, conversation_id, submission_id, server_context):
        record, offer = self._command(conversation_id, submission_id)
        self._access(offer, server_context)
        return self._view(record, self._business_receipt(record, offer))

    def get_offer(self, conversation_id, offer_id, server_context):
        offer = self._offer(conversation_id, offer_id)
        self._access(offer, server_context)
        with self.session_factory() as s:
            record = self._repo(s).get_by_offer(offer_id)
        receipt = self._business_receipt(record, offer) if record else None
        return public_refund_offer(offer, record, receipt, now=self.clock())

    def accept_submission(
        self, conversation_id, offer_id, request: RefundSubmitInput, server_context
    ):
        request = RefundSubmitInput.model_validate_json(request.model_dump_json())
        offer = self._offer(conversation_id, offer_id)
        self._access(offer, server_context)
        with self.session_factory() as s:
            existing = self._repo(s).get_by_offer(offer_id)
        if existing is None:
            self._usable_conversation(conversation_id)
        request_hash = canonical_hash(
            {
                "owner_id": self.owner_id,
                "conversation_id": conversation_id,
                "offer_id": offer_id,
                "confirm": True,
            }
        )
        with self.session_factory.begin() as s:
            record = self._repo(s).accept(offer_id, request.request_id, request_hash)
        return self._view(record, self._business_receipt(record, offer))

    def submit(self, conversation_id, offer_id, request, server_context):
        accepted = self.accept_submission(
            conversation_id, offer_id, request, server_context
        )
        if accepted.status == "completed":
            return accepted
        return self.execute(accepted.submission_id, server_context)

    def resume(self, conversation_id, submission_id, server_context):
        self._command(conversation_id, submission_id)
        return self.execute(submission_id, server_context)

    @contextmanager
    def submission_execution(self, command_id):
        with self.session_factory() as s:
            engine = s.get_bind()
        factory = self.execution_lock_factory or (
            lambda cid: mysql_submission_lock(engine, cid)
        )
        with factory(command_id) as lock:
            with self.session_factory.begin() as s:
                lease = self._repo(s).claim(command_id, uuid4().hex, self.clock(), 120)
                record = self._repo(s).owned(lease)
            execution = SubmissionExecution(self, lease, lock, record.lease_expires_at)
            stopped = Event()

            def heartbeat():
                while not stopped.wait(30):
                    try:
                        execution.heartbeat()
                    except Exception:  # noqa: BLE001 — failed heartbeat permanently fences
                        execution.failed.set()
                        return

            worker = Thread(
                target=heartbeat, daemon=True, name="refund-submission-heartbeat"
            )
            worker.start()
            try:
                yield execution
            finally:
                stopped.set()
                worker.join()
                try:
                    with self.session_factory.begin() as s:
                        self._repo(s).release(lease)
                finally:
                    execution.failed.set()

    def _publish(self, execution, receipt):
        execution.assert_owned()
        self._fault("refund_before_submission_publish")
        with self.session_factory.begin() as s:
            record = self._repo(s).publish(execution.lease, receipt)
            self._fault("refund_before_submission_publish_commit")
        self._fault("refund_after_submission_publish")
        return self._view(record, receipt)

    def _attempt(self, execution, record, offer, access):
        # A crashed attempt's last started event remains, followed by an explicit
        # unknown closure. Every new execution performs its own fresh reads.
        with self.session_factory.begin() as s:
            repo = self._repo(s)
            saved = repo.owned(execution.lease)
            for audit in saved.revalidation_audits:
                if audit.outcome == "started" and not any(
                    a.attempt_id == audit.attempt_id and a.outcome != "started"
                    for a in saved.revalidation_audits
                ):
                    repo.audit(
                        execution.lease,
                        RefundRevalidationAudit(
                            attempt_id=audit.attempt_id,
                            outcome="unknown",
                            at=self.clock(),
                            error_code="refund_revalidation_unknown",
                        ),
                    )
            attempt = uuid4().hex
            repo.audit(
                execution.lease,
                RefundRevalidationAudit(
                    attempt_id=attempt, outcome="started", at=self.clock()
                ),
            )
        try:
            self._fault("refund_before_revalidation")
            self._usable_conversation(offer.conversation_id)
            result = revalidate_refund_offer(
                offer,
                access=access,
                adapter=self.facts,
                registration=self,
                current_selection=self.selection_reader(),
                now=self.clock(),
                profile=self.profile,
            )
            self._fault("refund_after_revalidation")
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, RefundFlowError)
                else "refund_provider_failed"
            )
            with self.session_factory.begin() as s:
                self._repo(s).audit(
                    execution.lease,
                    RefundRevalidationAudit(
                        attempt_id=attempt,
                        outcome="failed",
                        at=self.clock(),
                        error_code=code,
                    ),
                )
            raise
        with self.session_factory.begin() as s:
            self._repo(s).audit(
                execution.lease,
                RefundRevalidationAudit(
                    attempt_id=attempt,
                    outcome="succeeded",
                    at=self.clock(),
                    result_hash=result.revalidation_hash,
                ),
                result,
            )
        self._fault("refund_after_revalidation_save")
        return result

    def execute(self, command_id, server_context):
        with self.session_factory() as s:
            initial = self._repo(s).get(command_id)
        if initial is None:
            raise RefundFlowError("refund_not_found")
        record, offer = self._command(initial.conversation_id, command_id)
        access = self._access(offer, server_context)
        receipt = self._business_receipt(record, offer)
        if record.status == "completed" or (
            record.status == "failed" and not record.retryable and receipt is None
        ):
            return self._view(record, receipt)
        try:
            with self.submission_execution(command_id) as execution:
                try:
                    record, offer = self._command(record.conversation_id, command_id)
                    receipt = self._business_receipt(record, offer)
                    if receipt is None:
                        result = self._attempt(execution, record, offer, access)
                        existing = result.differences == ("existing_application",)
                        if result.outcome != "passed" and not existing:
                            code = (
                                "refund_offer_expired"
                                if "offer_expired" in result.differences
                                else "refund_unavailable"
                                if result.outcome == "unavailable"
                                else "refund_needs_reassessment"
                            )
                            raise RefundFlowError(code)
                        self._fault("refund_before_business_begin")
                        self._usable_conversation(offer.conversation_id)
                        execution.assert_owned()
                        # Do not query MySQL or a provider while this local write
                        # transaction is held. The heartbeat shares this gate.
                        with execution.gate:
                            execution.assert_local()
                            with self.business_factory() as s:
                                begin_refund_write(s)
                                registration = RefundRegistrationService(s)
                                if (
                                    existing
                                    and registration.lookup(offer.binding) is None
                                ):
                                    raise RefundFlowError("refund_store_inconsistent")
                                command = self._registration_command(
                                    record, offer, result, self.clock()
                                )
                                registered = registration.register(command)
                                self._fault("refund_before_business_commit")
                                execution.assert_local()
                                check_refund_commit_time(
                                    offer,
                                    result,
                                    now=self.clock(),
                                    creating_application=registered.created,
                                )
                                s.commit()
                        self._fault("refund_after_business_commit")
                        record, offer = self._command(
                            record.conversation_id, command_id
                        )
                        receipt = self._business_receipt(record, offer)
                        if receipt is None:
                            raise RefundFlowError("refund_store_inconsistent")
                    return self._publish(execution, receipt)
                except Exception as exc:  # noqa: BLE001 — reconcile uncertain writes before public error
                    # A write acknowledgement may be lost: always resolve the
                    # original command before deciding this attempt failed.
                    record, offer = self._command(record.conversation_id, command_id)
                    receipt = self._business_receipt(record, offer)
                    if record.status == "completed":
                        return self._view(record, receipt)
                    error = (
                        exc
                        if isinstance(exc, RefundFlowError)
                        else RefundFlowError("refund_store_failed", True)
                    )
                    if (
                        receipt is not None
                        and error.code != "refund_store_inconsistent"
                    ):
                        error = RefundFlowError(
                            "refund_submission_publish_failed", True
                        )
                    try:
                        with self.session_factory.begin() as s:
                            record = self._repo(s).finish_error(
                                execution.lease,
                                error.code,
                                error.retryable,
                                reassess=error.code == "refund_needs_reassessment",
                            )
                    except RefundFlowError:
                        record, _ = self._command(record.conversation_id, command_id)
                    if error.code == "refund_store_inconsistent":
                        raise error
                    return self._view(record, receipt)
        except RefundFlowError as exc:
            if exc.code != "refund_submission_busy":
                raise
            record, offer = self._command(record.conversation_id, command_id)
            return self._view(record, self._business_receipt(record, offer))
