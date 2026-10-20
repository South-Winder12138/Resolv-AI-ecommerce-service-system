"""Isolated command execution locks for SQLite-only offline tests."""

from contextlib import contextmanager
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4

import pytest

from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_submission import RefundSubmitInput
from tests.refund_fixtures import NOW, observation
from tests.refund_flow_fixtures import drive, flow_harness

__all__ = ["flow_harness"]


class LocalLocks:
    def __init__(self):
        self.mutex = Lock()
        self.locks = {}

    @contextmanager
    def __call__(self, command_id):
        with self.mutex:
            lock = self.locks.setdefault(command_id, Lock())
        if not lock.acquire(blocking=False):
            raise RefundFlowError("refund_submission_busy", True)
        try:
            yield SimpleNamespace(assert_owned=lambda: None)
        finally:
            lock.release()


@pytest.fixture
def submission_harness(flow_harness):
    from resolv.services.refund_offer import finalize_refund_reply
    from resolv.services.refund_submission import RefundSubmissionService

    h = flow_harness
    h.service.refund_clock = lambda: NOW
    record = drive(h)
    finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
    h.service.mark_completed(h.run.run_id, h.execution.epoch)
    with h.factory() as s:
        offer = RefundFlowRepository(s, "local").offer_for_run(h.run.run_id)
    clock = SimpleNamespace(now=NOW)
    locks = LocalLocks()
    access = h.flow.identity.resolve({})
    # The flow fixture checks global pool emptiness for its serial stage tests.
    # Concurrent command tests may legitimately have another thread in SQL.
    identity = SimpleNamespace(resolve=lambda _: access)

    def read(name, binding, call_id, fields):
        h.counts[name] += 1
        if h.facts.fail:
            raise h.facts.fail
        return observation(name, bound=binding, call_id=call_id, fields=fields)

    h.facts.read = read
    h.facts.authorize = lambda _, binding: observation(
        bound=binding, call_id=uuid4().hex, fields={"authorized": True}
    )
    submitter = RefundSubmissionService(
        h.factory,
        h.business,
        owner_id="local",
        identity=identity,
        facts=h.facts,
        selection_reader=lambda: h.profile.selection,
        profile=h.profile,
        clock=lambda: clock.now,
        execution_lock_factory=locks,
    )
    yield SimpleNamespace(
        **vars(h),
        offer=offer,
        submitter=submitter,
        clock=clock,
        locks=locks,
        request=RefundSubmitInput(request_id="submit-1", confirm=True),
    )
