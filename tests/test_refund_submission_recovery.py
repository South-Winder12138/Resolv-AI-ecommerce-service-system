from datetime import timedelta

import pytest
from sqlalchemy import select

from resolv.models.refund_application import RefundApplication, RefundSubmissionReceipt
from resolv.models.refund_flow import ConversationRefundSubmission
from resolv.repositories.refund_submission import RefundSubmissionRepository
from resolv.schemas.refund import RefundFlowError
from tests.refund_submission_fixtures import flow_harness, submission_harness
from tests.test_refund_submission import counts, submit

__all__ = ["flow_harness", "submission_harness"]


def crash_at(h, point):
    def hook(actual):
        if actual == point:
            raise OSError("injected uncertain boundary")

    h.submitter.fault_hook = hook


def test_receipt_survives_expiry_disable_and_publish_failure(submission_harness):
    from resolv.schemas.knowledge_release import ReleaseSelection

    h = submission_harness
    crash_at(h, "refund_after_business_commit")
    first = submit(h)
    assert counts(h) == (1, 1)
    assert first.receipt is not None
    h.clock.now = h.offer.expires_at + timedelta(days=1)
    h.submitter.selection_reader = lambda: ReleaseSelection(status="disabled")
    h.submitter.fault_hook = None
    replay = h.submitter.resume(h.offer.conversation_id, first.submission_id, {})
    assert replay.receipt.application_id == first.receipt.application_id
    assert replay.status == "completed" and counts(h) == (1, 1)
    assert submit(h).receipt == replay.receipt


def test_resume_without_receipt_revalidates_again(submission_harness):
    h = submission_harness
    crash_at(h, "refund_before_business_begin")
    first = submit(h)
    assert first.receipt is None and first.retryable and counts(h) == (0, 0)
    before = h.counts["query_refund_order"]
    h.submitter.fault_hook = None
    h.facts.fail = TimeoutError("new permission query unavailable")
    replay = h.submitter.resume(h.offer.conversation_id, first.submission_id, {})
    assert replay.receipt is None and replay.retryable and counts(h) == (0, 0)
    assert h.counts["query_refund_order"] > before
    with h.factory() as s:
        record = RefundSubmissionRepository(s, "local").get(first.submission_id)
        assert len(record.revalidations) == 1
        assert [a.outcome for a in record.revalidation_audits] == [
            "started",
            "succeeded",
            "started",
            "failed",
        ]


def test_stale_submission_worker_cannot_publish(submission_harness):
    h = submission_harness

    def steal(point):
        if point == "refund_before_business_begin":
            with h.factory.begin() as s:
                row = s.scalar(select(ConversationRefundSubmission))
                row.execution_epoch += 1
                row.execution_owner_id = "new-worker"

    h.submitter.fault_hook = steal
    result = submit(h)
    assert result.receipt is None and counts(h) == (0, 0)
    with h.factory() as s:
        row = s.scalar(select(ConversationRefundSubmission))
        assert row.execution_owner_id == "new-worker" and row.status == "running"


@pytest.mark.parametrize(
    "fault",
    [
        "refund_before_business_commit",
        "refund_after_business_commit",
        "refund_before_submission_publish",
        "refund_after_submission_publish",
    ],
)
def test_commit_boundaries_recover_one_original_command(submission_harness, fault):
    h = submission_harness
    crash_at(h, fault)
    first = submit(h)
    h.submitter.fault_hook = None
    replay = h.submitter.resume(h.offer.conversation_id, first.submission_id, {})
    assert replay.receipt and replay.status == "completed" and counts(h) == (1, 1)
    if first.receipt:
        assert first.receipt == replay.receipt


@pytest.mark.parametrize("tamper", ["missing", "hash", "binding", "command_hash"])
def test_bad_or_lost_receipt_is_not_recreated(submission_harness, tamper):
    h = submission_harness
    first = submit(h)
    with h.business.begin() as s:
        row = s.get(RefundSubmissionReceipt, first.submission_id)
        if tamper == "missing":
            s.delete(row)
        elif tamper == "hash":
            row.result_hash = "0" * 64
        elif tamper == "command_hash":
            row.command_hash = "0" * 64
        else:
            s.get(RefundApplication, first.receipt.application_id).order_id = "OTHER"
    with pytest.raises(RefundFlowError, match="refund_store_inconsistent"):
        h.submitter.resume(h.offer.conversation_id, first.submission_id, {})
    assert counts(h)[0] == 1


def test_lost_unpublished_receipt_cannot_turn_created_into_existing(submission_harness):
    h = submission_harness
    crash_at(h, "refund_after_business_commit")
    first = submit(h)
    assert first.receipt and first.receipt.created
    with h.business.begin() as s:
        s.delete(s.get(RefundSubmissionReceipt, first.submission_id))
    h.submitter.fault_hook = None
    with pytest.raises(RefundFlowError, match="refund_store_inconsistent"):
        h.submitter.resume(h.offer.conversation_id, first.submission_id, {})
    assert counts(h) == (1, 0)


def test_resume_after_passed_recheck_reads_revoked_permission(submission_harness):
    h = submission_harness
    crash_at(h, "refund_before_business_begin")
    first = submit(h)
    h.submitter.fault_hook = None
    calls = []

    def revoked(*args):
        calls.append(True)
        raise RefundFlowError("refund_authorization_conflict")

    h.facts.authorize = revoked
    resumed = h.submitter.resume(h.offer.conversation_id, first.submission_id, {})
    assert calls and resumed.receipt is None and counts(h) == (0, 0)


def test_local_lease_expiry_before_commit_rolls_back_without_mysql_in_write(
    submission_harness,
):
    h = submission_harness

    def expire(point):
        if point == "refund_before_business_commit":
            h.clock.now += timedelta(seconds=121)

    h.submitter.fault_hook = expire
    first = submit(h)
    assert first.receipt is None and counts(h) == (0, 0)


def test_crashed_revalidation_attempt_is_closed_unknown_on_explicit_resume(
    submission_harness,
):
    h = submission_harness

    class ProcessDeath(BaseException):
        pass

    def die(point):
        if point == "refund_after_revalidation":
            raise ProcessDeath()

    h.submitter.fault_hook = die
    with pytest.raises(ProcessDeath):
        submit(h)
    with h.factory() as s:
        record = s.scalar(select(ConversationRefundSubmission))
        cid = record.command_id
    h.submitter.fault_hook = None
    resumed = h.submitter.resume(h.offer.conversation_id, cid, {})
    assert resumed.receipt and counts(h) == (1, 1)
    with h.factory() as s:
        saved = RefundSubmissionRepository(s, "local").get(cid)
        assert [a.outcome for a in saved.revalidation_audits] == [
            "started",
            "unknown",
            "started",
            "succeeded",
        ]


def test_post_commit_failure_never_becomes_a_nonresumable_denial(submission_harness):
    h = submission_harness
    def failure(point):
        if point == "refund_before_submission_publish":
            raise RefundFlowError("refund_needs_reassessment")
    h.submitter.fault_hook = failure
    first = submit(h)
    assert first.receipt is not None and first.retryable
    h.submitter.fault_hook = None
    replay = h.submitter.resume(h.offer.conversation_id, first.submission_id, {})
    assert replay.status == "completed" and replay.receipt == first.receipt
