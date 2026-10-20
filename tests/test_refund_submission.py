from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

import pytest
from sqlalchemy import func, select

from resolv.models.conversation import Conversation, ConversationMessage
from resolv.models.refund_application import RefundApplication, RefundSubmissionReceipt
from resolv.repositories.refund_submission import RefundSubmissionRepository
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_submission import RefundSubmitInput
from tests.refund_submission_fixtures import flow_harness, submission_harness

__all__ = ["flow_harness", "submission_harness"]


def submit(h, request=None):
    return h.submitter.submit(
        h.offer.conversation_id, h.offer.offer_id, request or h.request, {}
    )


def counts(h):
    with h.business() as s:
        return tuple(
            s.scalar(select(func.count()).select_from(t))
            for t in (RefundApplication, RefundSubmissionReceipt)
        )


def test_click_creates_pending_and_aliases_replay_original(submission_harness):
    h = submission_harness
    assert counts(h) == (0, 0)
    first = submit(h)
    assert first.status == "completed" and first.receipt.status == "pending"
    assert first.receipt.created and first.receipt.application_id
    before = h.counts.copy()
    replay = submit(h)
    alias = submit(h, RefundSubmitInput(request_id="submit-2", confirm=True))
    assert replay == alias == first
    assert h.counts == before and counts(h) == (1, 1)
    with h.factory() as s:
        saved = RefundSubmissionRepository(s, "local").get(first.submission_id)
        assert len(saved.revalidations) == 1
        assert [a.outcome for a in saved.revalidation_audits] == [
            "started",
            "succeeded",
        ]


def test_submission_races_share_one_application(submission_harness):
    h = submission_harness
    first = submit(h)
    barrier = Barrier(4)

    def race(i):
        barrier.wait()
        return submit(h, RefundSubmitInput(request_id=f"race-{i}", confirm=True))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(race, range(4)))
    assert all(
        r.receipt.application_id == first.receipt.application_id for r in results
    )
    assert counts(h) == (1, 1)


@pytest.mark.parametrize(
    "change,code",
    [
        ("expired", "refund_offer_expired"),
        ("disabled", "refund_needs_reassessment"),
        ("archived", "refund_conversation_archived"),
        ("revoked", "refund_provider_failed"),
    ],
)
def test_no_receipt_invalid_offer_does_not_write(submission_harness, change, code):
    h = submission_harness
    if change == "expired":
        h.clock.now = h.offer.expires_at
    elif change == "disabled":
        from resolv.schemas.knowledge_release import ReleaseSelection

        h.submitter.selection_reader = lambda: ReleaseSelection(status="disabled")
    elif change == "archived":
        with h.factory.begin() as s:
            s.get(Conversation, h.offer.conversation_id).status = "archived"
    else:
        h.facts.authorize = lambda *a: (_ for _ in ()).throw(
            RefundFlowError("refund_authorization_conflict")
        )
    try:
        view = submit(h)
        assert view.error_code == code and view.receipt is None
    except RefundFlowError as exc:
        assert exc.code == code
    assert counts(h) == (0, 0)


def test_offer_expires_after_recheck_before_commit_rolls_back(submission_harness):
    h = submission_harness
    h.clock.now = h.offer.expires_at - timedelta(seconds=1)

    def cross_boundary(point):
        if point == "refund_before_business_commit":
            h.clock.now += timedelta(seconds=2)

    h.submitter.fault_hook = cross_boundary
    result = submit(h)
    assert result.error_code == "refund_offer_expired"
    assert result.receipt is None and counts(h) == (0, 0)


def test_get_is_readonly_owned_and_does_not_revalidate(submission_harness):
    h = submission_harness
    first = submit(h)
    before = h.counts.copy()
    h.submitter.selection_reader = lambda: (_ for _ in ()).throw(
        AssertionError("GET selection")
    )
    h.submitter.execution_lock_factory = lambda _: (_ for _ in ()).throw(
        AssertionError("GET lock")
    )
    assert (
        h.submitter.get_submission(
            h.offer.conversation_id, first.submission_id, {}
        ).receipt
        == first.receipt
    )
    assert (
        h.submitter.get_offer(h.offer.conversation_id, h.offer.offer_id, {}).validity
        == "submitted"
    )
    assert h.counts == before
    with pytest.raises(RefundFlowError, match="refund_not_found"):
        h.submitter.get_submission("other-conversation", first.submission_id, {})
    h.submitter.owner_id = "other-owner"
    with pytest.raises(RefundFlowError, match="refund_not_found"):
        h.submitter.get_offer(h.offer.conversation_id, h.offer.offer_id, {})


def test_submission_does_not_change_later_chat_or_messages(submission_harness):
    h = submission_harness
    later, _ = h.service.accept(h.offer.conversation_id, "later", "继续聊天", "chat")
    with h.factory() as s:
        before = [
            (m.message_id, m.text) for m in s.scalars(select(ConversationMessage))
        ]
        assert (
            s.get(Conversation, h.offer.conversation_id).current_run_id == later.run_id
        )
    assert submit(h).receipt is not None
    with h.factory() as s:
        assert (
            s.get(Conversation, h.offer.conversation_id).current_run_id == later.run_id
        )
        assert [
            (m.message_id, m.text) for m in s.scalars(select(ConversationMessage))
        ] == before


def test_same_offer_concurrent_first_click_has_one_command(submission_harness):
    h = submission_harness
    barrier = Barrier(3)

    def race(i):
        barrier.wait()
        return submit(h, RefundSubmitInput(request_id=f"first-{i}", confirm=True))

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(race, range(3)))
    assert len({r.submission_id for r in results}) == 1
    replay = h.submitter.resume(h.offer.conversation_id, results[0].submission_id, {})
    assert replay.receipt and counts(h) == (1, 1)


@pytest.mark.parametrize("concurrent", [False, True])
def test_distinct_offers_same_order_preserve_first_content(
    tmp_path, submission_harness, concurrent
):
    from resolv.repositories.refund_flow import RefundFlowRepository
    from resolv.services.refund_offer import finalize_refund_reply
    from resolv.services.refund_submission import RefundSubmissionService
    from tests.refund_flow_fixtures import drive, flow_case

    h = submission_harness
    with flow_case(tmp_path, text="订单A001申请整单退款，因为另一个申请原因") as other:
        other.service.refund_clock = lambda: h.clock.now
        finalize_refund_reply(
            other.service, other.run.run_id, other.execution, drive(other)
        )
        other.service.mark_completed(other.run.run_id, other.execution.epoch)
        with other.factory() as s:
            offer = RefundFlowRepository(s, "local").offer_for_run(other.run.run_id)
        svc = RefundSubmissionService(
            other.factory,
            other.business,
            owner_id="local",
            identity=h.submitter.identity,
            facts=h.facts,
            selection_reader=lambda: other.profile.selection,
            profile=other.profile,
            clock=lambda: h.clock.now,
            execution_lock_factory=h.locks,
        )

        def second():
            return svc.submit(
                offer.conversation_id,
                offer.offer_id,
                RefundSubmitInput(request_id="other-offer", confirm=True),
                {},
            )

        if concurrent:
            barrier = Barrier(2)

            def wait(point):
                if point == "refund_before_business_begin":
                    barrier.wait(timeout=10)

            h.submitter.fault_hook = svc.fault_hook = wait
            with ThreadPoolExecutor(max_workers=2) as pool:
                a, b = pool.submit(submit, h), pool.submit(second)
                first, result = a.result(), b.result()
            assert sorted([first.receipt.created, result.receipt.created]) == [
                False,
                True,
            ]
        else:
            first, result = submit(h), second()
        assert result.receipt.application_id == first.receipt.application_id
        assert result.receipt.reason == first.receipt.reason
        assert result.receipt.money == first.receipt.money and counts(h) == (1, 2)
        with pytest.raises(RefundFlowError, match="refund_request_conflict"):
            svc.submit(offer.conversation_id, offer.offer_id, h.request, {})
