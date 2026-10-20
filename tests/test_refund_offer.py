"""Offer and answer share one transaction; no approval interruption or application write."""

from datetime import timedelta

import pytest
from sqlalchemy import func, select

from resolv.models.conversation import ConversationMessage
from resolv.models.refund_flow import ConversationRefundOffer
from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.schemas.conversation import ConversationError
from resolv.schemas.refund import RefundFlowError
from tests.refund_fixtures import NOW
from tests.refund_flow_fixtures import drive, flow_case, flow_harness

__all__ = ["flow_harness"]


def stored_offer(h):
    with h.factory() as s:
        return RefundFlowRepository(s, "local").offer_for_run(h.run.run_id)


def test_offer_and_answer_atomic_and_stable_expiry(flow_harness):
    from resolv.services.refund_offer import finalize_refund_reply
    from resolv.services.refund_reply_projection import public_refund_reply

    h = flow_harness
    record = drive(h)
    h.service.refund_clock = lambda: NOW

    def fail(point):
        if point == "refund_before_finalize_commit":
            raise RuntimeError("atomic rollback")

    h.service.fault_hook = fail
    with pytest.raises(RefundFlowError):
        finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
    with h.factory() as s:
        assert s.get(ConversationMessage, h.run.answer_message_id) is None
        assert s.scalar(select(func.count()).select_from(ConversationRefundOffer)) == 0
    h.service.fault_hook = None
    reply = finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
    offer = stored_offer(h)
    assert offer.expires_at - offer.created_at == timedelta(minutes=30)
    assert offer.created_at == NOW and reply.metadata.refund.offer_id == offer.offer_id
    with pytest.raises(ConversationError):
        public_refund_reply(h.service.get_run(h.run.run_id), record, offer)
    h.service.refund_clock = lambda: NOW + timedelta(hours=1)
    assert finalize_refund_reply(h.service, h.run.run_id, h.execution, record) == reply
    assert stored_offer(h) == offer
    h.service.mark_completed(h.run.run_id, h.execution.epoch)
    run = h.service.get_run(h.run.run_id)
    assert public_refund_reply(run, record, offer) == reply.metadata.model_dump(
        mode="json"
    )
    with h.factory() as s:
        assert s.get(ConversationRefundOffer, offer.offer_id)
        assert (
            s.scalar(
                select(func.count())
                .select_from(ConversationMessage)
                .where(ConversationMessage.role == "assistant")
            )
            == 1
        )


@pytest.mark.parametrize("length,expected", [(0, False), (500, True), (501, False)])
def test_reason_unicode_limit_and_missing_reason(tmp_path, length, expected):
    from resolv.services.refund_offer import finalize_refund_reply

    reason = "😀" * length
    text = "订单A001申请整单退款" + ("，因为" + reason if reason else "")
    with flow_case(tmp_path, text=text) as h:
        record = drive(h)
        h.service.refund_clock = lambda: NOW
        reply = finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
        offer = stored_offer(h)
        assert (offer is not None) == expected
        if expected:
            assert offer.reason == reason and len(offer.reason) == 500
        else:
            assert "原因" in reply.answer and reply.metadata.refund.offer_id is None


def test_finalize_after_commit_unknown_reconciles_original_offer(flow_harness):
    from resolv.services.refund_offer import finalize_refund_reply

    h = flow_harness
    record = drive(h)
    h.service.refund_clock = lambda: NOW

    def fault(point):
        if point == "refund_after_finalize_commit":
            raise OSError("lost response")

    h.service.fault_hook = fault
    reply = finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
    assert reply.metadata.refund.offer_id == stored_offer(h).offer_id


def test_manual_draft_does_not_create_offer(tmp_path):
    from resolv.services.refund_offer import finalize_refund_reply

    with flow_case(tmp_path, text="订单A001申请保修") as h:
        record = drive(h)
        reply = finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
        assert reply.metadata.suggested_actions[0].preview_only is True
        assert stored_offer(h) is None
