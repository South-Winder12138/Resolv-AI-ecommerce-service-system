"""Immutable v3 reply projection and independent dynamic offer status."""

from datetime import timedelta

import pytest

from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund import RefundFlowError
from resolv.services.hybrid_fingerprints import canonical_hash
from tests.refund_fixtures import NOW
from tests.refund_flow_fixtures import drive, flow_case, flow_harness
from tests.test_refund_offer import stored_offer

__all__ = ["flow_harness"]


def finalized(h):
    from resolv.services.refund_offer import finalize_refund_reply

    record = drive(h)
    h.service.refund_clock = lambda: NOW
    reply = finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
    h.service.mark_completed(h.run.run_id, h.execution.epoch)
    return record, reply, stored_offer(h)


def test_reply_v3_exact_json_and_legacy_hashes(flow_harness):
    from resolv.db.exact_json import ExactJSON
    from resolv.schemas.branch_execution import parse_reply_metadata
    from tests.test_knowledge_exact_json import lossy_native

    _, reply, _ = finalized(flow_harness)
    codec = ExactJSON(only_reply_v2=True)
    value = reply.metadata.model_dump(mode="json")
    assert canonical_hash(
        codec.process_result_value(
            lossy_native(codec.process_bind_param(value, None)), None
        )
    ) == canonical_hash(value)
    assert parse_reply_metadata(value) == reply.metadata
    old = {"schema_version": 1, "branch": "business"}
    assert codec.process_bind_param(old, None) == old
    old2 = {"schema_version": 2, "knowledge": {"score": 0.9919943477637045}}
    assert codec.process_bind_param(old2, None) == {
        "__resolv_exact_json__": 1,
        "payload": canonical_json(old2),
    }
    with pytest.raises(ValueError):
        parse_reply_metadata(value | {"schema_version": 4})
    corrupt = __import__("json").loads(reply.metadata.model_dump_json())
    corrupt["refund"]["evidence"]["chunks"] = []
    with pytest.raises(ValueError):
        parse_reply_metadata(corrupt)
    assert "subject_ref" not in canonical_json(
        value
    ) and "execution_owner_id" not in canonical_json(value)


def test_submitted_receipt_survives_offer_expiry_without_rewriting_reply(flow_harness):
    from resolv.schemas.refund_submission import RefundReceipt, RefundSubmissionRecord
    from resolv.services.refund_reply_projection import (
        public_refund_offer,
        public_refund_reply,
    )

    h = flow_harness
    record, reply, offer = finalized(h)
    before = reply.metadata.model_dump_json()
    assert (
        public_refund_offer(
            offer, None, None, now=offer.expires_at - timedelta(microseconds=1)
        ).validity
        == "available"
    )
    assert (
        public_refund_offer(offer, None, None, now=offer.expires_at).validity
        == "expired"
    )
    receipt = RefundReceipt(
        application_id="app",
        order_source=offer.binding.order_source,
        order_id=offer.binding.order_id,
        status="pending",
        created=True,
        money=offer.money.public(),
        reason=offer.reason,
        created_at=NOW,
    )
    submission = RefundSubmissionRecord(
        submission_id="command",
        offer_id=offer.offer_id,
        owner_id=offer.owner_id,
        conversation_id=offer.conversation_id,
        request_hash="a" * 64,
        status="completed",
        receipt=receipt,
        created_at=NOW,
        updated_at=NOW,
    )
    view = public_refund_offer(
        offer, submission, receipt, now=offer.expires_at + timedelta(days=1)
    )
    assert view.validity == "submitted" and view.submission.receipt == receipt
    assert reply.metadata.model_dump_json() == before
    assert public_refund_reply(
        h.service.get_run(h.run.run_id), record, offer
    ) == reply.metadata.model_dump(mode="json")
    with pytest.raises(RefundFlowError):
        public_refund_offer(
            offer, submission.model_copy(update={"owner_id": "other"}), receipt, now=NOW
        )


def test_policy_unavailable_status_never_exposes_placeholder_index(tmp_path):
    with flow_case(
        tmp_path, mode="status", policy=False, text="订单A001退款状态如何"
    ) as h:
        _, reply, offer = finalized(h)
        assert (
            offer is None
            and reply.metadata.refund.release_id is None
            and reply.metadata.refund.index_version is None
        )
        assert "refund-no-policy" not in reply.metadata.model_dump_json()


def test_nested_contract_json_roundtrip_preserves_evidence_and_timestamp_rules(
    flow_harness,
):
    from resolv.schemas.refund_flow import RefundRunRecord

    record = drive(flow_harness)
    assert RefundRunRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    "invalid", ["2026-10-16 04:00:00+00:00", "20261016T040000+00:00"]
)
def test_timestamps_require_rfc3339_syntax(invalid):
    from resolv.schemas.refund_submission import RefundReceipt

    with pytest.raises(ValueError):
        RefundReceipt.model_validate_json(
            __import__("json").dumps(
                {
                    "application_id": "x",
                    "order_source": "a",
                    "order_id": "a",
                    "status": "pending",
                    "created": True,
                    "money": None,
                    "reason": "x",
                    "created_at": invalid,
                }
            )
        )


def test_submitted_marker_without_receipt_is_store_inconsistency(flow_harness):
    from resolv.services.refund_reply_projection import public_refund_offer

    _, _, offer = finalized(flow_harness)
    with pytest.raises(RefundFlowError):
        public_refund_offer(
            offer.model_copy(update={"status": "submitted"}), None, None, now=NOW
        )
