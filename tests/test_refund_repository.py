"""Real isolated SQL transactions; no external services or model calls."""

from datetime import UTC, datetime, timedelta

import pytest

from resolv.schemas.conversation import ConversationError
from resolv.schemas.knowledge_branch import FrozenJSON
from resolv.schemas.refund import OrderBinding, RefundFlowError, RefundMoney
from resolv.schemas.refund_flow import (
    RefundModelAudit,
    RefundModelResponse,
    RefundToolCall,
)
from resolv.schemas.refund_submission import RefundOfferRecord, RefundReceipt
from resolv.services.hybrid_fingerprints import canonical_hash
from tests.refund_fixtures import NOW, observation, refund_case, refund_owned

__all__ = ["refund_owned"]


def create(h):
    from resolv.repositories.refund_flow import RefundFlowRepository

    with h.factory.begin() as session:
        return RefundFlowRepository(session, "local").create(
            h.run.run_id, h.execution.epoch, h.primary, h.profile
        )


def bound(h):
    return OrderBinding.create(
        order_source=h.profile.adapter.namespace,
        order_id="A001",
        understanding_id=h.primary.understanding_id,
        primary_task_id=h.primary.primary.task_id,
        primary_hash=canonical_hash(h.primary.model_dump(mode="json")),
        source_refs=h.primary.primary.evidence_refs,
    )


def binding_stage(h):
    from resolv.schemas.refund_flow import RefundBindingStage, RefundRequest

    return RefundBindingStage(
        binding=bound(h), request=RefundRequest(operation="apply")
    )


def offer(h, oid="offer-1"):
    return RefundOfferRecord(
        offer_id=oid,
        owner_id="local",
        conversation_id=h.run.conversation_id,
        run_id=h.run.run_id,
        primary_task_id=h.primary.primary.task_id,
        binding=bound(h),
        subject_ref="fixture-user",
        money=RefundMoney(currency="CNY", minor_units=100, unit_precision=2),
        reason="未激活",
        selection=h.profile.selection,
        evidence_hash="e" * 64,
        decision_hash="d" * 64,
        critical_facts=FrozenJSON.from_value({}),
        profile=h.profile,
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=30),
    )


def test_refund_stage_write_once_and_identity_fence(refund_owned):
    from resolv.repositories.refund_flow import RefundFlowRepository

    h = refund_owned
    record = create(h)
    for _ in range(2):
        with h.factory.begin() as s:
            result = RefundFlowRepository(s, "local").save_stage(
                h.run.run_id,
                h.execution.epoch,
                "binding",
                binding_stage(h),
                expected_input_hash=record.primary_hash,
            )
    assert result.binding == bound(h) and len(result.stage_hashes) == 1
    for owner, epoch, digest in [
        ("other", h.execution.epoch, record.primary_hash),
        ("local", h.execution.epoch - 1, record.primary_hash),
        ("local", h.execution.epoch, "0" * 64),
    ]:
        with (
            pytest.raises((RefundFlowError, ConversationError)),
            h.factory.begin() as s,
        ):
            RefundFlowRepository(s, owner).save_stage(
                h.run.run_id,
                epoch,
                "binding",
                binding_stage(h),
                expected_input_hash=digest,
            )
    with h.factory() as s:
        assert RefundFlowRepository(s, "other").get(h.run.run_id) is None
    assert "refund_config" not in h.service.get_run(h.run.run_id).model_dump()
    changed = binding_stage(h).model_copy(
        update={
            "binding": OrderBinding.create(
                **(bound(h).model_dump(exclude={"binding_hash"}) | {"order_id": "A002"})
            )
        }
    )
    with pytest.raises((RefundFlowError, ValueError)), h.factory.begin() as s:
        RefundFlowRepository(s, "local").save_stage(
            h.run.run_id,
            h.execution.epoch,
            "binding",
            changed,
            expected_input_hash=record.primary_hash,
        )


@pytest.mark.parametrize("change", ["primary", "profile", "conversation", "hash"])
def test_saved_run_corruption_rejected(refund_owned, change):
    from resolv.models.refund_flow import ConversationRefundRun
    from resolv.repositories.refund_flow import RefundFlowRepository

    h = refund_owned
    create(h)
    other = h.service.create() if change == "conversation" else None
    with h.factory.begin() as s:
        row = s.get(ConversationRefundRun, h.run.run_id)
        if change == "primary":
            row.primary_json = {}
        if change == "profile":
            row.profile_json = {}
        if change == "conversation":
            row.conversation_id = other.conversation_id
        if change == "hash":
            row.stage_data = {"binding": {"input_hash": "0" * 64}}
    with pytest.raises(RefundFlowError), h.factory() as s:
        RefundFlowRepository(s, "local").get(h.run.run_id)


def test_saved_model_request_and_observation_are_bound_and_write_once(refund_owned):
    from resolv.repositories.refund_flow import RefundFlowRepository

    h = refund_owned
    r = create(h)
    with h.factory.begin() as s:
        repo = RefundFlowRepository(s, "local")
        repo.save_stage(
            h.run.run_id,
            h.execution.epoch,
            "binding",
            binding_stage(h),
            expected_input_hash=r.primary_hash,
        )
        audit = RefundModelAudit(
            invocation_id="i",
            purpose="refund_decision",
            profile_fingerprint=h.profile.decision_model.fingerprint,
            input_hash="a" * 64,
            input_estimate=100,
            model_step=1,
            started_at=NOW,
            outcome="succeeded",
        )
        calls = (
            RefundToolCall(
                id="call-1", name="query_refund_order", args=FrozenJSON.from_value({})
            ),
        )
        reply = RefundModelResponse(
            model_step=1,
            tool_calls=calls,
            input_hash="a" * 64,
            output_hash=canonical_hash(
                {
                    "tool_calls": [c.model_dump(mode="json") for c in calls],
                    "decision": None,
                }
            ),
            audit=audit,
        )
        repo.save_model_request(h.run.run_id, h.execution.epoch, reply)
        obs = observation(bound=bound(h))
        for _ in range(2):
            repo.save_observation(h.run.run_id, h.execution.epoch, 1, obs)
    with pytest.raises(RefundFlowError), h.factory.begin() as s:
        RefundFlowRepository(s, "local").save_observation(
            h.run.run_id,
            h.execution.epoch,
            1,
            observation(bound=bound(h), call_id="unknown"),
        )
    with pytest.raises(RefundFlowError), h.factory.begin() as s:
        RefundFlowRepository(s, "local").save_observation(
            h.run.run_id,
            h.execution.epoch,
            1,
            observation(bound=bound(h), fields={"status": "changed"}),
        )
    with h.factory() as s:
        assert len(RefundFlowRepository(s, "local").get(h.run.run_id).observations) == 1


def save_offer(h):
    from resolv.repositories.refund_flow import RefundFlowRepository

    create(h)
    with h.factory.begin() as s:
        return RefundFlowRepository(s, "local").put_offer(
            h.run.run_id, h.execution.epoch, offer(h)
        )


def test_alias_conflicts_and_offer_command_uniqueness(refund_owned):
    from resolv.repositories.refund_flow import RefundFlowRepository
    from resolv.repositories.refund_submission import RefundSubmissionRepository

    h = refund_owned
    save_offer(h)
    with h.factory.begin() as s:
        repo = RefundSubmissionRepository(s, "local")
        first = repo.accept("offer-1", "r1", "a" * 64)
        alias = repo.accept("offer-1", "r2", "a" * 64)
        assert first.submission_id == alias.submission_id
    with pytest.raises(RefundFlowError), h.factory.begin() as s:
        RefundSubmissionRepository(s, "local").accept("offer-1", "r1", "b" * 64)
    with pytest.raises(RefundFlowError), h.factory.begin() as s:
        RefundSubmissionRepository(s, "other").accept("offer-1", "r1", "a" * 64)
    with refund_case(h.factory) as other:
        create(other)
        with h.factory.begin() as s:
            RefundFlowRepository(s, "local").put_offer(
                other.run.run_id, other.execution.epoch, offer(other, "offer-2")
            )
        with pytest.raises(RefundFlowError), h.factory.begin() as s:
            RefundSubmissionRepository(s, "local").accept("offer-2", "r1", "a" * 64)


def test_submission_lease_is_independent_and_fenced(refund_owned):
    from resolv.repositories.refund_submission import RefundSubmissionRepository

    h = refund_owned
    now = datetime.now(UTC)
    save_offer(h)
    with h.factory.begin() as s:
        repo = RefundSubmissionRepository(s, "local")
        command = repo.accept("offer-1", "r1", "a" * 64)
        first = repo.claim(command.submission_id, "worker1", now, 10)
    with pytest.raises(RefundFlowError), h.factory.begin() as s:
        RefundSubmissionRepository(s, "local").claim(
            command.submission_id, "worker2", now, 10
        )
    with h.factory.begin() as s:
        repo = RefundSubmissionRepository(s, "local")
        second = repo.claim(
            command.submission_id, "worker2", now + timedelta(seconds=11), 10
        )
        assert second.epoch == first.epoch + 1
        with pytest.raises(RefundFlowError):
            repo.owned(first)
        receipt = RefundReceipt(
            application_id="app",
            order_source=bound(h).order_source,
            order_id="A001",
            status="pending",
            created=True,
            money=offer(h).money.public(),
            reason="未激活",
            created_at=NOW,
        )
        assert repo.publish(second, receipt).receipt == receipt
    with h.factory() as s:
        assert (
            RefundSubmissionRepository(s, "local").get(command.submission_id).receipt
            == receipt
        )


def test_expired_submission_worker_cannot_publish_without_takeover(refund_owned):
    from resolv.repositories.refund_submission import RefundSubmissionRepository

    h = refund_owned
    save_offer(h)
    with h.factory.begin() as s:
        repo = RefundSubmissionRepository(s, "local")
        command = repo.accept("offer-1", "r1", "a" * 64)
        lease = repo.claim(
            command.submission_id,
            "worker1",
            datetime.now(UTC) - timedelta(minutes=1),
            10,
        )
    with pytest.raises(RefundFlowError), h.factory.begin() as s:
        RefundSubmissionRepository(s, "local").owned(lease)


def test_offer_rollback_and_run_uniqueness(refund_owned):
    from resolv.repositories.refund_flow import RefundFlowRepository

    h = refund_owned
    create(h)
    with pytest.raises(RuntimeError), h.factory.begin() as s:
        RefundFlowRepository(s, "local").put_offer(
            h.run.run_id, h.execution.epoch, offer(h)
        )
        raise RuntimeError("rollback")
    with h.factory() as s:
        assert RefundFlowRepository(s, "local").get_offer("offer-1") is None
    save_offer(h)
    with pytest.raises(RefundFlowError), h.factory.begin() as s:
        RefundFlowRepository(s, "local").put_offer(
            h.run.run_id, h.execution.epoch, offer(h, "offer-2")
        )
