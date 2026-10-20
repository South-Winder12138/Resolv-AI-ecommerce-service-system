"""Serial readonly reasoning and hard six-round ceiling."""

from sqlalchemy import func, select

from resolv.models.refund_application import RefundApplication, RefundSubmissionReceipt
from tests.refund_flow_fixtures import drive, flow_case, flow_harness

__all__ = ["flow_harness"]


def test_refund_reasoning_never_creates_application(flow_harness):
    h = flow_harness
    record = drive(h)
    assert record.decision.verdict == "eligible" and len(record.model_requests) == 2
    with h.business() as s:
        assert s.scalar(select(func.count()).select_from(RefundApplication)) == 0
        assert s.scalar(select(func.count()).select_from(RefundSubmissionReceipt)) == 0
    assert h.counts["prepare"] == h.counts["retrieve"] == 1
    assert len(record.observations) == 3


def test_sixth_refund_round_never_executes_pending_tools(flow_harness):
    h = flow_harness
    h.client.mode = "loop"
    record = drive(h)
    assert record.step_limit and record.decision is None
    assert len(record.model_requests) == 6 and len(record.observations) == 5
    assert h.counts["query_refund_order"] == 5
    assert h.flow.read_next(h.run.run_id, h.execution) == record
    assert h.flow.main_agent(h.run.run_id, h.execution) == record


def test_status_without_policy_only_reports_observed_application_facts(tmp_path):
    with flow_case(
        tmp_path, mode="status", policy=False, text="订单A001退款进度如何"
    ) as h:
        record = drive(h)
        assert record.decision.operation == "status" and record.decision.verdict is None
        assert record.evidence.windows == ()
        assert h.counts["prepare"] == h.counts["retrieve"] == 0


def test_normal_empty_policy_evidence_is_information_gap(flow_harness):
    from tests.knowledge_branch_fixtures import hybrid_result

    h = flow_harness
    h.client.mode = "undetermined"
    h.flow.retrievers.retrieve = lambda *a, **k: hybrid_result(())
    record = drive(h)
    assert record.decision.verdict == "undetermined" and record.error_code is None
    assert record.evidence.windows == () and record.decision.next_action != "show_offer"


def test_unconnected_identity_does_not_start_models_or_external_reads(flow_harness):
    from resolv.services.refund_adapters import UnconnectedRefundIdentity

    h = flow_harness
    h.flow.identity = UnconnectedRefundIdentity()
    record = drive(h)
    assert record.scope_reply.kind == "unavailable"
    assert (
        h.counts["prepare"]
        == h.counts["retrieve"]
        == h.counts["decide"]
        == h.counts["authorize"]
        == 0
    )


def test_existing_application_is_displayed_without_policy_and_never_reapplied(tmp_path):
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund_submission import RefundRegistrationCommand
    from resolv.services.refund_registration import RefundRegistrationService
    from tests.refund_fixtures import NOW

    with flow_case(tmp_path, mode="existing", policy=False) as h:
        record = h.flow.bind(h.run.run_id, h.execution)
        with h.business.begin() as s:
            command = RefundRegistrationCommand(
                command_id="old-command",
                offer_id="old-offer",
                request_hash="a" * 64,
                binding=record.binding,
                subject_ref="fixture-subject",
                money=__import__(
                    "resolv.schemas.refund", fromlist=["RefundMoney"]
                ).RefundMoney(currency="CNY", minor_units=100, unit_precision=2),
                reason="原申请原因",
                decision_hash="d" * 64,
                evidence_hash="e" * 64,
                revalidation_hash="f" * 64,
                snapshots=FrozenJSON.from_value({}),
                created_at=NOW,
            )
            receipt = RefundRegistrationService(s).register(command)
            s.get(RefundApplication, receipt.application_id).status = "rejected"
        record = drive(h)
        assert (
            record.decision.next_action == "show_status"
            and record.decision.verdict == "undetermined"
        )
        assert "rejected" in record.decision.reason
        with h.business() as s:
            assert s.scalar(select(func.count()).select_from(RefundApplication)) == 1
            assert (
                s.get(RefundApplication, receipt.application_id).reason == "原申请原因"
            )
        assert h.counts["prepare"] == h.counts["retrieve"] == 0
