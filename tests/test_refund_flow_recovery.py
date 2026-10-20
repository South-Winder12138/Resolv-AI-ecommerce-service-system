"""Crash replay reuses saved stages and leaves unknown invocations visible."""

import pytest

from resolv.schemas.refund import RefundFlowError
from tests.refund_flow_fixtures import drive, flow_harness

__all__ = ["flow_harness"]


@pytest.mark.parametrize(
    "point",
    [
        "refund_after_binding_saved",
        "refund_after_query_saved",
        "refund_after_retrieval_saved",
        "refund_after_model_request_saved",
        "refund_after_observation_saved",
        "refund_after_decision_saved",
    ],
)
def test_saved_refund_stages_and_observations_reused(flow_harness, point):
    h = flow_harness

    def fault(p):
        if p == point:
            h.service.fault_hook = None
            raise RuntimeError("task-owned crash")

    h.service.fault_hook = fault
    with pytest.raises(RuntimeError, match="task-owned crash"):
        drive(h)
    record = drive(h)
    assert (
        record.decision.verdict == "eligible"
        and h.counts["prepare"] == h.counts["retrieve"] == 1
    )
    assert h.counts["decide"] == 2 and len(record.observations) == 3


def test_tool_failure_is_technical_and_unknown_attempt_is_visible(flow_harness):
    h = flow_harness
    h.facts.fail = TimeoutError("isolated read timeout")
    with pytest.raises(RefundFlowError) as error:
        drive(h)
    assert error.value.retryable
    h.facts.fail = None
    record = drive(h)
    assert record.decision.verdict == "eligible"
    assert any(a.outcome == "unknown" for a in record.read_audits)
    assert h.counts["query_refund_order"] == 2


def test_crash_before_observation_save_counts_unknown_then_retries_only_that_read(
    flow_harness,
):
    h = flow_harness

    def fault(point):
        if point == "refund_before_observation_saved":
            h.service.fault_hook = None
            raise RuntimeError("crash before persist")

    h.service.fault_hook = fault
    with pytest.raises(RuntimeError):
        drive(h)
    record = drive(h)
    assert any(a.outcome == "unknown" for a in record.read_audits)
    assert h.counts["query_refund_order"] == 2 and h.counts["query_refund_payment"] == 1


def test_unsaved_model_response_is_counted_unknown_before_explicit_resume(flow_harness):
    h = flow_harness

    def fault(point):
        if point == "refund_before_model_request_saved":
            h.service.fault_hook = None
            raise RuntimeError("model response lost")

    h.service.fault_hook = fault
    with pytest.raises(RuntimeError):
        drive(h)
    record = drive(h)
    assert h.counts["decide"] == 3
    assert any(
        a.outcome == "unknown" and a.error_code == "refund_response_not_saved"
        for a in record.audits
    )


def test_matching_call_id_reuses_read_but_changed_call_id_parameters_conflict(
    flow_harness,
):
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund_flow import RefundToolCall
    from resolv.services.hybrid_fingerprints import canonical_hash

    h = flow_harness
    h.client.mode = "loop"
    rid, e = h.run.run_id, h.execution
    for method in (h.flow.bind, h.flow.prepare_query, h.flow.retrieve):
        method(rid, e)
    h.flow.main_agent(rid, e)
    h.flow.read_next(rid, e)
    original = h.client.decide

    def repeat(messages, **kwargs):
        response = original(messages, **kwargs)
        calls = (
            RefundToolCall(
                id="s1-0", name="query_refund_order", args=FrozenJSON.from_value({})
            ),
        )
        return response.model_copy(
            update={
                "tool_calls": calls,
                "output_hash": canonical_hash(
                    {
                        "tool_calls": [c.model_dump(mode="json") for c in calls],
                        "decision": None,
                    }
                ),
            }
        )

    h.client.decide = repeat
    record = h.flow.main_agent(rid, e)
    h.flow.read_next(rid, e)
    # Third decision rebuilds native history using the reused observation.
    record = h.flow.main_agent(rid, e)
    assert len(record.model_requests) == 3 and h.counts["query_refund_order"] == 1

    def changed(messages, **kwargs):
        response = original(messages, **kwargs)
        calls = (
            RefundToolCall(
                id="s1-0", name="query_refund_payment", args=FrozenJSON.from_value({})
            ),
        )
        return response.model_copy(
            update={
                "tool_calls": calls,
                "output_hash": canonical_hash(
                    {
                        "tool_calls": [c.model_dump(mode="json") for c in calls],
                        "decision": None,
                    }
                ),
            }
        )

    h.client.decide = changed
    with pytest.raises(RefundFlowError):
        h.flow.main_agent(rid, e)
    assert len(h.flow._get(rid).model_requests) == 3


def test_recoverable_summary_failure_keeps_original_stages_and_run(flow_harness):
    from resolv.services.summarization import SummaryFailure

    h = flow_harness
    original = h.service.prepare_memory

    def fail(*a, **kw):
        raise SummaryFailure()

    h.service.prepare_memory = fail
    with pytest.raises(SummaryFailure):
        drive(h)
    assert (
        not h.service.get_run(h.run.run_id).memory_prepared and h.counts["decide"] == 0
    )
    h.service.prepare_memory = original
    record = drive(h)
    assert record.run_id == h.run.run_id and record.decision.verdict == "eligible"
    assert h.counts["prepare"] == h.counts["retrieve"] == 1


def test_invalid_final_is_audited_without_freezing_bad_response(flow_harness):
    from resolv.services.hybrid_fingerprints import canonical_hash

    h = flow_harness
    original = h.client.decide

    def invalid(messages, **kwargs):
        response = original(messages, **kwargs)
        if response.decision is None:
            return response
        wrong = response.decision.model_copy(update={"evidence_hash": "0" * 64})
        return response.model_copy(
            update={
                "decision": wrong,
                "output_hash": canonical_hash(
                    {"tool_calls": [], "decision": wrong.model_dump(mode="json")}
                ),
            }
        )

    h.client.decide = invalid
    with pytest.raises(RefundFlowError):
        drive(h)
    record = h.flow._get(h.run.run_id)
    assert len(record.model_requests) == 1 and record.decision is None
    assert any(
        a.outcome == "failed" and a.purpose == "refund_decision" for a in record.audits
    )
    h.client.decide = original
    assert drive(h).decision.verdict == "eligible"
    assert h.counts["decide"] == 3
