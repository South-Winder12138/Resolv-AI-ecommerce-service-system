"""Fixed refund evidence/facts/schema survive optional history selection intact."""

import json
from types import SimpleNamespace

import pytest

from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_flow import RefundRequest
from tests.knowledge_branch_fixtures import bundle
from tests.refund_fixtures import refund_owned
from tests.test_refund_decision import facts
from tests.test_refund_repository import bound, create

__all__ = ["refund_owned"]


class Counter:
    def count(self, messages, schemas):
        assert len(schemas) in {0, 4}
        return (
            sum(len(m.content) for m in messages) // 10 + len(json.dumps(schemas)) // 10
        )

    def count_text(self, text):
        return len(text) // 10


def test_fixed_context_budget_fails_without_truncation(refund_owned):
    from resolv.services.refund_context import build_refund_context

    h = refund_owned
    r = create(h)
    e = bundle()
    # This test uses no tool call history; observed facts still enter fixed input.
    result = build_refund_context(
        r,
        RefundRequest(operation="apply"),
        bound(h),
        e,
        (),
        history=[],
        summary=None,
        counter=Counter(),
        profile=h.profile,
    )
    data = result.messages.value
    text = "".join(m["data"]["content"] for m in data)
    assert h.primary.primary.text in text and e.serialized in text
    assert result.input_estimate <= h.profile.decision_model.input_limit

    class Overflow(Counter):
        def count(self, messages, schemas):
            return 12001

    with pytest.raises(RefundFlowError) as error:
        build_refund_context(
            r,
            RefundRequest(operation="apply"),
            bound(h),
            e,
            (),
            history=[],
            summary=None,
            counter=Overflow(),
            profile=h.profile,
        )
    assert error.value.retryable is False


def test_observation_binding_is_checked_before_context(refund_owned):
    from resolv.services.refund_context import build_refund_context

    h = refund_owned
    with pytest.raises(RefundFlowError):
        build_refund_context(
            create(h),
            RefundRequest(operation="apply"),
            bound(h),
            bundle(),
            facts(),
            history=[],
            summary=None,
            counter=Counter(),
            profile=h.profile,
        )


def test_optional_history_does_not_push_out_fixed_evidence(refund_owned):
    from langchain_core.messages import AIMessage, HumanMessage

    from resolv.services.refund_context import build_refund_context

    h = refund_owned
    e = bundle()
    messages = [
        HumanMessage(content="旧问题" * 100000),
        AIMessage(content="旧回答" * 100000),
    ]
    result = build_refund_context(
        create(h),
        RefundRequest(operation="apply"),
        bound(h),
        e,
        (),
        history=messages,
        summary=None,
        counter=Counter(),
        profile=h.profile,
    )
    text = "".join(m["data"]["content"] for m in result.messages.value)
    assert e.serialized in text and "旧问题" not in text


def test_refund_summary_has_separate_purpose_and_propagates_storage_failure(
    refund_owned,
):
    from langchain_core.messages import AIMessage

    from resolv.core.config import SummaryModelSettings
    from resolv.services.business_summary import summary_profile
    from resolv.services.context import TokenCounter
    from resolv.services.refund_context import RefundSummaryAdapter
    from resolv.services.summarization import SummaryService

    settings = SummaryModelSettings(
        _env_file=None,
        api_key="offline",
        model_name="fixture-summary",
        base_url="https://offline.invalid/v1",
        model_context_window_tokens=16000,
    )

    class Model:
        def invoke(self, messages):
            return AIMessage(
                content='{"user_goal":"","confirmed_facts":[],"completed_actions":[],"open_questions":[]}',
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "total_tokens": 20,
                },
                response_metadata={"finish_reason": "stop"},
            )

    base = SummaryService(Model(), TokenCounter("o200k_base"), settings)
    h = refund_owned
    profile = h.profile.model_copy(
        update={
            "summary_profile": __import__(
                "resolv.schemas.knowledge_branch", fromlist=["FrozenJSON"]
            ).FrozenJSON.from_value(summary_profile(settings))
        }
    )
    audits = []
    adapter = RefundSummaryAdapter(base, profile, audits.append)
    wrapped = adapter.for_run(SimpleNamespace(), None)
    wrapped.model.invoke([])
    assert [a.purpose for a in audits] == ["refund_summary", "refund_summary"]

    def broken(a):
        raise OSError("storage")

    wrapped = RefundSummaryAdapter(base, profile, broken).for_run(
        SimpleNamespace(), None
    )
    with pytest.raises(OSError):
        wrapped.model.invoke([])


def test_real_token_counter_accounts_for_frozen_final_schema(refund_owned):
    from resolv.core.refund_config import default_refund_profile
    from resolv.services.context import TokenCounter
    from resolv.services.refund_context import build_refund_context

    h = refund_owned
    p = default_refund_profile(
        understanding_config=h.profile.query_model,
        business_config=h.profile.decision_model,
        summary_profile={},
        adapter_descriptor=h.profile.adapter,
    )
    run = create(h).model_copy(update={"profile": p})
    result = build_refund_context(
        run,
        RefundRequest(operation="apply"),
        bound(h),
        bundle(),
        (),
        history=[],
        summary=None,
        counter=TokenCounter("o200k_base"),
        profile=p,
    )
    assert (
        p.decision_model.provider_input_overhead_tokens
        < result.input_estimate
        <= p.decision_model.input_limit
    )
    assert "policy_citations" in result.messages.value[0]["data"]["content"]


def test_native_observations_are_fixed_and_evaluation_time_is_frozen(refund_owned):
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund_flow import (
        RefundModelAudit,
        RefundModelResponse,
        RefundToolCall,
    )
    from resolv.services.hybrid_fingerprints import canonical_hash
    from resolv.services.refund_context import build_refund_context
    from tests.refund_fixtures import NOW, observation

    h = refund_owned
    b = bound(h)
    calls = (
        RefundToolCall(
            id="tc", name="query_refund_order", args=FrozenJSON.from_value({})
        ),
    )
    audit = RefundModelAudit(
        invocation_id="i",
        purpose="refund_decision",
        profile_fingerprint=h.profile.decision_model.fingerprint,
        input_hash="a" * 64,
        model_step=1,
        input_estimate=1,
        started_at=NOW,
        outcome="succeeded",
    )
    response = RefundModelResponse(
        model_step=1,
        tool_calls=calls,
        input_hash=audit.input_hash,
        output_hash=canonical_hash(
            {"tool_calls": [c.model_dump(mode="json") for c in calls], "decision": None}
        ),
        audit=audit,
    )
    obs = observation(bound=b, call_id="tc", fields={"order_id": "A001"})
    run = create(h).model_copy(
        update={"model_requests": (response,), "observations": (obs,)}
    )

    def build(obs):
        return build_refund_context(
            run,
            RefundRequest(operation="apply"),
            b,
            bundle(),
            (obs,),
            history=[],
            summary=None,
            counter=Counter(),
            profile=h.profile,
        )

    result = build(obs)
    current = json.loads(result.messages.value[1]["data"]["content"])
    assert (
        current["evaluated_at"] == NOW.isoformat()
        and current["business_timezone"] == "Asia/Shanghai"
    )
    assert any(
        m["type"] == "tool" and '"order_id"' in m["data"]["content"]
        for m in result.messages.value
    )
    huge = observation(
        bound=b, call_id="tc", fields={"order_id": "A001", "detail": "x" * 200000}
    )
    with pytest.raises(RefundFlowError, match="refund_model_input_budget_exceeded"):
        build(huge)
