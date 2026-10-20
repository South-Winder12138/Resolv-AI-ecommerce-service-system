"""Actual policy windows and actual observations are prerequisites, not model claims."""

import pytest

from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund import RefundFlowError, RefundMoney
from resolv.schemas.refund_flow import RefundDecision, RefundRequest
from tests.knowledge_branch_fixtures import bundle
from tests.refund_fixtures import binding, observation, refund_owned

__all__ = ["refund_owned"]


def facts():
    b = binding()
    return (
        observation(
            bound=b,
            fields={
                "authorized": True,
                "subject_ref": "fixture-user",
                "order_id": b.order_id,
                "status": "paid",
                "activated": False,
            },
        ),
        observation(
            "query_refund_status",
            bound=b,
            call_id="status",
            fields={"authorized": True, "local_application": None},
        ),
        observation(
            "query_refund_payment",
            bound=b,
            call_id="payment",
            fields={
                "refundable_money": RefundMoney(
                    currency="CNY", minor_units=100, unit_precision=2
                ).model_dump(mode="json")
            },
        ),
    )


def decision(*, verdict="eligible", **changes):
    e = bundle()
    b = binding()
    o = facts()
    quote = e.chunks[0].chunk.answer
    citation = {
        "citation_number": 1,
        "window_id": e.windows[0].window_id,
        "quote": quote,
    }
    fact = {"observation_id": o[0].observation_id, "field": "activated"}
    data = {
        "operation": "apply",
        "verdict": verdict,
        "order_binding_hash": b.binding_hash,
        "evidence_hash": e.evidence_hash,
        "observation_refs": [v.observation_id for v in o],
        "policy_citations": [citation],
        "rules": [
            {"rule": "未激活条件", "policy_citations": [citation], "fact_refs": [fact]}
        ],
        "reason": "符合所引用的申请条件，提交后等待审核。",
        "critical_fields": [fact],
        "next_action": "show_offer" if verdict == "eligible" else "none",
    }
    return RefundDecision.model_validate_json(canonical_json(data | changes))


def validate(h, value=None, *, observations=None, evidence=None):
    from resolv.services.refund_decision import validate_refund_decision

    return validate_refund_decision(
        value or decision(),
        request=RefundRequest(operation="apply"),
        binding=binding(),
        evidence=evidence or bundle(),
        observations=observations if observations is not None else facts(),
        profile=h.profile,
    )


def test_decision_requires_policy_and_real_observations(refund_owned):
    assert validate(refund_owned).verdict == "eligible"
    for obs in [(), facts()[1:], facts()[:1], facts()[:2]]:
        with pytest.raises(RefundFlowError):
            validate(refund_owned, observations=obs)
    with pytest.raises(RefundFlowError):
        validate(refund_owned, evidence=bundle(()))
    with pytest.raises(RefundFlowError):
        validate(refund_owned, decision(order_binding_hash="0" * 64))
    forged = {"citation_number": 1, "window_id": "0" * 64, "quote": "伪造政策"}
    with pytest.raises(RefundFlowError):
        validate(refund_owned, decision(policy_citations=[forged]))
    with pytest.raises(RefundFlowError):
        validate(
            refund_owned,
            decision(
                critical_fields=[{"observation_id": "invented", "field": "activated"}]
            ),
        )


def test_unknown_critical_fact_allows_only_undetermined(refund_owned):
    obs = (
        *facts()[:2],
        observation("query_refund_payment", call_id="payment", fields={}),
    )
    d = decision(
        verdict="undetermined",
        next_action="request_information",
        missing_information=["权威整单可退金额"],
        critical_fields=[],
    )
    assert validate(refund_owned, d, observations=obs).verdict == "undetermined"
    with pytest.raises(RefundFlowError):
        validate(refund_owned, observations=obs)
    obs = (
        *facts()[:1],
        observation(
            "query_refund_status",
            call_id="status",
            fields={"authorized": True, "local_application": {"status": "rejected"}},
        ),
        facts()[2],
    )
    with pytest.raises(RefundFlowError):
        validate(refund_owned, observations=obs)


def test_fake_fact_field_and_unavailable_observation_rejected(refund_owned):
    with pytest.raises(RefundFlowError):
        validate(
            refund_owned,
            decision(
                critical_fields=[
                    {"observation_id": facts()[0].observation_id, "field": "invented"}
                ]
            ),
        )
    obs = (observation(availability="unknown"), *facts()[1:])
    with pytest.raises(RefundFlowError):
        validate(refund_owned, observations=obs)


def test_status_can_report_local_observation_without_policy(refund_owned):
    from resolv.services.refund_decision import validate_refund_decision

    e = bundle(())
    d = decision(
        operation="status",
        verdict=None,
        next_action="show_status",
        rules=[],
        policy_citations=[],
        critical_fields=[],
        evidence_hash=e.evidence_hash,
    )
    value = validate_refund_decision(
        d,
        request=RefundRequest(operation="status"),
        binding=binding(),
        evidence=e,
        observations=facts(),
        profile=refund_owned.profile,
    )
    assert value.verdict is None


def test_recheck_condition_requires_actual_quote_and_explicit_comparison(refund_owned):
    from resolv.schemas.refund_flow import RefundRecheckCondition

    condition = RefundRecheckCondition(
        fact_ref={"observation_id": facts()[0].observation_id, "field": "activated"},
        policy_window_id=bundle().windows[0].window_id,
        policy_quote="虚构政策",
        operator="equals",
        expected=False,
    )
    value = decision().model_copy(update={"recheck_conditions": (condition,)})
    with pytest.raises(RefundFlowError):
        validate(refund_owned, value)
    condition = condition.model_copy(
        update={
            "policy_quote": bundle().chunks[0].chunk.answer,
            "operator": "elapsed_seconds_lte",
            "expected": 604800,
        }
    )
    with pytest.raises(RefundFlowError):
        validate(
            refund_owned,
            decision().model_copy(update={"recheck_conditions": (condition,)}),
        )


def test_null_critical_fact_cannot_authorize_offer(refund_owned):
    obs = (
        observation(
            fields={
                "authorized": True,
                "subject_ref": "fixture-user",
                "order_id": "A001",
                "activated": None,
            }
        ),
        *facts()[1:],
    )
    with pytest.raises(RefundFlowError):
        validate(refund_owned, observations=obs)


def test_empty_policy_predicate_cannot_authorize_offer(refund_owned):
    from resolv.schemas.refund_flow import RefundRecheckCondition

    condition = RefundRecheckCondition(
        fact_ref={"observation_id": facts()[0].observation_id, "field": "activated"},
        policy_window_id=bundle().windows[0].window_id,
        policy_quote=bundle().chunks[0].chunk.answer,
        operator="equals",
        expected="",
    )
    with pytest.raises(RefundFlowError):
        validate(
            refund_owned,
            decision().model_copy(update={"recheck_conditions": (condition,)}),
        )


def test_later_payment_observation_cannot_be_omitted_from_eligibility(refund_owned):
    newer = observation(
        "query_refund_payment",
        call_id="new-payment",
        fields={
            "refundable_money": RefundMoney(
                currency="CNY", minor_units=200, unit_precision=2
            ).model_dump(mode="json")
        },
    )
    with pytest.raises(RefundFlowError):
        validate(refund_owned, observations=(*facts(), newer))


def test_model_cannot_reorder_observations_to_override_latest_critical_fact(
    refund_owned,
):
    newer = observation(
        call_id="new-order",
        fields={
            "authorized": True,
            "subject_ref": "fixture-user",
            "order_id": "A001",
            "activated": True,
        },
    )
    refs = [newer.observation_id, *[o.observation_id for o in facts()]]
    with pytest.raises(RefundFlowError):
        validate(
            refund_owned,
            decision(observation_refs=refs),
            observations=(*facts(), newer),
        )


def test_unavailable_latest_payment_does_not_reuse_older_known_amount(refund_owned):
    newer = observation(
        "query_refund_payment", call_id="new-payment", availability="unknown"
    )
    refs = [o.observation_id for o in (*facts(), newer)]
    with pytest.raises(RefundFlowError):
        validate(
            refund_owned,
            decision(observation_refs=refs),
            observations=(*facts(), newer),
        )
