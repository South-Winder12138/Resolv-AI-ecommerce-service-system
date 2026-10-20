"""Fresh authoritative values and policy-defined time comparisons before writes."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from resolv.schemas.knowledge_branch import FrozenJSON
from resolv.schemas.refund import RefundFlowError, RefundMoney, RefundObservation
from resolv.schemas.refund_flow import RefundRecheckCondition
from tests.refund_fixtures import NOW, observation
from tests.refund_flow_fixtures import drive, flow_case


def replace(obs, **changes):
    data = obs.model_dump(exclude={"audit_hash", "original_observed_at"}) | changes
    return RefundObservation.create(**data)


@pytest.fixture
def h(tmp_path):
    from resolv.services.refund_offer import prepare_refund_offer

    with flow_case(tmp_path) as h:
        original = h.facts.query_payment

        def payment(access, bound, call_id):
            result = original(access, bound, call_id)
            fields = result.model_dump()["fields"]
            fields["paid_money"] = {
                "availability": "available",
                "value": RefundMoney(
                    currency="CNY", minor_units=100, unit_precision=2
                ).model_dump(mode="json"),
            }
            return replace(result, fields=fields)

        h.facts.query_payment = payment
        record = drive(h)
        yield SimpleNamespace(
            **(
                vars(h)
                | {
                    "record": record,
                    "offer": prepare_refund_offer(record, now=NOW),
                    "access": h.flow.identity.resolve(None),
                }
            )
        )


def recheck(h, **changes):
    from resolv.services.refund_revalidation import revalidate_refund_offer

    args = {
        "access": h.access,
        "adapter": h.facts,
        "registration": h.flow.registration,
        "current_selection": h.offer.selection,
        "now": NOW + timedelta(seconds=1),
        "profile": h.offer.profile,
    } | changes
    return revalidate_refund_offer(h.offer, **args)


def change_field(h, tool, field, value=None, *, availability="available"):
    method = getattr(h.facts, tool)

    def changed(access, bound, call_id):
        result = method(access, bound, call_id)
        fields = result.model_dump()["fields"]
        fields[field] = {"availability": availability, "value": value}
        return replace(result, fields=fields)

    setattr(h.facts, tool, changed)


def test_revalidation_checks_values_not_provider_version(h):
    assert recheck(h).outcome == "passed"
    original = h.facts.query_order

    def metadata(access, bound, call_id):
        result = original(access, bound, call_id)
        fields = result.model_dump()["fields"] | {
            "display_note": {"availability": "available", "value": "无关展示字段"}
        }
        return replace(
            result,
            provider_version="provider-v2",
            observed_at=NOW + timedelta(seconds=1),
            fields=fields,
        )

    h.facts.query_order = metadata
    assert recheck(h).outcome == "passed"
    change_field(h, "query_order", "activated", True)
    result = recheck(h)
    assert result.outcome == "needs_reassessment" and result.revalidation_hash


@pytest.mark.parametrize(
    "tool,field,value,availability",
    [
        ("query_order", "status", "delivered", "available"),
        ("query_order", "product_type", "hardware", "available"),
        (
            "query_payment",
            "paid_money",
            {"currency": "CNY", "minor_units": 200, "unit_precision": 2},
            "available",
        ),
        (
            "query_payment",
            "refundable_money",
            {"currency": "CNY", "minor_units": 200, "unit_precision": 2},
            "available",
        ),
        (
            "query_payment",
            "refundable_money",
            {"currency": "USD", "minor_units": 100, "unit_precision": 2},
            "available",
        ),
        (
            "query_payment",
            "refundable_money",
            {"currency": "CNY", "minor_units": 100, "unit_precision": 3},
            "available",
        ),
        (
            "query_payment",
            "refundable_money",
            {"currency": "CNY", "minor_units": 0, "unit_precision": 2},
            "available",
        ),
        ("query_payment", "refundable_money", None, "unknown"),
    ],
)
def test_revalidation_amount_permission_policy_changes(
    h, tool, field, value, availability
):
    change_field(h, tool, field, value, availability=availability)
    assert recheck(h).outcome == "needs_reassessment"


def test_revalidation_identity_policy_and_permission_each_block_submission(h):
    assert (
        recheck(h, access=h.access.model_copy(update={"subject_ref": "other"})).outcome
        == "needs_reassessment"
    )
    assert (
        recheck(
            h,
            current_selection=h.offer.selection.model_copy(
                update={"release_id": "new-release"}
            ),
        ).outcome
        == "needs_reassessment"
    )
    h.facts.authorize = lambda access, bound: observation(
        bound=bound, call_id="auth", availability="forbidden"
    )
    result = recheck(h)
    assert result.outcome == "needs_reassessment"
    assert "forbidden" in result.observations.serialized


def test_revalidation_capability_missing_and_transport_failure_are_distinct(h):
    h.facts.query_payment = lambda access, bound, call_id: observation(
        "query_refund_payment",
        bound=bound,
        call_id=call_id,
        availability="not_connected",
    )
    assert recheck(h).outcome == "unavailable"
    h.facts.fail = TimeoutError("provider timeout")
    with pytest.raises(RefundFlowError) as error:
        recheck(h)
    assert error.value.retryable and error.value.code == "refund_provider_failed"


def test_revalidation_never_calls_application_writer(h):
    from sqlalchemy import func, select

    from resolv.models.refund_application import RefundApplication

    result = recheck(h)
    assert result.outcome == "passed"
    with h.business() as s:
        assert s.scalar(select(func.count()).select_from(RefundApplication)) == 0


def test_all_rule_facts_are_frozen_even_when_model_omits_critical_list(h):
    from resolv.services.refund_offer import prepare_refund_offer

    record = h.record.model_copy(
        update={
            "decision": h.record.decision.model_copy(update={"critical_fields": ()})
        }
    )
    h.offer = prepare_refund_offer(record, now=NOW)
    assert any(
        f["field"] == "activated" for f in h.offer.critical_facts.value["fields"]
    )
    change_field(h, "query_order", "activated", True)
    assert recheck(h).outcome == "needs_reassessment"


def condition(
    operator, quote, expected, *, field="delivered_at", time="2026-10-15T15:55:00Z"
):
    obs = observation(
        "query_refund_delivery" if field == "delivered_at" else "query_refund_payment",
        call_id="time",
        fields={field: time},
    )
    c = RefundRecheckCondition(
        fact_ref={"observation_id": obs.observation_id, "field": field},
        policy_window_id="a" * 64,
        policy_quote=quote,
        operator=operator,
        expected=expected,
    )
    return c, obs


@pytest.mark.parametrize(
    "clock,zone,expected",
    [
        ("2026-10-16T15:59:59Z", "Asia/Shanghai", True),
        ("2026-10-16T16:00:00Z", "Asia/Shanghai", False),
        ("2026-10-16T16:00:00Z", "UTC", True),
    ],
)
def test_revalidation_clock_timezone_and_unspecified_boundary(clock, zone, expected):
    from resolv.services.refund_revalidation import evaluate_refund_conditions

    c, obs = condition(
        "calendar_days_lte",
        "签收后按业务时区日期差不超过1个自然日（签收当日为第0天）可申请。",
        1,
    )
    assert (
        evaluate_refund_conditions(
            (c,), (obs,), now=datetime.fromisoformat(clock), business_timezone=zone
        ).passed
        is expected
    )
    ambiguous = c.model_copy(
        update={"policy_quote": "签收后七天内可申请。", "expected": 7}
    )
    assert not evaluate_refund_conditions(
        (ambiguous,), (obs,), now=datetime.fromisoformat(clock), business_timezone=zone
    ).passed


@pytest.mark.parametrize(
    "clock,expected",
    [("2026-10-16T04:00:00Z", True), ("2026-10-16T04:00:00.000001Z", False)],
)
def test_elapsed_threshold_includes_exact_limit_and_rejects_later(clock, expected):
    from resolv.services.refund_revalidation import evaluate_refund_conditions

    c, obs = condition(
        "elapsed_seconds_lte",
        "支付后24小时内可申请。",
        86400,
        field="paid_at",
        time="2026-10-15T12:00:00+08:00",
    )
    assert (
        evaluate_refund_conditions(
            (c,),
            (obs,),
            now=datetime.fromisoformat(clock),
            business_timezone="Asia/Shanghai",
        ).passed
        is expected
    )


def test_revalidation_hash_rejects_tampering(h):
    from pydantic import ValidationError

    result = recheck(h)
    with pytest.raises(ValidationError):
        result.model_copy(update={"outcome": "needs_reassessment"})


def test_unchanged_time_fact_crosses_policy_limit_before_offer_expires(tmp_path):
    import json

    from langchain_core.messages import ToolMessage

    from resolv.schemas.refund_flow import RefundFactRef
    from resolv.services.hybrid_fingerprints import canonical_hash
    from resolv.services.refund_offer import prepare_refund_offer
    from resolv.services.refund_revalidation import revalidate_refund_offer

    with flow_case(tmp_path) as h:
        retrieve = h.flow.retrievers.retrieve

        def policy(*a, **kw):
            result = retrieve(*a, **kw)
            quote = "软件商品未激活且支付后24小时内可申请整单退款。"
            return result.model_copy(
                update={
                    "chunks": {
                        cid: body.model_copy(
                            update={
                                "chunk": body.chunk.model_copy(
                                    update={"answer": quote, "text": "退货\n" + quote}
                                )
                            }
                        )
                        for cid, body in result.chunks.items()
                    }
                }
            )

        h.flow.retrievers.retrieve = policy
        original = h.facts.query_payment

        def payment(access, bound, call_id):
            result = original(access, bound, call_id)
            fields = result.model_dump()["fields"] | {
                "paid_at": {
                    "availability": "available",
                    "value": "2026-10-15T04:00:05Z",
                }
            }
            return replace(result, fields=fields)

        h.facts.query_payment = payment
        decide = h.client.decide

        def final(messages, **kwargs):
            response = decide(messages, **kwargs)
            if response.decision is None:
                return response
            observed = next(
                json.loads(m.content)
                for m in messages
                if isinstance(m, ToolMessage) and m.name == "query_refund_payment"
            )
            ref = RefundFactRef(
                observation_id=observed["observation_id"], field="paid_at"
            )
            c = response.decision.policy_citations[0]
            condition = RefundRecheckCondition(
                fact_ref=ref,
                policy_window_id=c.window_id,
                policy_quote=c.quote,
                operator="elapsed_seconds_lte",
                expected=86400,
            )
            decision = response.decision.model_copy(
                update={
                    "critical_fields": (*response.decision.critical_fields, ref),
                    "recheck_conditions": (condition,),
                }
            )
            return response.model_copy(
                update={
                    "decision": decision,
                    "output_hash": canonical_hash(
                        {"tool_calls": [], "decision": decision.model_dump(mode="json")}
                    ),
                }
            )

        h.client.decide = final
        record = drive(h)
        offer = prepare_refund_offer(record, now=NOW)
        args = {
            "access": h.flow.identity.resolve(None),
            "adapter": h.facts,
            "registration": h.flow.registration,
            "current_selection": offer.selection,
            "profile": offer.profile,
        }
        assert (
            revalidate_refund_offer(
                offer, now=NOW + timedelta(seconds=4), **args
            ).outcome
            == "passed"
        )
        result = revalidate_refund_offer(offer, now=NOW + timedelta(seconds=6), **args)
        assert result.outcome == "needs_reassessment" and result.differences == (
            "condition_not_satisfied",
        )
        assert result.checked_at < offer.expires_at and h.counts["decide"] == 2

        # The same verified fact can also cross its policy deadline between
        # the fresh read and the actual SQLite commit, while the lease is valid.
        from resolv.schemas.refund_submission import RefundSubmitInput
        from resolv.services.refund_offer import finalize_refund_reply
        from resolv.services.refund_submission import RefundSubmissionService
        from tests.refund_submission_fixtures import LocalLocks
        from tests.test_refund_submission import counts

        h.service.refund_clock = lambda: NOW
        finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
        h.service.mark_completed(h.run.run_id, h.execution.epoch)
        clock = SimpleNamespace(now=NOW + timedelta(seconds=4))

        def cross_policy_limit(point):
            if point == "refund_before_business_commit":
                clock.now += timedelta(seconds=2)

        svc = RefundSubmissionService(
            h.factory,
            h.business,
            owner_id="local",
            identity=h.flow.identity,
            facts=h.facts,
            selection_reader=lambda: offer.selection,
            profile=offer.profile,
            clock=lambda: clock.now,
            execution_lock_factory=LocalLocks(),
            fault_hook=cross_policy_limit,
        )
        view = svc.submit(
            offer.conversation_id,
            offer.offer_id,
            RefundSubmitInput(request_id="cross-policy", confirm=True),
            {},
        )
        assert view.error_code == "refund_needs_reassessment"
        assert view.receipt is None and counts(h) == (0, 0)


def test_existing_application_requires_transactional_existing_result_not_new_write(h):
    from resolv.schemas.refund_submission import RefundRegistrationCommand
    from resolv.services.refund_registration import RefundRegistrationService

    command = RefundRegistrationCommand(
        command_id="existing",
        offer_id="old-offer",
        request_hash="a" * 64,
        binding=h.offer.binding,
        subject_ref=h.offer.subject_ref,
        money=h.offer.money,
        reason="原申请",
        decision_hash="d" * 64,
        evidence_hash="e" * 64,
        revalidation_hash="f" * 64,
        snapshots=FrozenJSON.from_value({}),
        created_at=NOW,
    )
    with h.business.begin() as s:
        RefundRegistrationService(s).register(command)
    result = recheck(h)
    assert result.outcome == "needs_reassessment" and result.differences == (
        "existing_application",
    )


def test_same_time_in_different_offset_is_the_same_critical_fact(h):
    from resolv.schemas.refund import RefundFieldValue
    from resolv.services.refund_revalidation import _comparison_value

    first = RefundFieldValue(
        availability="available",
        value="2026-10-16T12:00:00+08:00",
        original_time="original",
    )
    second = RefundFieldValue(
        availability="available",
        value="2026-10-16T04:00:00Z",
        original_time="changed formatting",
    )
    assert _comparison_value(first, "paid_at") == _comparison_value(second, "paid_at")
