"""Validate provenance and explicit comparisons; semantic judgement remains the model's."""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from resolv.schemas.knowledge_branch import KnowledgeEvidenceBundle
from resolv.schemas.refund import (
    OrderBinding,
    RefundFlowError,
    RefundMoney,
    RefundObservation,
)
from resolv.schemas.refund_flow import RefundDecision, RefundRequest
from resolv.services.hybrid_fingerprints import canonical_hash


def policy_window_text(evidence, window_id):
    window = next((w for w in evidence.windows if w.window_id == window_id), None)
    if window is None:
        raise RefundFlowError("refund_policy_reference_invalid", True)
    chunks = {c.chunk.chunk_id: c.chunk for c in evidence.chunks}
    return "\n".join(chunks[c].text for c in window.chunk_ids)


def _quote(evidence, window_id, quote):
    if not quote.strip() or quote not in policy_window_text(evidence, window_id):
        raise RefundFlowError("refund_policy_reference_invalid", True)


def _field(obs, ref):
    observation = obs.get(ref.observation_id)
    if observation is None or ref.field not in observation.fields:
        raise RefundFlowError("refund_fact_reference_invalid", True)
    return observation, observation.fields[ref.field]


def available_field(observation, name):
    field = observation.fields.get(name)
    if (
        field is None
        or field.availability != "available"
        or field.applicability != "applicable"
    ):
        raise RefundFlowError("refund_fact_unavailable", True)
    return field.value


def authoritative_money(observations):
    payment = next(
        (o for o in reversed(observations) if o.tool_name == "query_refund_payment"),
        None,
    )
    if payment is None or payment.availability != "available" or payment.error_code:
        raise RefundFlowError("refund_amount_unavailable", True)
    try:
        money = RefundMoney.model_validate(available_field(payment, "refundable_money"))
        if money.minor_units <= 0:
            raise ValueError("Positive refundable money required")
        return money
    except (ValidationError, ValueError, TypeError) as exc:
        raise RefundFlowError("refund_amount_unavailable", True) from exc


def _number(text):
    if text.isdecimal():
        return int(text)
    ones = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if text in ones:
        return ones[text]
    if "十" in text:
        left, right = text.split("十", 1)
        if (not left or left in ones) and (not right or right in ones):
            return ones.get(left, 1) * 10 + ones.get(right, 0)
    return None


def validate_recheck_condition(condition, *, evidence, observations):
    _quote(evidence, condition.policy_window_id, condition.policy_quote)
    obs = {o.observation_id: o for o in observations}
    observation, field = _field(obs, condition.fact_ref)
    if (
        observation.availability != "available"
        or observation.error_code
        or field.availability != "available"
        or field.applicability != "applicable"
    ):
        raise RefundFlowError("refund_condition_invalid", True)
    validate_condition_comparison(condition, field)


def validate_condition_comparison(condition, field):
    """Validate an already source-checked policy's literal comparison grammar."""
    quote = condition.policy_quote
    if condition.operator == "equals":
        # Only literal policy predicates have a stable code interpretation.
        expected = condition.expected
        if condition.fact_ref.field == "activated" and type(expected) is bool:
            phrase = "已激活" if expected else "未激活"
            supported = phrase in quote
        else:
            supported = type(expected) in {str, int} and str(expected) in quote
        if (
            not supported
            or (isinstance(expected, str) and not expected.strip())
            or type(field.value) is not type(expected)
        ):
            raise RefundFlowError("refund_condition_ambiguous", True)
        return
    if type(condition.expected) is not int or condition.expected < 0:
        raise RefundFlowError("refund_condition_ambiguous", True)
    time_fields = {
        "delivered_at": ("签收", "送达"),
        "paid_at": ("支付",),
        "ordered_at": ("下单",),
    }
    if not any(word in quote for word in time_fields.get(condition.fact_ref.field, ())):
        raise RefundFlowError("refund_condition_ambiguous", True)
    try:
        time = datetime.fromisoformat(field.value)
        if time.tzinfo is None or time.utcoffset() is None:
            raise ValueError("Aware timestamp required")
    except (ValueError, TypeError) as exc:
        raise RefundFlowError("refund_condition_invalid", True) from exc
    if not re.search(r"内|不超过|至多", quote):
        raise RefundFlowError("refund_condition_ambiguous", True)
    matches = re.findall(
        r"([0-9零一二两三四五六七八九十]+)\s*(?:个)?(自然日|日历日|小时|分钟|秒)", quote
    )
    thresholds = []
    for numeral, unit in matches:
        value = _number(numeral)
        if value is None:
            continue
        if (
            condition.operator == "calendar_days_lte"
            and unit in {"自然日", "日历日"}
            and re.search(r"日期差|当日为第[0零]天", quote)
        ):
            thresholds.append(value)
        if condition.operator == "elapsed_seconds_lte" and unit in {
            "小时",
            "分钟",
            "秒",
        }:
            thresholds.append(value * {"小时": 3600, "分钟": 60, "秒": 1}[unit])
    if thresholds != [condition.expected]:
        raise RefundFlowError("refund_condition_ambiguous", True)


def validate_refund_decision(
    decision, *, request, binding, evidence, observations, profile
):
    decision = RefundDecision.model_validate_json(decision.model_dump_json())
    request = RefundRequest.model_validate_json(request.model_dump_json())
    binding = OrderBinding.model_validate_json(binding.model_dump_json())
    evidence = KnowledgeEvidenceBundle.model_validate_json(evidence.model_dump_json())
    observations = tuple(
        RefundObservation.model_validate_json(o.model_dump_json()) for o in observations
    )
    ZoneInfo(profile.business_timezone)
    if (
        decision.operation != request.operation
        or decision.order_binding_hash != binding.binding_hash
        or decision.evidence_hash != evidence.evidence_hash
        or binding.order_source != profile.adapter.namespace
    ):
        raise RefundFlowError("refund_decision_binding_invalid", True)
    obs = {o.observation_id: o for o in observations}
    if len(obs) != len(observations) or len(set(decision.observation_refs)) != len(
        decision.observation_refs
    ):
        raise RefundFlowError("refund_fact_reference_invalid", True)
    if any(
        o.binding_hash != binding.binding_hash or o.source != binding.order_source
        for o in observations
    ):
        raise RefundFlowError("refund_decision_binding_invalid", True)
    if any(ref not in obs for ref in decision.observation_refs):
        raise RefundFlowError("refund_fact_reference_invalid", True)
    selected = [
        o for o in observations if o.observation_id in decision.observation_refs
    ]
    latest = {o.tool_name: o for o in observations}
    required = {"query_refund_order", "query_refund_status"}
    if decision.verdict == "eligible":
        required.add("query_refund_payment")
    if any(
        name not in latest
        or latest[name].observation_id not in decision.observation_refs
        for name in required
    ):
        raise RefundFlowError("refund_required_observation_missing", True)
    if any(o.error_code for o in selected):
        raise RefundFlowError("refund_provider_failed", True)
    order = next(
        (o for o in reversed(selected) if o.tool_name == "query_refund_order"), None
    )
    status = next(
        (o for o in reversed(selected) if o.tool_name == "query_refund_status"), None
    )
    if (
        order is None
        or status is None
        or order.availability != "available"
        or status.availability != "available"
    ):
        raise RefundFlowError("refund_required_observation_missing", True)
    if (
        available_field(order, "authorized") is not True
        or available_field(order, "order_id") != binding.order_id
    ):
        raise RefundFlowError("refund_authorization_conflict", True)
    application = available_field(status, "local_application")
    if application is not None and (
        decision.verdict in {"eligible", "ineligible"}
        or decision.next_action not in {"show_status", "human_draft"}
    ):
        raise RefundFlowError("refund_existing_application", True)
    citations = {
        (c.citation_number, c.window_id, c.quote) for c in decision.policy_citations
    }
    for citation in decision.policy_citations:
        actual = next(
            (
                c
                for c in evidence.citations
                if c.citation_number == citation.citation_number
            ),
            None,
        )
        if actual is None or actual.window_id != citation.window_id:
            raise RefundFlowError("refund_policy_reference_invalid", True)
        _quote(evidence, citation.window_id, citation.quote)
    facts = list(decision.critical_fields)
    for rule in decision.rules:
        for citation in rule.policy_citations:
            if (
                citation.citation_number,
                citation.window_id,
                citation.quote,
            ) not in citations:
                raise RefundFlowError("refund_policy_reference_invalid", True)
        facts.extend(rule.fact_refs)
    for ref in facts:
        observation, field = _field(obs, ref)
        if observation.observation_id not in decision.observation_refs:
            raise RefundFlowError("refund_fact_reference_invalid", True)
        newest = latest[observation.tool_name].fields.get(ref.field)
        if decision.verdict in {"eligible", "ineligible"} and (
            newest is None
            or canonical_hash(newest.model_dump(mode="json", exclude={"original_time"}))
            != canonical_hash(field.model_dump(mode="json", exclude={"original_time"}))
        ):
            raise RefundFlowError("refund_fact_conflict", True)
        if decision.verdict in {"eligible", "ineligible"} and (
            observation.availability != "available"
            or field.availability != "available"
            or field.applicability != "applicable"
            or (
                field.value is None
                and not (
                    observation.tool_name == "query_refund_status"
                    and ref.field == "local_application"
                )
            )
        ):
            raise RefundFlowError("refund_fact_unavailable", True)
    if decision.verdict in {"eligible", "ineligible"} and not evidence.windows:
        raise RefundFlowError("refund_policy_reference_invalid", True)
    if decision.verdict == "eligible":
        authoritative_money(selected)
        subject = available_field(order, "subject_ref")
        if not isinstance(subject, str) or not subject.strip():
            raise RefundFlowError("refund_identity_unavailable", True)
    critical = {(f.observation_id, f.field) for f in decision.critical_fields}
    for condition in decision.recheck_conditions:
        if (
            condition.fact_ref.observation_id,
            condition.fact_ref.field,
        ) not in critical:
            raise RefundFlowError("refund_condition_invalid", True)
        validate_recheck_condition(
            condition, evidence=evidence, observations=observations
        )
    return decision
