"""Fresh bounded reads and explicit policy comparisons; never invoke a model or writer."""

from datetime import timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import TypeAdapter, ValidationError

from resolv.schemas.knowledge_branch import FrozenJSON, canonical_json
from resolv.schemas.knowledge_release import ReleaseSelection
from resolv.schemas.refund import (
    AdapterDescriptor,
    RefundAccessContext,
    RefundFlowError,
    RefundMoney,
    RefundObservation,
    RefundTimestamp,
)
from resolv.schemas.refund_flow import RefundFactRef, RefundProfile, RefundToolCall
from resolv.schemas.refund_submission import (
    RefundConditionResult,
    RefundCriticalSnapshot,
    RefundOfferRecord,
    RefundRevalidation,
)
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.refund_decision import (
    authoritative_money,
    validate_condition_comparison,
)
from resolv.services.refund_tools import execute_refund_read

CLOCK = TypeAdapter(RefundTimestamp)


def evaluate_refund_conditions(conditions, observations, *, now, business_timezone):
    now = CLOCK.validate_python(now)
    try:
        zone = ZoneInfo(business_timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise RefundFlowError("refund_contract_invalid") from exc
    observed = {o.observation_id: o for o in observations}
    reasons = []
    for condition in conditions:
        observation = observed.get(condition.fact_ref.observation_id)
        field = (
            observation.fields.get(condition.fact_ref.field) if observation else None
        )
        if (
            observation is None
            or observation.availability != "available"
            or observation.error_code
            or field is None
            or field.availability != "available"
            or field.applicability != "applicable"
        ):
            reasons.append("condition_fact_unavailable")
            continue
        try:
            validate_condition_comparison(condition, field)
        except RefundFlowError:
            reasons.append("condition_comparison_undefined")
            continue
        if condition.operator == "equals":
            passed = (
                type(field.value) is type(condition.expected)
                and field.value == condition.expected
            )
        else:
            try:
                event = CLOCK.validate_python(field.value)
            except (ValidationError, ValueError, TypeError) as exc:
                raise RefundFlowError("refund_provider_failed", True) from exc
            if event > now:
                passed = False
            elif condition.operator == "calendar_days_lte":
                passed = (
                    now.astimezone(zone).date() - event.astimezone(zone).date()
                ).days <= condition.expected
            else:
                try:
                    passed = now - event <= timedelta(seconds=condition.expected)
                except OverflowError:
                    passed = False
        if not passed:
            reasons.append("condition_not_satisfied")
    return RefundConditionResult(
        passed=not reasons, reasons=tuple(dict.fromkeys(reasons))
    )


def _comparison_value(field, name):
    value = field.value
    if (
        field.availability == "available"
        and field.applicability == "applicable"
        and value is not None
    ):
        try:
            if name in {"paid_money", "refundable_money"}:
                value = RefundMoney.model_validate(value).model_dump(mode="json")
            elif name.endswith("_at"):
                value = CLOCK.dump_python(CLOCK.validate_python(value), mode="json")
        except (ValidationError, ValueError, TypeError) as exc:
            raise RefundFlowError("refund_provider_failed", True) from exc
    return {
        "availability": field.availability,
        "applicability": field.applicability,
        "value": value,
    }


def _offer_conditions(offer, observations):
    snapshot = RefundCriticalSnapshot.model_validate_json(
        offer.critical_facts.serialized
    )
    by_tool = {o.tool_name: o for o in observations}
    remapped = []
    for condition in offer.conditions:
        source = next(
            (
                f
                for f in snapshot.fields
                if f.observation_id == condition.fact_ref.observation_id
                and f.field == condition.fact_ref.field
            ),
            None,
        )
        if source is None or source.tool_name not in by_tool:
            raise RefundFlowError("refund_offer_conflict")
        remapped.append(
            condition.model_copy(
                update={
                    "fact_ref": RefundFactRef(
                        observation_id=by_tool[source.tool_name].observation_id,
                        field=source.field,
                    )
                }
            )
        )
    return tuple(remapped)


def check_refund_commit_time(offer, revalidation, *, now, creating_application):
    """Local checks at the commit boundary; no provider or other-store reads."""
    now = CLOCK.validate_python(now)
    if now < offer.created_at or now >= offer.expires_at:
        raise RefundFlowError("refund_offer_expired")
    if creating_application:
        observations = tuple(
            RefundObservation.model_validate_json(canonical_json(value))
            for value in revalidation.observations.value["reads"]
        )
        conditions = evaluate_refund_conditions(
            _offer_conditions(offer, observations),
            observations,
            now=now,
            business_timezone=offer.profile.business_timezone,
        )
        if not conditions.passed:
            raise RefundFlowError("refund_needs_reassessment")


def revalidate_refund_offer(
    offer, *, access, adapter, registration, current_selection, now, profile
):
    offer = RefundOfferRecord.model_validate_json(offer.model_dump_json())
    profile = RefundProfile.model_validate_json(profile.model_dump_json())
    now = CLOCK.validate_python(now)
    selection = ReleaseSelection.model_validate_json(
        current_selection.model_dump_json()
    )
    descriptor = AdapterDescriptor.model_validate_json(
        adapter.descriptor.model_dump_json()
    )
    snapshot = RefundCriticalSnapshot.model_validate_json(
        offer.critical_facts.serialized
    )
    if (
        snapshot.binding_hash != offer.binding.binding_hash
        or snapshot.subject_ref != offer.subject_ref
    ):
        raise RefundFlowError("refund_offer_conflict")
    observations = []
    payload = {
        "offer_id": offer.offer_id,
        "binding_hash": offer.binding.binding_hash,
        "selection": selection.model_dump(mode="json"),
        "adapter": descriptor.model_dump(mode="json"),
        "reads": [],
    }

    def result(outcome, differences=(), conditions=None):
        payload["reads"] = [o.model_dump(mode="json") for o in observations]
        if conditions is not None:
            payload["conditions"] = conditions.model_dump(mode="json")
        return RefundRevalidation.create(
            outcome=outcome,
            observations=payload,
            differences=tuple(dict.fromkeys(differences)),
            checked_at=now,
        )

    if access is None or not descriptor.connected:
        return result("unavailable", ("identity_or_provider_not_connected",))
    access = RefundAccessContext.model_validate_json(access.model_dump_json())
    if (
        descriptor != offer.profile.adapter
        or profile.adapter != offer.profile.adapter
        or access.namespace != offer.binding.order_source
        or access.authorization_version != offer.profile.adapter.authorization_version
    ):
        return result("unavailable", ("adapter_contract_changed",))
    if access.subject_ref != offer.subject_ref:
        return result("needs_reassessment", ("subject_changed",))
    if now < offer.created_at or now >= offer.expires_at:
        return result("needs_reassessment", ("offer_expired",))
    if selection.status == "invalid":
        raise RefundFlowError("refund_provider_failed", True)
    if (
        selection.fingerprint != offer.selection.fingerprint
        or profile.business_timezone != offer.profile.business_timezone
    ):
        return result("needs_reassessment", ("policy_or_timezone_changed",))
    if offer.status == "needs_reassessment":
        return result("needs_reassessment", ("offer_invalidated",))
    names = ["query_refund_order", "query_refund_status", "query_refund_payment"]
    names.extend(sorted({f.tool_name for f in snapshot.fields} - set(names)))
    by_tool = {}
    for name in names:
        call = RefundToolCall(
            id="recheck-" + uuid4().hex, name=name, args=FrozenJSON.from_value({})
        )
        try:
            obs = execute_refund_read(
                call,
                binding=offer.binding,
                access=access,
                adapter=adapter,
                registration=registration,
                observations=(),
            )
        except Exception as exc:
            if isinstance(exc, RefundFlowError) and exc.code in {
                "refund_store_failed",
                "refund_store_inconsistent",
            }:
                raise
            raise RefundFlowError("refund_provider_failed", True) from exc
        obs = RefundObservation.model_validate_json(obs.model_dump_json())
        observations.append(obs)
        by_tool[name] = obs
        if obs.error_code:
            raise RefundFlowError("refund_provider_failed", True)
        if obs.availability in {"not_connected", "not_supported"}:
            return result("unavailable", ("provider_capability_unavailable",))
        if obs.availability != "available":
            return result(
                "needs_reassessment", ("order_access_or_information_changed",)
            )
        for field_name, value in obs.fields.items():
            _comparison_value(value, field_name)
    status = by_tool["query_refund_status"].fields.get("local_application")
    if status is None or status.availability != "available":
        return result("needs_reassessment", ("application_information_unknown",))
    if status.value is not None:
        return result("needs_reassessment", ("existing_application",))
    order = by_tool["query_refund_order"]
    oid = order.fields.get("order_id")
    if (
        oid is None
        or oid.availability != "available"
        or oid.value != offer.binding.order_id
    ):
        return result("needs_reassessment", ("order_identity_changed",))
    subject = order.fields.get("subject_ref")
    if subject is None or subject.value != offer.subject_ref:
        return result("needs_reassessment", ("subject_changed",))
    differences = []
    for frozen in snapshot.fields:
        current = by_tool[frozen.tool_name].fields.get(frozen.field)
        if current is None or canonical_hash(
            _comparison_value(current, frozen.field)
        ) != canonical_hash(_comparison_value(frozen.fact, frozen.field)):
            differences.append("fact." + frozen.tool_name + "." + frozen.field)
    try:
        money = authoritative_money(tuple(observations))
    except RefundFlowError:
        money = None
    if money != offer.money:
        differences.append("refundable_money")
    if differences:
        return result("needs_reassessment", differences)
    conditions = evaluate_refund_conditions(
        _offer_conditions(offer, observations),
        tuple(observations),
        now=now,
        business_timezone=offer.profile.business_timezone,
    )
    return result(
        "passed" if conditions.passed else "needs_reassessment",
        conditions.reasons,
        conditions,
    )
