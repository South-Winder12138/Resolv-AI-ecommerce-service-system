"""Freeze an independent offer with the final assistant reply in one transaction."""

from datetime import timedelta

from resolv.db.conversation_base import utc_now
from resolv.models.conversation import ConversationMessage
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.schemas.knowledge_branch import FrozenJSON
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_submission import RefundOfferRecord
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.refund_decision import (
    authoritative_money,
    available_field,
    validate_refund_decision,
)


def prepare_refund_offer(record, *, now):
    decision = record.decision
    if (
        decision is None
        or decision.verdict != "eligible"
        or decision.next_action != "show_offer"
        or record.request is None
        or record.request.reason is None
        or not record.request.reason.strip()
        or len(record.request.reason) > 500
        or not record.profile.adapter.connected
        or record.profile.selection.status != "configured"
    ):
        return None
    validate_refund_decision(
        decision,
        request=record.request,
        binding=record.binding,
        evidence=record.evidence,
        observations=record.observations,
        profile=record.profile,
    )
    order = next(
        o for o in reversed(record.observations) if o.tool_name == "query_refund_order"
    )
    subject = available_field(order, "subject_ref")
    money = authoritative_money(record.observations)
    selected = {}
    refs = [
        *decision.critical_fields,
        *(ref for rule in decision.rules for ref in rule.fact_refs),
        *(condition.fact_ref for condition in decision.recheck_conditions),
    ]
    for ref in refs:
        obs = next(
            o for o in record.observations if o.observation_id == ref.observation_id
        )
        selected[(obs.tool_name, ref.field, obs.observation_id)] = (
            obs,
            obs.fields[ref.field],
        )
    mandatory = {
        "query_refund_order": ("order_id", "status", "product_type"),
        "query_refund_payment": ("refundable_money", "paid_money"),
    }
    for tool, fields in mandatory.items():
        obs = next(
            (o for o in reversed(record.observations) if o.tool_name == tool), None
        )
        if obs:
            for field in fields:
                if field in obs.fields:
                    selected[(tool, field, obs.observation_id)] = (
                        obs,
                        obs.fields[field],
                    )
    snapshot = {
        "binding_hash": record.binding.binding_hash,
        "subject_ref": subject,
        "fields": [
            {
                "tool_name": tool,
                "field": field,
                "observation_id": obs.observation_id,
                "fact": value.model_dump(mode="json"),
            }
            for (tool, field, _observation_id), (obs, value) in sorted(selected.items())
        ],
    }
    offer_id = canonical_hash(
        {
            "rule": "refund-offer-v1",
            "run_id": record.run_id,
            "decision_hash": decision.fingerprint,
        }
    )[:32]
    return RefundOfferRecord(
        offer_id=offer_id,
        owner_id=record.owner_id,
        conversation_id=record.conversation_id,
        run_id=record.run_id,
        primary_task_id=record.primary_task_id,
        binding=record.binding,
        subject_ref=subject,
        money=money,
        reason=record.request.reason,
        selection=record.profile.selection,
        evidence_hash=record.evidence.evidence_hash,
        decision_hash=decision.fingerprint,
        critical_facts=FrozenJSON.from_value(snapshot),
        conditions=decision.recheck_conditions,
        profile=record.profile,
        created_at=now,
        expires_at=now + timedelta(seconds=record.profile.offer_ttl_seconds),
    )


def finalize_refund_reply(service, run_id, execution, record):
    from resolv.services.refund_reply_projection import build_refund_reply

    execution.assert_owned()
    try:
        with service.session_factory.begin() as s:
            repo = RefundFlowRepository(s, service.owner_id)
            stored = repo.get(run_id)
            run = ConversationRepository(s, service.owner_id).owned(
                run_id, execution.epoch
            )
            if (
                stored is None
                or stored != record
                or stored.run_id != run_id
                or run.orchestration_version != "routed-v4"
            ):
                raise RefundFlowError("refund_reply_conflict")
            offer = repo.offer_for_run(run_id)
            if offer is None:
                offer = prepare_refund_offer(
                    record, now=getattr(service, "refund_clock", utc_now)()
                )
                if offer is not None:
                    offer = repo.put_offer(run_id, execution.epoch, offer)
            reply = build_refund_reply(record, offer)
            ConversationRepository(s, service.owner_id).commit_answer(
                run_id, execution.epoch, reply.answer, metadata=reply.metadata
            )
            service.fault("refund_before_finalize_commit")
        service.fault("refund_after_finalize_commit")
        return reply
    except Exception as exc:
        execution.assert_owned()
        try:
            with service.session_factory() as s:
                repo = RefundFlowRepository(s, service.owner_id)
                saved = repo.get(run_id)
                offer = repo.offer_for_run(run_id)
                row = ConversationRepository(s, service.owner_id).run(run_id)
                answer = s.get(ConversationMessage, row.answer_message_id)
                if saved == record:
                    expected = build_refund_reply(record, offer)
                    if (
                        answer is not None
                        and answer.text == expected.answer
                        and row.result_json == expected.metadata.model_dump(mode="json")
                    ):
                        return expected
        except Exception as store:
            raise RefundFlowError("refund_store_failed", True, run_id) from store
        if isinstance(exc, RefundFlowError):
            raise
        raise RefundFlowError("refund_store_failed", True, run_id) from exc
