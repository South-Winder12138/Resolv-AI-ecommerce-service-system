"""Public immutable judgement and independently refreshed offer/command views."""

from resolv.schemas.branch_execution import PrimaryExecution, parse_reply_metadata
from resolv.schemas.conversation import ConversationError
from resolv.schemas.knowledge_branch import (
    FrozenActionDraft,
    FrozenSuggestedAction,
    canonical_json,
)
from resolv.schemas.refund import RefundApplicationRecord, RefundFlowError
from resolv.schemas.refund_flow import RefundRunRecord
from resolv.schemas.refund_submission import (
    BranchReplyV3,
    RefundOfferView,
    RefundPublicResult,
    RefundReceipt,
    RefundSubmissionView,
    ReplyMetadataV3,
)
from resolv.services.branch_replies import draft_content
from resolv.services.hybrid_fingerprints import canonical_hash

REASON_REQUIRED = (
    "请补充或精简退款原因（最多500个字符），并完整描述退款诉求后重新判断。"
)


def _application(record):
    for observation in reversed(record.observations):
        if observation.tool_name != "query_refund_status":
            continue
        field = observation.fields.get("local_application")
        if field is None or field.availability != "available" or field.value is None:
            return None
        value = RefundApplicationRecord.model_validate_json(canonical_json(field.value))
        if record.binding is None or (value.order_source, value.order_id) != (
            record.binding.order_source,
            record.binding.order_id,
        ):
            raise RefundFlowError("refund_reply_conflict")
        return RefundReceipt(
            application_id=value.application_id,
            order_source=value.order_source,
            order_id=value.order_id,
            status=value.status,
            created=False,
            money=value.money.public() if value.money else None,
            reason=value.reason,
            created_at=value.created_at,
        )
    return None


def build_refund_reply(record, offer):
    record = RefundRunRecord.model_validate_json(record.model_dump_json())
    if record.decision is None and record.scope_reply is None and not record.step_limit:
        raise RefundFlowError("refund_stage_predecessor_missing")
    decision = record.decision
    application = _application(record)
    kind = "answer"
    next_action = decision.next_action if decision else "none"
    missing = decision.missing_information if decision else ()
    reason = (
        decision.reason
        if decision
        else record.scope_reply.reason
        if record.scope_reply
        else "本轮暂未完成可靠判断，您可以补充完整诉求或整理人工处理草稿。"
    )
    verdict = (
        decision.verdict if decision else "undetermined" if record.step_limit else None
    )
    if record.scope_reply:
        kind = (
            "clarification"
            if record.scope_reply.kind == "clarification"
            else "unavailable"
            if record.scope_reply.kind == "unavailable"
            else "answer"
        )
        next_action = (
            "human_draft"
            if record.scope_reply.kind in {"human_draft", "unavailable"}
            else "request_information"
        )
    if record.step_limit:
        kind, next_action = "step_limit", "human_draft"
    answer = reason
    if decision and decision.next_action == "show_offer" and offer is None:
        kind, next_action = "clarification", "request_information"
        missing = (*missing, "退款原因（最多500个字符）")
        answer += "\n" + REASON_REQUIRED
    if offer is not None:
        if (
            decision is None
            or offer.run_id != record.run_id
            or offer.owner_id != record.owner_id
            or offer.conversation_id != record.conversation_id
            or offer.decision_hash != decision.fingerprint
            or offer.binding != record.binding
            or offer.evidence_hash != record.evidence.evidence_hash
        ):
            raise RefundFlowError("refund_reply_conflict")
        answer += (
            "\n您可以通过独立入口提交整单退款申请。提交后为等待审核，并非退款成功。"
        )
    if application:
        answer += "\n以上仅为本地申请记录，不代表资金退款或到账；外部进度需另行核验。"
    profile = record.profile
    selection = (
        profile.selection
        if record.retrieval and profile.selection.status == "configured"
        else None
    )
    public = RefundPublicResult(
        operation=record.request.operation if record.request else None,
        verdict=verdict,
        order_id=record.binding.order_id if record.binding else None,
        reason=reason,
        missing_information=missing,
        unverified_conditions=decision.unverified_conditions if decision else (),
        next_action=next_action,
        release_id=selection.release_id if selection else None,
        index_version=selection.index_version if selection else None,
        evidence=record.evidence,
        policy_citations=decision.policy_citations if decision else (),
        offer_id=offer.offer_id if offer else None,
        application=application,
    )
    actions = ()
    if next_action == "human_draft":
        primary = PrimaryExecution.model_validate_json(record.primary_json.serialized)
        actions = (
            FrozenSuggestedAction(
                suggestion_id=canonical_hash(
                    {"run": record.run_id, "draft": "refund-human"}
                ),
                type="transfer_human",
                label="人工处理草稿",
                draft=FrozenActionDraft(
                    title="退款或售后处理说明", content=draft_content(primary.primary)
                ),
            ),
        )
    return BranchReplyV3(
        answer=answer,
        metadata=ReplyMetadataV3(
            reply_kind=kind,
            understanding_id=record.understanding_id,
            primary_task_id=record.primary_task_id,
            branch="refund_flow",
            suggested_actions=actions,
            refund=public,
        ),
    )


def public_refund_reply(run, record, offer):
    try:
        if (
            run.orchestration_version != "routed-v4"
            or run.status != "completed"
            or run.result_json is None
        ):
            raise ValueError("Reply is not completed")
        meta = parse_reply_metadata(run.result_json)
        if (
            not isinstance(meta, ReplyMetadataV3)
            or meta.understanding_id != run.understanding_id
        ):
            raise ValueError("Reply identity mismatch")
        if meta.branch != "refund_flow" or run.mode == "chat":
            if meta.refund is not None:
                raise ValueError("Wrong refund branch")
            return meta.model_dump(mode="json")
        if record is None or (
            record.run_id,
            record.conversation_id,
            record.understanding_id,
        ) != (run.run_id, run.conversation_id, run.understanding_id):
            raise ValueError("Refund record missing")
        if canonical_hash(run.refund_config) != record.profile.fingerprint:
            raise ValueError("Refund profile mismatch")
        expected = build_refund_reply(record, offer).metadata
        if meta != expected:
            raise ValueError("Immutable public judgement mismatch")
        return meta.model_dump(mode="json")
    except (ValueError, TypeError, KeyError, AttributeError, RefundFlowError) as exc:
        raise ConversationError("run_state_conflict", run_id=run.run_id) from exc


def public_refund_offer(offer, submission, receipt, *, now):
    known = receipt or (submission.receipt if submission else None)
    if offer.status == "submitted" and known is None:
        raise RefundFlowError("refund_store_inconsistent", True)
    if submission is not None and (
        submission.offer_id != offer.offer_id
        or submission.owner_id != offer.owner_id
        or submission.conversation_id != offer.conversation_id
    ):
        raise RefundFlowError("refund_not_found")
    if known is not None:
        if (known.order_source, known.order_id) != (
            offer.binding.order_source,
            offer.binding.order_id,
        ):
            raise RefundFlowError("refund_receipt_conflict")
        if submission is None or (
            submission.receipt is not None and submission.receipt != known
        ):
            raise RefundFlowError("refund_receipt_conflict")
    view = (
        RefundSubmissionView(
            submission_id=submission.submission_id,
            status=submission.status,
            receipt=known,
            error_code=submission.error_code,
            retryable=submission.retryable,
        )
        if submission
        else None
    )
    validity = (
        "submitted"
        if known is not None or offer.status == "submitted"
        else "needs_reassessment"
        if offer.status == "needs_reassessment"
        else "expired"
        if now >= offer.expires_at
        else "available"
    )
    if validity == "available" and not offer.profile.adapter.connected:
        validity = "unavailable"
    return RefundOfferView(
        offer_id=offer.offer_id,
        order_id=offer.binding.order_id,
        money=offer.money.public(),
        reason=offer.reason,
        created_at=offer.created_at,
        expires_at=offer.expires_at,
        validity=validity,
        submission=view,
    )
