"""Narrow refund operations and sourced orders, after upstream understanding."""

import re

from resolv.repositories.understanding import source
from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund import OrderBinding, RefundFlowError
from resolv.schemas.refund_flow import (
    RefundRequest,
    RefundScopeReply,
    RefundScopeResult,
)
from resolv.schemas.understanding import EvidenceRef, UnderstandingError
from resolv.services.business_policy import build_business_policy
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.reference_resolution import contains_entity
from resolv.services.understanding_context import json_value_at, snapshot_digest


def _reply(kind, reason):
    return RefundScopeReply(kind=kind, reason=reason)


def _verify(ref, snapshot, source_loader):
    fingerprints = {
        (s.kind, s.source_id): s.source_hash for s in snapshot.source_fingerprints
    }
    key = ref.kind, ref.source_id
    original = next(
        (s for s in snapshot.model_sources if (s.kind, s.source_id) == key), None
    )
    if ref.kind in {"message", "action"}:
        original = source_loader(ref) if source_loader else None
    if original is None:
        raise RefundFlowError("refund_source_unavailable")
    rebuilt = source(
        original.kind,
        original.source_id,
        text=original.text,
        json_value=original.json_value,
        observed_at=original.observed_at,
    )
    if (
        key != (original.kind, original.source_id)
        or key not in fingerprints
        or rebuilt.source_hash != original.source_hash
        or fingerprints[key] != original.source_hash
    ):
        raise RefundFlowError("refund_source_conflict")
    if original.text is not None:
        if (
            ref.span is None
            or ref.json_path
            or ref.span.end > len(original.text)
            or original.text[ref.span.start : ref.span.end] != ref.quote
        ):
            raise RefundFlowError("refund_source_conflict")
    else:
        if ref.span is not None or not ref.json_path:
            raise RefundFlowError("refund_source_conflict")
        try:
            value = json_value_at(original.json_value, ref.json_path)
        except UnderstandingError as exc:
            raise RefundFlowError("refund_source_conflict") from exc
        if ref.quote != (value if isinstance(value, str) else canonical_json(value)):
            raise RefundFlowError("refund_source_conflict")
    return original


def _validate(primary, snapshot, source_loader):
    primary = PrimaryExecution.model_validate_json(primary.model_dump_json())
    if (
        primary.conversation_id != snapshot.conversation_id
        or primary.snapshot_hash != snapshot.snapshot_hash
        or snapshot.snapshot_hash != snapshot_digest(snapshot)
        or primary.primary.branch != "refund_flow"
    ):
        raise RefundFlowError("refund_identity_conflict")
    for ref in primary.primary.evidence_refs:
        _verify(ref, snapshot, source_loader)
    return primary


def _ids(primary, text):
    p = primary.model_copy(
        update={
            "primary": primary.primary.model_copy(update={"text": text, "bindings": []})
        }
    )
    return build_business_policy(p).allowed_order_ids


def _original_proof(primary, snapshot, value, source_loader, namespace):
    if source_loader is None:
        return None
    for fingerprint in snapshot.source_fingerprints:
        if fingerprint.kind not in {"message", "action"}:
            continue
        hint = EvidenceRef(
            kind=fingerprint.kind,
            source_id=fingerprint.source_id,
            span=None,
            json_path=[],
            quote=value,
        )
        original = source_loader(hint)
        if original is None:
            continue
        rebuilt = source(
            original.kind,
            original.source_id,
            text=original.text,
            json_value=original.json_value,
            observed_at=original.observed_at,
        )
        if (
            rebuilt.source_hash != fingerprint.source_hash
            or original.source_hash != fingerprint.source_hash
        ):
            raise RefundFlowError("refund_source_conflict")
        if original.text and value in _ids(primary, original.text):
            return EvidenceRef(
                kind=original.kind,
                source_id=original.source_id,
                span={"start": 0, "end": len(original.text)},
                json_path=[],
                quote=original.text,
            )
        if (
            isinstance(original.json_value, dict)
            and original.json_value.get("order_id") == value
        ):
            if original.json_value.get("order_source") != namespace:
                raise RefundFlowError("refund_source_conflict")
            return EvidenceRef(
                kind=original.kind,
                source_id=original.source_id,
                span=None,
                json_path=["order_id"],
                quote=value,
            )
    return None


def bind_refund_order(primary, snapshot, *, order_source, source_loader=None):
    primary = _validate(primary, snapshot, source_loader)
    if not order_source or not order_source.strip():
        raise RefundFlowError("refund_adapter_conflict")
    candidates = _ids(primary, primary.primary.text)
    refs = list(primary.primary.evidence_refs)
    proven = set()
    for ref in refs:
        original = _verify(ref, snapshot, source_loader)
        if ref.kind == "summary":
            continue
        if ref.kind == "action":
            if (
                not isinstance(original.json_value, dict)
                or original.json_value.get("order_source") != order_source
            ):
                raise RefundFlowError("refund_source_conflict")
            if ref.json_path not in (["order_id"], ["order_no"]):
                continue
            proven.update(value for value in candidates if value == ref.quote)
        elif original.text is not None:
            proven.update(
                value for value in candidates if value in _ids(primary, ref.quote)
            )
    for binding in primary.primary.bindings:
        if binding.entity_type != "order" or (
            candidates and binding.value not in candidates
        ):
            continue
        for ref in binding.evidence_refs:
            value = _verify(ref, snapshot, source_loader)
            if ref.kind == "summary":
                proof = _original_proof(
                    primary, snapshot, binding.value, source_loader, order_source
                )
                if proof is None:
                    continue
                refs.append(proof)
                proven.add(binding.value)
            elif contains_entity(ref.quote, binding.value):
                if (
                    isinstance(value.json_value, dict)
                    and value.json_value.get("order_source") != order_source
                ):
                    raise RefundFlowError("refund_source_conflict")
                if value.text is not None and binding.value not in _ids(
                    primary, value.text
                ):
                    continue
                if (
                    value.json_value is not None
                    and ref.kind == "action"
                    and ref.json_path not in (["order_id"], ["order_no"])
                ):
                    continue
                refs.append(ref)
                proven.add(binding.value)
    if len(proven) != 1:
        return _reply(
            "clarification", "请明确需要处理的唯一订单号，并完整描述退款诉求。"
        )
    value = next(iter(proven))
    return OrderBinding.create(
        order_source=order_source,
        order_id=value,
        understanding_id=primary.understanding_id,
        primary_task_id=primary.primary.task_id,
        primary_hash=canonical_hash(primary.model_dump(mode="json")),
        source_refs=tuple({r.model_dump_json(): r for r in refs}.values()),
    )


def resolve_refund_scope(primary, snapshot, *, source_loader=None):
    primary = _validate(primary, snapshot, source_loader)
    text = primary.primary.text
    after_sales = re.search(r"维修|换货|保修", text)
    refund = re.search(r"退款|退货", text)
    if after_sales:
        kind = (
            "clarification"
            if refund and not re.search(r"不(?:要|用)?退款|不退货", text)
            else "human_draft"
        )
        return RefundScopeResult(
            scope_reply=_reply(
                kind,
                "请明确退款或其他售后诉求。"
                if kind == "clarification"
                else "此售后诉求需人工处理，可提供处理草稿。",
            )
        )
    if re.search(
        r"部分退款|只退.{0,12}(?:元|件)|退.{0,5}一件|差价|(?:撤回|取消|撤销|修改).*退款|退款.*(?:撤回|取消|撤销)|更改.*退款原因|重新申请|再次申请",
        text,
    ):
        return RefundScopeResult(
            scope_reply=_reply(
                "human_draft",
                "当前仅支持整单退款资格、申请和进度查询，此诉求需人工处理。",
            )
        )
    if re.search(
        r"退款(?:申请)?(?:的)?(?:进度|状态|结果)|申请.{0,8}(?:审核|通过)|退到哪|到账了吗",
        text,
    ):
        operation = "status"
    elif refund and re.search(
        r"能否|能不能|是否|可以|可不可以|支持|资格|符合|吗|[？?]", text
    ):
        operation = "eligibility"
    elif (
        refund
        and not re.search(
            r"(?:不想|不需要|不要|不用|别).{0,6}(?:申请|办理|提交|退款)", text
        )
        and re.search(r"申请|帮我|我要|我想|请.{0,12}退款|整单退款", text)
    ):
        operation = "apply"
    else:
        return RefundScopeResult(
            scope_reply=_reply(
                "clarification",
                "请明确是咨询整单退款资格、申请退款，还是查询已有申请进度。",
            )
        )
    reason, reason_refs = None, []
    match = re.search(r"(?:因为|由于|原因(?:是|为|[:：]))\s*([^。；;\n]+)", text)
    if match and len(match.group(1).strip()) <= 500:
        candidate = match.group(1).strip()
        for ref in primary.primary.evidence_refs:
            if ref.span and candidate in ref.quote:
                start = ref.span.start + ref.quote.index(candidate)
                reason_refs.append(
                    EvidenceRef(
                        kind=ref.kind,
                        source_id=ref.source_id,
                        span={"start": start, "end": start + len(candidate)},
                        json_path=[],
                        quote=candidate,
                    )
                )
                reason = candidate
                break
    return RefundScopeResult(
        request=RefundRequest(
            operation=operation,
            reason=reason,
            reason_refs=tuple(reason_refs),
            user_claims=(text,),
        )
    )
