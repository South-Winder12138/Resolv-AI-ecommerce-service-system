"""Provenance and narrow scope checks over frozen understanding sources."""

import pytest

from resolv.repositories.understanding import source
from resolv.schemas.branch_execution import PrimaryExecution
from resolv.schemas.refund import OrderBinding, RefundFlowError
from resolv.schemas.understanding import EntityBinding, SourceFingerprint
from resolv.services.understanding_context import snapshot_digest
from tests.branch_fixtures import make_view
from tests.understanding_fixtures import make_snapshot


def case(text, *, bindings=(), extra_sources=()):
    sources = [source("input", "current", text=text), *extra_sources]
    snapshot = make_snapshot(
        text,
        model_sources=sources,
        source_fingerprints=[
            SourceFingerprint(
                kind=s.kind, source_id=s.source_id, source_hash=s.source_hash
            )
            for s in sources
        ],
    )
    snapshot.snapshot_hash = snapshot_digest(snapshot)
    p = make_view(text, "refund_return").primary_request
    p.bindings = list(bindings)
    return PrimaryExecution(
        conversation_id="c",
        understanding_id="u-1",
        snapshot_hash=snapshot.snapshot_hash,
        primary=p,
    ), snapshot


def bind(p, s, **kwargs):
    from resolv.services.refund_scope import bind_refund_order

    return bind_refund_order(p, s, order_source="tenant-a", **kwargs)


def test_refund_order_provenance_and_correction():
    p, s = case("订单不是A001，是B002，帮我申请整单退款")
    result = bind(p, s)
    assert isinstance(result, OrderBinding) and result.order_id == "B002"
    for text in [
        "帮我申请退款",
        "订单A001和B002都退款",
        "商品SKU123申请退款",
        "运单SF123申请退款",
        "订单金额100元退款",
    ]:
        p, s = case(text)
        assert bind(p, s).kind == "clarification"
    p, s = case("订单A001退款")
    s.original_text = "订单B002退款"
    with pytest.raises(RefundFlowError):
        bind(p, s)


def test_summary_is_only_a_hint_and_original_source_must_match():
    summary = source("summary", "summary:1", json_value={"order": "A001"})
    original = source("message", "m", text="订单A001")
    b = EntityBinding(
        span={"start": 0, "end": 3},
        entity_type="order",
        value="A001",
        evidence_refs=[
            {
                "kind": "summary",
                "source_id": "summary:1",
                "span": None,
                "json_path": ["order"],
                "quote": "A001",
            }
        ],
    )
    p, s = case("这笔单申请退款", bindings=[b], extra_sources=[summary, original])
    assert bind(p, s).kind == "clarification"
    result = bind(p, s, source_loader=lambda _: original)
    assert result.order_id == "A001" and any(
        r.kind == "message" for r in result.source_refs
    )
    with pytest.raises(RefundFlowError):
        bind(p, s, source_loader=lambda _: source("message", "m", text="订单B002"))


def test_action_source_namespace_cannot_be_remapped():
    action = source(
        "action", "a", json_value={"order_id": "A001", "order_source": "tenant-b"}
    )
    b = EntityBinding(
        span={"start": 0, "end": 3},
        entity_type="order",
        value="A001",
        evidence_refs=[
            {
                "kind": "action",
                "source_id": "a",
                "span": None,
                "json_path": ["order_id"],
                "quote": "A001",
            }
        ],
    )
    p, s = case("这笔单申请退款", bindings=[b], extra_sources=[action])
    with pytest.raises(RefundFlowError):
        bind(p, s, source_loader=lambda _: action)


@pytest.mark.parametrize("path", ["binding", "primary", "summary"])
@pytest.mark.parametrize("namespace", [None, "tenant-a"])
def test_historical_action_requires_explicit_source(path, namespace):
    # This is the actual old OrderService.get_order result shape. Its ID alone
    # must never be reinterpreted in the currently configured external source.
    value = {"found": True, "order_id": "A001", "status": "processing"}
    if namespace is not None:
        value["order_source"] = namespace
    action = source("action", "old-order-action", json_value=value)
    ref = {
        "kind": "action",
        "source_id": action.source_id,
        "span": None,
        "json_path": ["order_id"],
        "quote": "A001",
    }
    extra = [action]
    if path == "summary":
        extra.append(source("summary", "summary:1", json_value={"order": "A001"}))
        ref = {
            **ref,
            "kind": "summary",
            "source_id": "summary:1",
            "json_path": ["order"],
        }
    binding = EntityBinding(
        span={"start": 0, "end": 3},
        entity_type="order",
        value="A001",
        evidence_refs=[ref],
    )
    p, s = case("这笔单申请退款", bindings=[binding], extra_sources=extra)
    if path == "primary":
        p.primary.text = "订单A001申请退款"
        p.primary.evidence_refs = [ref]
        p.primary.bindings = []
    if namespace is None:
        with pytest.raises(RefundFlowError, match="refund_source_conflict"):
            bind(p, s, source_loader=lambda _: action)
    else:
        assert bind(p, s, source_loader=lambda _: action).order_id == "A001"


def test_rewritten_product_or_summary_cannot_become_order_proof():
    p, s = case("商品SKU123申请退款")
    p.primary.text = "订单SKU123申请退款"
    assert bind(p, s).kind == "clarification"
    summary = source("summary", "summary:1", json_value={"order": "A001"})
    p, s = case("这笔单申请退款", extra_sources=[summary])
    p.primary.text = "订单A001申请退款"
    p.primary.evidence_refs = [
        {
            "kind": "summary",
            "source_id": "summary:1",
            "span": None,
            "json_path": ["order"],
            "quote": "A001",
        }
    ]
    assert bind(p, s).kind == "clarification"


@pytest.mark.parametrize(
    "text,kind",
    [
        ("订单A001维修", "human_draft"),
        ("订单A001换货", "human_draft"),
        ("订单A001保修", "human_draft"),
        ("订单A001维修或退款", "clarification"),
        ("订单A001退款", "clarification"),
        ("订单A001部分退款", "human_draft"),
        ("订单A001取消退款申请", "human_draft"),
        ("订单A001不要申请退款", "clarification"),
    ],
)
def test_scope_after_sales_and_ambiguity(text, kind):
    from resolv.services.refund_scope import resolve_refund_scope

    p, s = case(text)
    assert resolve_refund_scope(p, s).scope_reply.kind == kind


@pytest.mark.parametrize(
    "text,operation",
    [
        ("订单A001可以退款吗", "eligibility"),
        ("订单A001帮我申请退款，因为商品未激活", "apply"),
        ("订单A001退款进度如何", "status"),
    ],
)
def test_explicit_operation_and_reason_are_sourced_user_claims(text, operation):
    from resolv.services.refund_scope import resolve_refund_scope

    p, s = case(text)
    request = resolve_refund_scope(p, s).request
    assert request.operation == operation
    if operation == "apply":
        assert (
            request.reason == "商品未激活"
            and request.reason_refs[0].quote == "商品未激活"
        )
        assert request.user_claims
