import importlib

import pytest
from langchain_core.messages import AIMessage

from resolv.core.understanding_model import JsonStageModel
from resolv.schemas.understanding import Ambiguity, EvidenceRef, UnderstandingError
from resolv.understanding_rules import select_route
from tests.test_understanding_context import harness
from tests.understanding_fixtures import (
    FakeJsonModel,
    input_ref,
    make_classification,
    make_pending_task,
    make_resolution,
)


def module():
    try:
        return importlib.import_module("resolv.services.intent_classification")
    except ModuleNotFoundError:
        pytest.fail("intent classifier missing")


def classify(h, result, resolution=None):
    model = JsonStageModel(FakeJsonModel([AIMessage(content=result.model_dump_json())]))
    classifier = module().IntentClassifier(model, h.context)
    return classifier.classify(
        h.snapshot, resolution or make_resolution(), lambda _: None, lambda _: None
    )


@pytest.mark.parametrize(
    "intent,branch",
    [
        ("logistics", "business"),
        ("order", "business"),
        ("product_consultation", "knowledge"),
        ("refund_return", "refund_flow"),
        ("after_sales", "refund_flow"),
        ("complaint", "escalate"),
        ("human", "business"),
        ("chitchat", "fallback_script"),
        ("other", "fallback_script"),
    ],
)
def test_valid_model_contract_uses_program_mapping(intent, branch):
    h = harness()
    result = classify(h, make_classification(["查它物流"], [intent]))
    assert select_route(h.snapshot, make_resolution(), result.output).branch == branch


def test_uncertain_main_does_not_fall_back_to_other():
    h = harness()
    c = make_classification(["查它物流"], [None])
    c.requests[0].ambiguities = [
        Ambiguity(kind="intent", question="查询订单还是申请退款？", evidence_refs=[])
    ]
    result = classify(h, c)
    selected = select_route(h.snapshot, make_resolution(), result.output)
    assert (
        selected.branch is None
        and not selected.route_ready
        and selected.clarification_question
    )


@pytest.mark.parametrize(
    "mutation", ["binding", "priority", "source", "entity", "missing_ambiguity"]
)
def test_invalid_classification_cannot_publish(mutation):
    h = harness()
    c = make_classification(["查它物流"], ["logistics"])
    request = c.requests[0]
    if mutation == "binding":
        request.binding_indexes = [0]
    elif mutation == "priority":
        request.explicit_priority = 0
        request.priority_evidence_refs = request.evidence_refs
    elif mutation == "source":
        request.evidence_refs[0].source_id = "foreign"
    elif mutation == "entity":
        request.text = "查 MH999 的物流"
    else:
        request.intent = None
    with pytest.raises(UnderstandingError):
        classify(h, c)


def test_secondary_ambiguity_does_not_change_resolution():
    h = harness()
    h.snapshot.original_text = "查物流，另外退它"
    from resolv.repositories.understanding import source
    from resolv.schemas.understanding import SourceFingerprint

    current = source("input", "current", text=h.snapshot.original_text)
    h.snapshot.model_sources = [current]
    h.snapshot.source_fingerprints = [
        SourceFingerprint(
            kind="input", source_id="current", source_hash=current.source_hash
        )
    ]
    c = make_classification(
        ["查物流", "退它"],
        ["logistics", "refund_return"],
        original_text=h.snapshot.original_text,
    )
    c.requests[1].ambiguities = [
        Ambiguity(kind="object", question="哪笔退款？", evidence_refs=[])
    ]
    resolution = make_resolution(
        unresolved_references=[
            {
                "span": {"start": 7, "end": 8},
                "candidate_refs": [],
                "question": "哪笔退款？",
            }
        ]
    )
    before = resolution.model_dump_json()
    result = classify(h, c, resolution)
    assert resolution.model_dump_json() == before
    assert select_route(h.snapshot, resolution, result.output).route_ready


def test_known_intent_cannot_introduce_business_parameter_clarification():
    from resolv.schemas.understanding import UnderstandingInput

    h = harness()
    text = "查退款进度"
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text=text)
    )
    c = make_classification([text], ["refund_return"])
    c.requests[0].ambiguities = [
        Ambiguity(
            kind="missing_parameter",
            question="请提供退款编号？",
            evidence_refs=c.requests[0].evidence_refs,
        )
    ]
    with pytest.raises(UnderstandingError, match="classification_ambiguity_invalid"):
        classify(h, c, make_resolution(resolved_text=text))


def test_saved_classification_can_be_revalidated_without_model_call():
    h = harness()
    client = FakeJsonModel([AssertionError("no inference during revalidation")])
    classifier = module().IntentClassifier(JsonStageModel(client), h.context)
    output = make_classification(["查它物流"], ["logistics"])
    verified = classifier.validate(
        h.snapshot, make_resolution(), output, lambda _: None
    )
    assert {r.source_id for r in verified.values()} == {"current"}
    assert client.calls == []


def test_same_class_pending_without_distinguishing_input_cannot_choose_one():
    h = harness(
        pending=[
            make_pending_task("a", "refund_return"),
            make_pending_task("b", "refund_return"),
        ]
    )
    c = make_classification(["查它物流"], ["refund_return"])
    c.requests[0].evidence_refs.append(
        EvidenceRef(
            kind="task", source_id="a", span=None, json_path=["text"], quote="退款"
        )
    )
    with pytest.raises(UnderstandingError, match="continuation_conflict"):
        classify(h, c, make_resolution(continuation_selection="a"))


def test_unknown_enum_and_model_routing_field_are_rejected():
    h = harness()
    payload = '{"requests":[{"text":"查它物流","intent":"order","branch":"business"}]}'
    model = JsonStageModel(FakeJsonModel([AIMessage(content=payload)]))
    with pytest.raises(UnderstandingError, match="model_output_invalid"):
        module().IntentClassifier(model, h.context).classify(
            h.snapshot, make_resolution(), lambda _: None, lambda _: None
        )


def test_explicit_entities_cannot_move_between_action_evidence():
    from resolv.schemas.understanding import UnderstandingInput

    h = harness()
    text = "订单 MH001 申请退款，同时查订单 MH002 金额"
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text=text)
    )
    c = make_classification(
        ["订单 MH001 申请退款", "查订单 MH002 金额"],
        ["refund_return", "order"],
        original_text=text,
    )
    c.requests[0].text = "订单 MH002 申请退款"
    c.requests[1].text = "查订单 MH001 金额"
    with pytest.raises(UnderstandingError, match="classification_entity_invalid"):
        classify(h, c, make_resolution(resolved_text=text))


def test_multiple_actions_cannot_share_whole_input_as_their_only_evidence():
    from resolv.schemas.understanding import UnderstandingInput

    h = harness()
    text = "订单 MH001 申请退款，同时查订单 MH002 金额"
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text=text)
    )
    c = make_classification(
        ["订单 MH001 申请退款", "查订单 MH002 金额"],
        ["refund_return", "order"],
        original_text=text,
    )
    for request in c.requests:
        request.evidence_refs = [EvidenceRef(**input_ref(text))]
    with pytest.raises(UnderstandingError, match="classification_evidence_invalid"):
        classify(h, c, make_resolution(resolved_text=text))


def test_negated_old_identifier_cannot_become_the_active_request_object():
    from resolv.schemas.understanding import UnderstandingInput

    h = harness()
    text = "不是订单 MH016，是订单 MH017，查它物流"
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text=text)
    )
    c = make_classification([text], ["logistics"])
    c.requests[0].text = "查 MH016 物流"
    with pytest.raises(UnderstandingError, match="classification_entity_invalid"):
        classify(h, c, make_resolution(resolved_text=text))


def test_binding_index_must_belong_to_request_action_span():
    from resolv.schemas.understanding import UnderstandingInput

    h = harness()
    text = "查它物流，同时退那笔"
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text=text)
    )
    resolution = make_resolution(
        resolved_text="查 MH001 物流，同时退 MH002",
        bindings=[
            {
                "span": {"start": 1, "end": 2},
                "entity_type": "order",
                "value": "MH001",
                "evidence_refs": [input_ref("查它物流")],
            },
            {
                "span": {"start": 8, "end": 10},
                "entity_type": "order",
                "value": "MH002",
                "evidence_refs": [input_ref("退那笔", 7)],
            },
        ],
    )
    c = make_classification(
        ["查它物流", "退那笔"], ["logistics", "refund_return"], original_text=text
    )
    c.requests[0].text = "查 MH002 物流"
    c.requests[0].binding_indexes = [1]
    with pytest.raises(UnderstandingError, match="classification_entity_invalid"):
        classify(h, c, resolution)


@pytest.mark.parametrize("rewritten", ["已完成退款", "退款成功", "用户已批准退款"])
def test_classification_cannot_claim_business_completion_or_approval(rewritten):
    from resolv.schemas.understanding import UnderstandingInput

    h = harness()
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text="我要退款")
    )
    c = make_classification(["我要退款"], ["refund_return"])
    c.requests[0].text = rewritten
    with pytest.raises(UnderstandingError, match="request_semantics_invalid"):
        classify(h, c, make_resolution(resolved_text="我要退款"))


def continuation_case(tasks, text, selected):
    from resolv.schemas.understanding import UnderstandingInput

    h = harness(pending=tasks)
    h.snapshot = h.context.freeze(
        h.sources, UnderstandingInput(request_id="r", text=text)
    )
    c = make_classification([text], ["refund_return"])
    task = next(t for t in tasks if t.task_id == selected)
    c.requests[0].evidence_refs.append(
        EvidenceRef(
            kind="task",
            source_id=selected,
            span=None,
            json_path=["text"],
            quote=task.text,
        )
    )
    return h, c, make_resolution(resolved_text=text, continuation_selection=selected)


def test_continuation_explicit_task_objects_disambiguate_without_bindings():
    tasks = [
        make_pending_task(i, "refund_return", text=f"订单 {v} 申请退款")
        for i, v in [("a", "MH001"), ("b", "MH002")]
    ]
    h, c, r = continuation_case(tasks, "继续订单 MH001 退款", "a")
    assert classify(h, c, r).output.requests[0].intent == "refund_return"


def test_shared_object_does_not_uniquely_select_different_pending_actions():
    from tests.test_reference_resolution import binding

    tasks = [
        make_pending_task(
            i,
            "refund_return",
            text=t,
            bindings=[binding({"start": 1, "end": 2}, "old-message")],
        )
        for i, t in [("a", "查 MH001 退款进度"), ("b", "给 MH001 申请退货")]
    ]
    h, c, r = continuation_case(tasks, "继续 MH001 退款", "a")
    with pytest.raises(UnderstandingError, match="continuation_conflict"):
        classify(h, c, r)


def test_shared_object_with_distinct_action_can_select_unique_task():
    tasks = [
        make_pending_task(i, "refund_return", text=t)
        for i, t in [("a", "查 MH001 退款进度"), ("b", "给 MH001 申请退货")]
    ]
    h, c, r = continuation_case(tasks, "继续 MH001 退款进度", "a")
    assert classify(h, c, r).output.requests[0].intent == "refund_return"


def test_unknown_pending_intent_can_be_refined_by_current_input():
    h, c, r = continuation_case(
        [make_pending_task("a", None, text="退它", branch=None)],
        "继续退款，是订单 MH002",
        "a",
    )
    assert classify(h, c, r).output.requests[0].intent == "refund_return"
