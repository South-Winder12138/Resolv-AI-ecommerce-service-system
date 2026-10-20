import importlib

import pytest
from pydantic import ValidationError


def get_api():
    # A missing production contract is a feature failure, not a collection error.
    try:
        schema = importlib.import_module("resolv.schemas.understanding")
        rules = importlib.import_module("resolv.understanding_rules")
        builders = importlib.import_module("tests.understanding_fixtures")
    except ModuleNotFoundError as exc:
        pytest.fail(f"understanding contract missing: {exc.name}")
    return schema, rules, builders


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
def test_nine_intents_route_without_business_execution(intent, branch):
    api = get_api()
    _, rules, f = api
    selection = rules.select_route(
        f.make_snapshot("请求"),
        f.make_resolution(),
        f.make_classification(["请求"], [intent]),
    )
    assert selection.branch == branch and selection.route_ready
    assert selection.primary_index == 0 and selection.secondary_indexes == []


@pytest.mark.parametrize(
    "patch",
    [
        {"bindings": None},
        {"extra": "execute"},
        {"continuation_selection": 123},
        {"unresolved_references": ()},
        {"resolved_text": 12},
    ],
)
def test_model_schema_rejects_coercion_and_extra_fields(patch):
    api = get_api()
    _, _, f = api
    with pytest.raises(ValidationError):
        f.make_resolution(**patch)


def test_all_model_fields_are_required():
    api = get_api()
    schema, _, f = api
    data = f.make_resolution().model_dump()
    for field in data:
        with pytest.raises(ValidationError):
            schema.ResolutionResult.model_validate(
                {k: v for k, v in data.items() if k != field}
            )
    classified = f.make_classification(["查订单"], ["order"]).model_dump()["requests"][
        0
    ]
    for field in classified:
        with pytest.raises(ValidationError):
            schema.ClassifiedRequest.model_validate(
                {k: v for k, v in classified.items() if k != field}
            )
    with pytest.raises(ValidationError):
        f.make_classification(["查订单"], ["unknown"])


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "", "text": "查订单"},
        {"request_id": "r" * 129, "text": "查订单"},
        {"request_id": "r", "text": "  "},
        {
            "request_id": "r",
            "text": "查订单",
            "clarifies_understanding_id": "p",
            "continue_task_id": "t",
        },
        {"request_id": "r", "text": "查订单", "execute": True},
    ],
)
def test_input_rejects_invalid_or_conflicting_links(payload):
    api = get_api()
    schema, _, _ = api
    with pytest.raises(ValidationError):
        schema.UnderstandingInput.model_validate(payload)


def test_default_order_uses_original_spans_not_model_list_order():
    api = get_api()
    _, rules, f = api
    classified = f.make_classification(
        ["投诉", "查订单"], ["complaint", "order"], original_text="查订单，然后投诉"
    )
    result = rules.select_route(
        f.make_snapshot("查订单，然后投诉"), f.make_resolution(), classified
    )
    assert result.primary_index == 1 and result.secondary_indexes == [0]


def test_clarification_order_uses_parent_action_spans_before_answer_spans():
    schema, rules, f = get_api()
    c = f.make_classification(
        ["退款", "查物流"], ["refund_return", "logistics"], original_text="查物流，退款"
    )
    for request in c.requests:
        request.evidence_refs = [
            request.evidence_refs[0].model_copy(
                update={"kind": "understanding", "source_id": "parent"}
            ),
            schema.EvidenceRef(**f.input_ref("MH001")),
        ]
    snapshot = f.make_snapshot(
        "MH001",
        linked_inputs={"parent": "查物流，退款"},
        parent={"understanding_id": "parent"},
    )
    result = rules.select_route(snapshot, f.make_resolution(), c)
    assert result.primary_index == 1 and result.secondary_indexes == [0]


def test_explicit_priority_beats_appearance():
    api = get_api()
    _, rules, f = api
    c = f.make_classification(
        ["退款", "查物流"],
        ["refund_return", "logistics"],
        original_text="退款先等等，先查物流",
    )
    c.requests[1].explicit_priority = 0
    c.requests[1].priority_evidence_refs = [f.input_ref("先查物流", 6)]
    result = rules.select_route(
        f.make_snapshot("退款先等等，先查物流"), f.make_resolution(), c
    )
    assert result.primary_index == 1


def test_conflicting_priority_clarifies_without_reselecting():
    api = get_api()
    _, rules, f = api
    c = f.make_classification(["物流", "退款"], ["logistics", "refund_return"])
    for request in c.requests:
        request.explicit_priority = 0
        request.priority_evidence_refs = request.evidence_refs
    result = rules.select_route(f.make_snapshot("物流，退款"), f.make_resolution(), c)
    assert result.primary_index == 0 and not result.route_ready
    assert result.clarification_question


@pytest.mark.parametrize("index,ready", [(0, False), (1, True)])
def test_only_primary_ambiguity_blocks_route(index, ready):
    api = get_api()
    schema, rules, f = api
    c = f.make_classification(["查订单", "退款"], ["order", "refund_return"])
    c.requests[index].ambiguities = [
        schema.Ambiguity(kind="object", question="哪笔订单？", evidence_refs=[])
    ]
    result = rules.select_route(f.make_snapshot("查订单，退款"), f.make_resolution(), c)
    assert result.primary_index == 0 and result.branch == "business"
    assert result.route_ready is ready


def test_unresolved_primary_keeps_suggested_branch():
    api = get_api()
    _, rules, f = api
    resolution = f.make_resolution(
        unresolved_references=[
            {
                "span": {"start": 1, "end": 2},
                "candidate_refs": [],
                "question": "它是哪笔订单？",
            }
        ]
    )
    result = rules.select_route(
        f.make_snapshot("查它物流"),
        resolution,
        f.make_classification(["查它物流"], ["logistics"]),
    )
    assert result.branch == "business" and not result.route_ready


def test_unknown_intent_is_clarification_not_other():
    api = get_api()
    schema, rules, f = api
    c = f.make_classification(["帮我处理"], [None])
    c.requests[0].ambiguities = [
        schema.Ambiguity(kind="intent", question="需要什么帮助？", evidence_refs=[])
    ]
    result = rules.select_route(f.make_snapshot("帮我处理"), f.make_resolution(), c)
    assert result.branch is None and not result.route_ready


def test_explicit_order_beats_continuation_without_duplicate():
    api = get_api()
    schema, rules, f = api
    snapshot = f.make_snapshot(
        "先查物流，再继续退款",
        pending_requests=[f.make_pending_task("old-refund", "refund_return")],
    )
    resolution = f.make_resolution(continuation_selection="old-refund")
    classified = f.make_classification(
        ["查物流", "继续退款"],
        ["logistics", "refund_return"],
        original_text=snapshot.original_text,
    )
    classified.requests[1].evidence_refs.append(
        schema.EvidenceRef(
            kind="task",
            source_id="old-refund",
            span=None,
            json_path=["text"],
            quote="退款",
        )
    )
    selected = rules.select_route(snapshot, resolution, classified)
    assert (selected.primary_index, selected.branch) == (0, "business")
    assert selected.continuation_task_id is None and selected.secondary_indexes == []


def test_continuation_id_has_no_priority_and_is_selected_only_when_primary():
    api = get_api()
    schema, rules, f = api
    snapshot = f.make_snapshot(
        "继续退款",
        pending_requests=[f.make_pending_task("old-refund", "refund_return")],
    )
    c = f.make_classification(["继续退款"], ["refund_return"])
    c.requests[0].evidence_refs.append(
        schema.EvidenceRef(
            kind="task",
            source_id="old-refund",
            span=None,
            json_path=["text"],
            quote="退款",
        )
    )
    result = rules.select_route(
        snapshot, f.make_resolution(continuation_selection="old-refund"), c
    )
    assert result.continuation_task_id == "old-refund"
