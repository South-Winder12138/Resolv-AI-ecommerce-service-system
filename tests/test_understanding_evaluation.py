import importlib
from pathlib import Path

import pytest

from resolv.schemas.understanding import UnderstandingError

CASES = Path(__file__).parent / "fixtures/understanding_acceptance.json"


def module():
    try:
        return importlib.import_module("resolv.services.understanding_evaluation")
    except ModuleNotFoundError:
        pytest.fail("understanding evaluation missing")


def predicted(case):
    def item(gold):
        return {
            "intent": gold.intent,
            "entities": sorted(module().expected_objects(gold, case.text)),
            "evidence_spans": [gold.span.model_dump()],
            "task_key": gold.task_key,
        }

    return {
        "case_id": case.case_id,
        "primary": item(case.gold_primary),
        "secondary": [item(g) for g in case.gold_secondary],
        "binding_values": case.expected_entities,
        "needs_clarification": case.needs_clarification,
        "status": "needs_clarification" if case.needs_clarification else "ready",
        "violations": [],
        "audits": [{"input_tokens": 500, "input_estimate": 550, "input_limit": 12000}],
        "error_code": None,
    }


def test_fixed_gold_has_nine_by_ten_thirty_references_and_twenty_multi():
    cases = module().load_cases(CASES)
    assert len(cases) == 140
    assert (
        sum(c.group == "reference" and not c.needs_clarification for c in cases) == 20
    )
    assert sum(c.group == "reference" and c.needs_clarification for c in cases) == 10
    assert sum(c.group == "multi" for c in cases) == 20
    assert {
        intent: sum(
            c.group == "intent" and c.gold_primary.intent == intent for c in cases
        )
        for intent in {c.gold_primary.intent for c in cases}
    } == {
        "logistics": 10,
        "order": 10,
        "product_consultation": 10,
        "refund_return": 10,
        "after_sales": 10,
        "complaint": 10,
        "human": 10,
        "chitchat": 10,
        "other": 10,
    }


def test_dangerous_binding_blocks_even_high_accuracy():
    mod = module()
    cases = mod.load_cases(CASES)
    results = [predicted(c) for c in cases]
    report = mod.score_cases(cases, results)
    mod.assert_acceptance(report)
    results[90]["binding_values"] = ["MH999"]
    report = mod.score_cases(cases, results)
    assert report.intent_accuracy == 1 and report.wrong_binding_count == 1
    with pytest.raises(UnderstandingError, match="acceptance_failed"):
        mod.assert_acceptance(report)


def test_failed_cases_remain_in_denominator():
    mod = module()
    cases = mod.load_cases(CASES)
    results = [predicted(c) for c in cases[1:]]
    report = mod.score_cases(cases, results)
    assert report.case_count == 140 and report.per_class_correct["logistics"] == 9
    assert report.intent_accuracy == 89 / 90


def test_intent_accuracy_measures_intent_even_when_object_needs_clarification():
    mod = module()
    case = mod.load_cases(CASES)[0]
    result = predicted(case)
    result["status"] = "needs_clarification"
    result["needs_clarification"] = True
    assert mod.score_cases([case], [result]).per_class_correct["logistics"] == 1


def test_evidence_may_quote_action_within_gold_request_range():
    mod = module()
    case = mod.load_cases(CASES)[0]
    result = predicted(case)
    result["primary"]["evidence_spans"] = [{"start": 9, "end": 13}]
    assert mod.score_cases([case], [result]).per_class_correct["logistics"] == 1
    result["primary"]["evidence_spans"] = [{"start": 30, "end": 34}]
    assert mod.score_cases([case], [result]).per_class_correct["logistics"] == 0


def test_one_prediction_cannot_match_two_secondary_requests():
    mod = module()
    cases = mod.load_cases(CASES)
    c = next(c for c in cases if len(c.gold_secondary) == 2)
    result = predicted(c)
    result["secondary"] = result["secondary"][:1]
    report = mod.score_cases([c], [result])
    assert report.secondary_precision == 1 and report.secondary_recall == 0.5


def test_explicit_action_entity_swap_is_a_blocking_evaluation_error():
    mod = module()
    case = next(
        c for c in mod.load_cases(CASES) if c.case_id == "intent-refund_return-03"
    )
    result = predicted(case)
    result["primary"]["entities"] = ["MH001"]
    report = mod.score_cases([case], [result])
    assert report.wrong_binding_count == 1
    assert report.per_class_correct["refund_return"] == 0
    assert report.constraint_violations


def test_objects_preceding_action_gold_spans_are_associated_by_original_clause():
    mod = module()
    case = next(c for c in mod.load_cases(CASES) if c.case_id == "multi-08")
    result = predicted(case)
    result["primary"]["entities"] = ["MH001"]
    result["secondary"][0]["entities"] = ["MH002"]
    correct = mod.score_cases([case], [result])
    assert not correct.wrong_binding_count and correct.multi_primary_correct == 1
    assert correct.secondary_recall == 1
    result["primary"]["entities"], result["secondary"][0]["entities"] = (
        ["MH002"],
        ["MH001"],
    )
    wrong = mod.score_cases([case], [result])
    assert wrong.wrong_binding_count == 2 and wrong.multi_primary_correct == 0
    assert wrong.secondary_recall == 0


def test_whole_multi_input_is_not_unique_object_action_evidence():
    mod = module()
    case = next(c for c in mod.load_cases(CASES) if c.case_id == "multi-08")
    result = predicted(case)
    for item in [result["primary"], *result["secondary"]]:
        item["evidence_spans"] = [{"start": 0, "end": len(case.text)}]
    assert mod.score_cases([case], [result]).constraint_violations


def test_negated_previous_object_is_not_an_expected_current_binding():
    mod = module()
    case = next(c for c in mod.load_cases(CASES) if c.case_id == "reference-clear-16")
    result = predicted(case)
    result["primary"]["entities"] = ["MH017"]
    report = mod.score_cases([case], [result])
    assert report.resolved_correct == 1 and not report.wrong_binding_count


@pytest.mark.parametrize(
    "code", ["classification_entity_invalid", "request_semantics_invalid"]
)
def test_rejected_unsafe_model_assertions_still_block_acceptance(code):
    mod = module()
    case = mod.load_cases(CASES)[0]
    result = predicted(case)
    result.update(status="interrupted", primary=None, error_code=code)
    assert mod.score_cases([case], [result]).constraint_violations


@pytest.mark.parametrize(
    "audit",
    [
        {"input_tokens": 501, "input_estimate": 500, "input_limit": 12000},
        {"input_tokens": None, "input_estimate": 550, "input_limit": 12000},
        {"input_tokens": 500, "input_estimate": 12001, "input_limit": 12000},
    ],
)
def test_bad_or_unknown_usage_never_passes_calibration(audit):
    mod = module()
    cases = mod.load_cases(CASES)
    results = [predicted(c) for c in cases]
    results[0]["audits"] = [audit]
    with pytest.raises(UnderstandingError, match="acceptance_failed"):
        mod.assert_acceptance(mod.score_cases(cases, results))
