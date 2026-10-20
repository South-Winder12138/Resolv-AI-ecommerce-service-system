import importlib
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures/branch_execution_acceptance.json"


def api():
    try:
        return importlib.import_module("resolv.services.branch_evaluation")
    except ModuleNotFoundError:
        pytest.fail("fixed denominator branch evaluation missing")


def perfect():
    module = api()
    cases = module.load_branch_cases(FIXTURE)
    outcomes = [
        {
            "case_id": c["id"],
            "intent": c["intent"],
            "branch": c["branch"],
            "status": c["expect"]["status"],
            "reply_kind": c["expect"]["reply_kind"],
            "counters": {name: 0 for name in module.BLOCKERS},
        }
        for c in cases
    ]
    return cases, outcomes


@pytest.mark.parametrize(
    "counter",
    [
        "route_errors",
        "unexpected_terminal_states",
        "wrong_entity_bindings",
        "fabricated_results",
        "out_of_scope_tools",
        "unapproved_cancellations",
        "duplicate_cancellations",
        "false_escalation_claims",
        "missing_query_results",
        "inconsistent_business_state",
        "preview_submission_requests",
        "technical_errors",
        "usage_unknown",
        "input_underestimates",
        "input_budget_violations",
        "output_budget_violations",
    ],
)
def test_any_unsafe_result_blocks_acceptance(counter):
    cases, outcomes = perfect()
    outcomes[0]["counters"][counter] = 1
    report = api().score_branch_cases(cases, outcomes)
    with pytest.raises(ValueError):
        api().assert_branch_acceptance(report)


def test_deleted_failure_cannot_shrink_denominator():
    cases, outcomes = perfect()
    report = api().score_branch_cases(cases, outcomes[:-1])
    assert report["intent_total"] == 48 and report["boundary_total"] == 12
    with pytest.raises(ValueError):
        api().assert_branch_acceptance(report)


def test_classification_errors_are_counted_even_if_branch_matches():
    cases, outcomes = perfect()
    outcomes[0]["intent"] = "order"
    report = api().score_branch_cases(cases, outcomes)
    assert (
        report["intent_correct"] == 47
        and report["per_class"]["logistics"]["correct"] == 7
    )
    api().assert_branch_acceptance(report)
    outcomes[1]["intent"] = "order"
    with pytest.raises(ValueError):
        api().assert_branch_acceptance(api().score_branch_cases(cases, outcomes))
