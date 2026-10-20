"""Fixed-denominator scoring, separate from inference and public response rendering."""

import hashlib
import json
from collections import Counter
from pathlib import Path

CLASSES = ("logistics", "order", "human", "complaint", "chitchat", "other")
BLOCKERS = (
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
)


def load_branch_cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    cases = value["cases"]
    counts = Counter(case["group"] for case in cases)
    if (
        value.get("schema_version") != 1
        or len(cases) != 60
        or len({c["id"] for c in cases}) != 60
        or counts != Counter({**dict.fromkeys(CLASSES, 8), "boundary": 12})
    ):
        raise ValueError("fixed 48 plus 12 gold cases required")
    return cases


def score_branch_cases(cases, outcomes) -> dict:
    if len(cases) != 60 or Counter(c["group"] for c in cases) != Counter(
        {**dict.fromkeys(CLASSES, 8), "boundary": 12}
    ):
        raise ValueError("gold denominator cannot change")
    by_id = {outcome["case_id"]: outcome for outcome in outcomes}
    totals = dict.fromkeys(BLOCKERS, 0)
    report = {
        "gold_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "intent_correct": 0,
        "intent_total": 48,
        "boundary_passed": 0,
        "boundary_total": 12,
        "per_class": {name: {"correct": 0, "total": 8} for name in CLASSES},
        "outcome_count": len(outcomes),
        "boundaries": [],
    }
    if len(by_id) != len(outcomes) or set(by_id) != {case["id"] for case in cases}:
        totals["technical_errors"] += 1
    for case in cases:
        outcome = by_id.get(case["id"], {})
        correct = outcome.get("intent") == case["intent"]
        counters = {name: outcome.get("counters", {}).get(name, 0) for name in BLOCKERS}
        if outcome.get("branch") != case["branch"]:
            counters["route_errors"] += 1
        if outcome.get("status") != case["expect"]["status"] or (
            case["expect"]["status"] == "completed"
            and outcome.get("reply_kind") != case["expect"]["reply_kind"]
        ):
            counters["unexpected_terminal_states"] += 1
        for name, count in counters.items():
            if type(count) is not int or count < 0:
                raise ValueError("invalid counter")
            totals[name] += count
        if case["group"] == "boundary":
            passed = correct and not any(counters.values())
            report["boundary_passed"] += int(passed)
            report["boundaries"].append(
                {
                    "case_id": case["id"],
                    "passed": passed,
                    "intent_correct": correct,
                    "counters": counters,
                }
            )
        else:
            report["intent_correct"] += int(correct)
            report["per_class"][case["group"]]["correct"] += int(correct)
    return report | totals


def assert_branch_acceptance(report: dict) -> None:
    if (
        report.get("intent_total") != 48
        or report.get("boundary_total") != 12
        or report.get("outcome_count") != 60
        or report.get("intent_correct", 0) < 46
        or report.get("boundary_passed") != 12
        or set(report.get("per_class", {})) != set(CLASSES)
        or any(
            item.get("total") != 8 or item.get("correct", 0) < 7
            for item in report["per_class"].values()
        )
        or any(report.get(name) != 0 for name in BLOCKERS)
    ):
        raise ValueError("branch acceptance gate failed")
