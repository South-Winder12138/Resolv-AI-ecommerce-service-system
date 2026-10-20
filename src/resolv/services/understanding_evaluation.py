"""Fixed denominators and one-to-one action/intent/entity scoring."""

from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter

from resolv.schemas.understanding import (
    Intent,
    StrictRecord,
    TextSpan,
    UnderstandingError,
)
from resolv.services.reference_resolution import active_business_ids


class ExpectedRequest(StrictRecord):
    intent: Intent
    entities: list[str]
    span: TextSpan
    task_key: str | None


class EvaluationCase(StrictRecord):
    case_id: str
    group: Literal["intent", "reference", "multi"]
    text: str
    history: list[dict]
    pending: list[dict]
    summary: dict | None
    foreign_history: list[dict]
    gold_primary: ExpectedRequest
    gold_secondary: list[ExpectedRequest]
    expected_entities: list[str]
    allowed_entities: list[str]
    needs_clarification: bool


class EvaluationReport(StrictRecord):
    case_count: int
    intent_accuracy: float
    per_class_correct: dict[str, int]
    resolved_correct: int
    wrong_binding_count: int
    ambiguous_correct: int
    multi_primary_correct: int
    secondary_precision: float
    secondary_recall: float
    constraint_violations: list[str]
    token_underestimates: list[str]
    failures: list[str]
    per_case: list[dict] = Field(default_factory=list)


def load_cases(path: Path):
    cases = TypeAdapter(list[EvaluationCase]).validate_json(
        path.read_bytes(), strict=True
    )
    if len({c.case_id for c in cases}) != len(cases):
        raise UnderstandingError("invalid_evaluation_cases")
    for case in cases:
        for request in [case.gold_primary, *case.gold_secondary]:
            if request.span.end > len(case.text):
                raise UnderstandingError("invalid_evaluation_cases")
    return cases


def expected_objects(gold, text):
    # Gold spans may identify just the action verb. Its fixed original clause,
    # including an object prefix, supplies explicit IDs without editing labels
    # or borrowing an ID from another action's clause/model output.
    if gold.entities:
        return set(gold.entities)
    left = (
        max((text.rfind(c, 0, gold.span.start) for c in "，,；;。\n"), default=-1) + 1
    )
    right = min(
        (p for c in "，,；;。\n" if (p := text.find(c, gold.span.end)) >= 0),
        default=len(text),
    )
    return set(active_business_ids(text[left:right]))


def matches(gold, prediction, text=""):
    if (
        not prediction
        or prediction.get("intent") != gold.intent
        or prediction.get("task_key") != gold.task_key
    ):
        return False
    if not expected_objects(gold, text).issubset(prediction.get("entities", [])):
        return False
    return any(
        span["start"] < gold.span.end and gold.span.start < span["end"]
        for span in prediction.get("evidence_spans", [])
    )


def score_cases(cases, results):
    by_id = {r["case_id"]: r for r in results}
    per_class = {
        intent: 0
        for intent in (
            "logistics",
            "order",
            "product_consultation",
            "refund_return",
            "after_sales",
            "complaint",
            "human",
            "chitchat",
            "other",
        )
    }
    resolved = wrong = ambiguous = multi = tp = predicted = expected = 0
    constraints, token_errors, failures, per_case = [], [], [], []
    intent_denominator = sum(c.group == "intent" for c in cases)
    for case in cases:
        value = by_id.get(case.case_id, {})
        if value.get("error_code") in {
            "classification_entity_invalid",
            "request_semantics_invalid",
        }:
            constraints.append(
                case.case_id + ": rejected unsafe model output " + value["error_code"]
            )
        accepted = value.get("status") in {"ready", "needs_clarification"}
        primary = (
            accepted
            and matches(case.gold_primary, value.get("primary"), case.text)
            and value.get("needs_clarification") == case.needs_clarification
        )
        bindings = value.get("binding_values", [])
        bad_bindings = [v for v in bindings if v not in case.allowed_entities]
        for prediction in [value.get("primary"), *value.get("secondary", [])]:
            if not prediction:
                continue
            supported = [
                g
                for g in [case.gold_primary, *case.gold_secondary]
                if g.task_key == prediction.get("task_key")
                and any(
                    span["start"] < g.span.end and g.span.start < span["end"]
                    for span in prediction.get("evidence_spans", [])
                )
            ]
            if len(supported) > 1:
                constraints.append(case.case_id + ": nonunique action evidence")
            if supported:
                allowed_objects = set().union(
                    *(expected_objects(g, case.text) for g in supported)
                )
                bad_bindings.extend(
                    v
                    for v in prediction.get("entities", [])
                    if v not in allowed_objects and v not in bad_bindings
                )
        wrong += len(bad_bindings)
        if bad_bindings:
            constraints.append(
                case.case_id + ": wrong binding " + ",".join(bad_bindings)
            )
        if (
            case.group == "intent"
            and accepted
            and matches(case.gold_primary, value.get("primary"), case.text)
        ):
            per_class[case.gold_primary.intent] += 1
        if case.group == "reference":
            if case.needs_clarification:
                ambiguous += int(
                    accepted
                    and value.get("needs_clarification") is True
                    and not bindings
                )
            else:
                resolved += int(
                    primary
                    and set(case.expected_entities).issubset(bindings)
                    and not bad_bindings
                )
        if case.group == "multi":
            multi += int(primary)
            remaining = list(case.gold_secondary)
            predictions = value.get("secondary", []) if accepted else []
            predicted += len(predictions)
            expected += len(remaining)
            for prediction in predictions:
                matched = next(
                    (
                        i
                        for i, gold in enumerate(remaining)
                        if matches(gold, prediction, case.text)
                    ),
                    None,
                )
                if matched is not None:
                    tp += 1
                    remaining.pop(matched)
        constraints.extend(case.case_id + ": " + v for v in value.get("violations", []))
        for audit in value.get("audits", []):
            actual, estimate, limit = (
                audit.get("input_tokens"),
                audit["input_estimate"],
                audit["input_limit"],
            )
            if actual is None or not actual <= estimate <= limit:
                token_errors.append(case.case_id)
        if accepted and not value.get("audits"):
            token_errors.append(case.case_id)
        if not primary or bad_bindings:
            failures.append(case.case_id)
        per_case.append(
            {
                "case_id": case.case_id,
                "primary_correct": bool(primary),
                "wrong_bindings": bad_bindings,
                "status": value.get("status"),
                "error_code": value.get("error_code"),
            }
        )
    return EvaluationReport(
        case_count=len(cases),
        intent_accuracy=sum(per_class.values()) / intent_denominator
        if intent_denominator
        else 0.0,
        per_class_correct=per_class,
        resolved_correct=resolved,
        wrong_binding_count=wrong,
        ambiguous_correct=ambiguous,
        multi_primary_correct=multi,
        secondary_precision=tp / predicted
        if predicted
        else (1.0 if not expected else 0.0),
        secondary_recall=tp / expected if expected else 1.0,
        constraint_violations=constraints,
        token_underestimates=token_errors,
        failures=failures,
        per_case=per_case,
    )


def assert_acceptance(report):
    if (
        report.case_count != 140
        or report.intent_accuracy < 0.9
        or min(report.per_class_correct.values()) < 8
        or report.resolved_correct < 19
        or report.wrong_binding_count
        or report.ambiguous_correct != 10
        or report.multi_primary_correct < 19
        or report.secondary_precision < 0.95
        or report.secondary_recall < 0.95
        or report.constraint_violations
        or report.token_underestimates
    ):
        raise UnderstandingError("acceptance_failed")
