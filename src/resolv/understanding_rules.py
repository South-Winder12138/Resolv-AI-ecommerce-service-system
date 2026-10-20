"""Deterministic routing; a ready route carries no execution authority."""

from resolv.schemas.understanding import (
    Branch,
    ClassificationResult,
    ContextSnapshot,
    Intent,
    ResolutionResult,
    RouteSelection,
    UnderstandingError,
)

INTENT_BRANCHES_V1: dict[Intent, Branch] = {
    "logistics": "business",
    "order": "business",
    "product_consultation": "knowledge",
    "refund_return": "refund_flow",
    "after_sales": "refund_flow",
    "complaint": "escalate",
    "human": "business",
    "chitchat": "fallback_script",
    "other": "fallback_script",
}


def request_position(request, snapshot):
    if snapshot.parent:
        parent_id = snapshot.parent["understanding_id"]
        parent_positions = [
            ref.span.start
            for ref in request.evidence_refs
            if ref.kind == "understanding" and ref.source_id == parent_id and ref.span
        ]
        if parent_positions:
            return min(parent_positions)
    positions = []
    for ref in request.evidence_refs:
        if ref.span is not None and (
            ref.kind == "input"
            and ref.source_id == "current"
            or (
                ref.kind in {"input", "understanding"}
                and ref.source_id in snapshot.linked_inputs
            )
        ):
            positions.append(ref.span.start)
    if not positions:
        raise UnderstandingError("classification_evidence_invalid")
    offset = (
        len(snapshot.linked_inputs.get(snapshot.parent["understanding_id"], ""))
        if snapshot.parent
        else 0
    )
    return offset + min(positions)


def referenced_task(request, snapshot):
    available = {task.task_id for task in snapshot.pending_requests}
    ids = {ref.source_id for ref in request.evidence_refs if ref.kind == "task"}
    if not ids.issubset(available) or len(ids) > 1:
        raise UnderstandingError("continuation_conflict")
    return next(iter(ids), None)


def preserved_parent_task(request, snapshot):
    """Clarification may retain an existing secondary, without selecting it."""
    task_id = referenced_task(request, snapshot)
    parent = snapshot.parent
    if not task_id or not parent or not parent.get("final_json"):
        return False
    if not any(
        t["task_id"] == task_id for t in parent["final_json"]["pending_requests"]
    ):
        return False
    task = next(t for t in snapshot.pending_requests if t.task_id == task_id)
    parent_requests = parent["classification"]["output"]["requests"]
    if task.understanding_id == parent["understanding_id"]:
        original = [parent_requests[task.ordinal]]
    else:
        original = [
            r
            for r in parent_requests
            if any(
                ref["kind"] == "task" and ref["source_id"] == task_id
                for ref in r["evidence_refs"]
            )
        ]
    return any(
        ref.kind == "understanding"
        and ref.source_id == parent["understanding_id"]
        and ref.span
        and old["kind"] == "input"
        and old["source_id"] == "current"
        and old["span"]
        and ref.span.start < old["span"]["end"]
        and old["span"]["start"] < ref.span.end
        for ref in request.evidence_refs
        for r in original
        for old in r["evidence_refs"]
    )


def select_route(
    snapshot: ContextSnapshot,
    resolution: ResolutionResult,
    classification: ClassificationResult,
) -> RouteSelection:
    requests = classification.requests
    order = sorted(
        range(len(requests)), key=lambda i: request_position(requests[i], snapshot)
    )
    priorities = [
        r.explicit_priority for r in requests if r.explicit_priority is not None
    ]
    for r in requests:
        if r.explicit_priority is not None and not r.priority_evidence_refs:
            raise UnderstandingError("priority_evidence_invalid")
    conflict = len(priorities) != len(set(priorities))
    if priorities and not conflict:
        order.sort(
            key=lambda i: (
                requests[i].explicit_priority is None,
                requests[i].explicit_priority or 0,
                request_position(requests[i], snapshot),
            )
        )
    primary = order[0]
    request = requests[primary]
    questions = [a.question for a in request.ambiguities]
    for unresolved in resolution.unresolved_references:
        if any(
            ref.kind == "input"
            and ref.source_id == "current"
            and ref.span
            and ref.span.start < unresolved.span.end
            and unresolved.span.start < ref.span.end
            for ref in request.evidence_refs
        ):
            questions.append(unresolved.question)
    if conflict:
        questions.append("请明确这些诉求的处理先后顺序。")
    if request.intent is None and not questions:
        raise UnderstandingError("classification_ambiguity_missing")
    task_id = referenced_task(request, snapshot)
    if task_id is not None and task_id != resolution.continuation_selection:
        raise UnderstandingError("continuation_conflict")
    return RouteSelection(
        primary_index=primary,
        secondary_indexes=[
            i for i in order[1:] if referenced_task(requests[i], snapshot) is None
        ],
        branch=INTENT_BRANCHES_V1.get(request.intent),
        route_ready=not questions,
        clarification_question=" ".join(dict.fromkeys(questions)) or None,
        continuation_task_id=task_id,
    )
