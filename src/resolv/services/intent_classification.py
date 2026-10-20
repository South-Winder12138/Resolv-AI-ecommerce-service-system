"""Strict model classification with anchored source, priority and binding checks."""

import re

from resolv.schemas.understanding import (
    ClassificationResult,
    IntentStageResult,
    SourceFingerprint,
    UnderstandingError,
)
from resolv.services.reference_resolution import BUSINESS_ID, active_business_ids
from resolv.services.understanding_safety import validate_request_semantics
from resolv.understanding_rules import (
    preserved_parent_task,
    referenced_task,
    request_position,
)


def action_type(text):
    if re.search(r"进度|到账|状态|结果", text):
        return "progress"
    if re.search(r"条件|能否|能退|资格|符合", text):
        return "eligibility"
    if re.search(r"申请|办理|退掉|我要退", text):
        return "application"
    return None


def continuation_candidates(snapshot, request, intent):
    candidates = [t for t in snapshot.pending_requests if t.intent in {intent, None}]
    # Task evidence describes a candidate; it cannot make that candidate unique.
    current = " ".join(
        ref.quote
        for ref in request.evidence_refs
        if ref.kind == "input" and ref.source_id == "current"
    )
    ids = set(BUSINESS_ID.findall(current))
    if ids:

        def objects(task):
            return set(BUSINESS_ID.findall(task.text)) | {
                b.value for b in task.bindings
            }

        matches = [t for t in candidates if ids <= objects(t)]
        # An unresolved object can be clarified, but cannot compete with an
        # already matching explicit object when selecting among multiple tasks.
        candidates = matches or [t for t in candidates if not objects(t)]
    kind = action_type(current)
    if kind:
        candidates = [t for t in candidates if action_type(t.text) in {kind, None}]
    return candidates


def validate_classification(snapshot, resolution, classification):
    seen_tasks = set()
    for request in classification.requests:
        request_position(request, snapshot)
        others = [
            ref
            for other in classification.requests
            if other is not request
            for ref in other.evidence_refs
        ]
        if len(classification.requests) > 1 and not any(
            ref.kind == "task"
            and not any(
                r.kind == "task" and r.source_id == ref.source_id for r in others
            )
            or ref.kind in {"input", "understanding"}
            and ref.span
            and not any(
                r.kind == ref.kind
                and r.source_id == ref.source_id
                and r.span
                and r.span.start < ref.span.end
                and ref.span.start < r.span.end
                for r in others
            )
            for ref in request.evidence_refs
        ):
            raise UnderstandingError("classification_evidence_invalid", True)
        for ref in [*request.evidence_refs, *request.priority_evidence_refs]:
            if ref.kind not in {"input", "understanding", "task"}:
                raise UnderstandingError("classification_evidence_invalid", True)
            if ref.kind in {"input", "understanding"}:
                text = (
                    snapshot.original_text
                    if ref.kind == "input" and ref.source_id == "current"
                    else snapshot.linked_inputs.get(ref.source_id)
                )
                if (
                    text is None
                    or not ref.span
                    or ref.json_path
                    or ref.span.end > len(text)
                    or text[ref.span.start : ref.span.end] != ref.quote
                ):
                    raise UnderstandingError("classification_evidence_invalid", True)
        if request.intent is None and not request.ambiguities:
            raise UnderstandingError("classification_ambiguity_missing", True)
        if (
            request.intent is not None
            and request.ambiguities
            and not any(
                ref.kind == "input"
                and ref.source_id == "current"
                and ref.span
                and ref.span.start < unresolved.span.end
                and unresolved.span.start < ref.span.end
                for ref in request.evidence_refs
                for unresolved in resolution.unresolved_references
            )
        ):
            # Object/task uncertainty belongs to the saved resolver output.
            # A known category cannot initiate future business parameter intake.
            # Uncertain intent must be represented by null; priority conflicts
            # are handled by the deterministic selector.
            raise UnderstandingError("classification_ambiguity_invalid", True)
        if len(request.binding_indexes) != len(set(request.binding_indexes)) or any(
            i < 0 or i >= len(resolution.bindings) for i in request.binding_indexes
        ):
            raise UnderstandingError("binding_index_invalid", True)
        # Each object must belong to this action, not merely occur somewhere in
        # the conversation or in a different request from the same input.
        action_refs = [
            r
            for r in request.evidence_refs
            if r.kind in {"input", "understanding", "task"}
        ]
        allowed = {
            value for ref in action_refs for value in active_business_ids(ref.quote)
        }
        validate_request_semantics(request.text, [ref.quote for ref in action_refs])
        for i in request.binding_indexes:
            span = resolution.bindings[i].span
            if not any(
                ref.kind == "input"
                and ref.source_id == "current"
                and ref.span
                and ref.span.start < span.end
                and span.start < ref.span.end
                for ref in action_refs
            ):
                raise UnderstandingError("classification_entity_invalid", True)
        allowed.update(resolution.bindings[i].value for i in request.binding_indexes)
        allowed.update(ref.source_id for ref in action_refs if ref.kind == "task")
        if any(value not in allowed for value in active_business_ids(request.text)):
            raise UnderstandingError("classification_entity_invalid", True)
        if request.explicit_priority is not None:
            if not request.priority_evidence_refs or not any(
                re.search(
                    r"先|再|最后|优先|首先|然后|接着|第[一二三四五六七八九十0-9]+|只",
                    ref.quote,
                )
                for ref in request.priority_evidence_refs
            ):
                raise UnderstandingError("priority_evidence_invalid", True)
        elif request.priority_evidence_refs:
            raise UnderstandingError("priority_evidence_invalid", True)
        task_id = referenced_task(request, snapshot)
        if task_id:
            preserved = preserved_parent_task(request, snapshot)
            if task_id in seen_tasks or (
                task_id != resolution.continuation_selection and not preserved
            ):
                raise UnderstandingError("continuation_conflict", True)
            seen_tasks.add(task_id)
            task = next(t for t in snapshot.pending_requests if t.task_id == task_id)
            if task.intent is not None and request.intent != task.intent:
                raise UnderstandingError("continuation_conflict", True)
            if not preserved:
                candidates = continuation_candidates(snapshot, request, request.intent)
                if len(candidates) != 1 or candidates[0].task_id != task_id:
                    raise UnderstandingError("continuation_conflict", True)
    if (
        resolution.continuation_selection is not None
        and resolution.continuation_selection not in seen_tasks
    ):
        raise UnderstandingError("continuation_conflict", True)


class IntentClassifier:
    def __init__(self, model, context):
        self.model, self.context = model, context

    def classify(self, snapshot, resolution, source_loader, audit_hook):
        messages, estimate = self.context.messages(snapshot, "intents", resolution)
        output, audit = self.model.invoke(
            "intents",
            messages,
            ClassificationResult,
            estimate,
            snapshot.config.reserved_output_tokens,
            audit_hook,
            reasoning_effort=snapshot.config.reasoning_effort,
            enable_thinking=snapshot.config.enable_thinking,
            thinking_budget=snapshot.config.thinking_budget,
            top_p=snapshot.config.top_p,
            output_format=snapshot.config.output_format,
            output_schema=self.context.output_schema(snapshot, "intents", resolution),
        )
        try:
            verified = self.validate(snapshot, resolution, output, source_loader)
        except UnderstandingError:
            audit_hook(audit.model_copy(update={"outcome": "failed"}))
            raise
        audit = audit.model_copy(update={"verified_refs": list(verified.values())})
        audit_hook(audit)
        return IntentStageResult(output=output, audit=audit)

    def validate(self, snapshot, resolution, output, source_loader):
        """Recheck a saved classification without inference or business execution."""
        verified = {}
        validate_classification(snapshot, resolution, output)
        for request in output.requests:
            refs = [
                *request.evidence_refs,
                *request.priority_evidence_refs,
                *(ref for a in request.ambiguities for ref in a.evidence_refs),
            ]
            for ref in refs:
                s = self.context.verify(ref, snapshot, source_loader)
                verified[(s.kind, s.source_id)] = SourceFingerprint(
                    kind=s.kind, source_id=s.source_id, source_hash=s.source_hash
                )
        return verified
