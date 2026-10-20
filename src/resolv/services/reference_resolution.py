"""Validate resolution provenance. Semantic accuracy is measured by real gold cases."""

import re
from difflib import SequenceMatcher
from itertools import pairwise

from resolv.schemas.understanding import (
    EvidenceRef,
    ResolutionResult,
    ResolutionStageResult,
    SourceFingerprint,
    UnderstandingError,
)
from resolv.services.understanding_safety import validate_request_semantics

BUSINESS_ID = re.compile(
    r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*[0-9])"
    r"[A-Za-z0-9][A-Za-z0-9_-]*(?![A-Za-z0-9_-])"
    r"|(?<![A-Za-z0-9_-])[0-9]{3,}(?![A-Za-z0-9_-])"
)


def contains_entity(text, value):
    return (
        re.search(r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])", text)
        is not None
    )


def active_business_ids(text):
    return [
        m.group()
        for m in BUSINESS_ID.finditer(text)
        if not re.search(
            r"不是(?:订单|商品|退款|运单)?\s*$",
            text[max(0, m.start() - 12) : m.start()],
        )
    ]


class ReferenceResolver:
    def __init__(self, model, context):
        self.model, self.context = model, context

    def resolve(self, snapshot, source_loader, audit_hook):
        messages, estimate = self.context.messages(snapshot, "resolution")
        output, audit = self.model.invoke(
            "resolution",
            messages,
            ResolutionResult,
            estimate,
            snapshot.config.reserved_output_tokens,
            audit_hook,
            reasoning_effort=snapshot.config.reasoning_effort,
            enable_thinking=snapshot.config.enable_thinking,
            thinking_budget=snapshot.config.thinking_budget,
            top_p=snapshot.config.top_p,
            output_format=snapshot.config.output_format,
            output_schema=self.context.output_schema(snapshot, "resolution"),
        )
        try:
            verified = self.validate(snapshot, output, source_loader)
        except UnderstandingError:
            audit_hook(audit.model_copy(update={"outcome": "failed"}))
            raise
        audit = audit.model_copy(update={"verified_refs": list(verified.values())})
        audit_hook(audit)
        return ResolutionStageResult(output=output, audit=audit)

    def validate(self, snapshot, output, source_loader):
        """Recheck a saved public stage output; this method never invokes a model."""
        verified = {}
        selected = next(
            (
                task
                for task in snapshot.pending_requests
                if task.task_id == output.continuation_selection
            ),
            None,
        )
        # A continued task owns its original speech act. Other clauses from its
        # parent input (for example an earlier logistics query) are provenance,
        # not constraints that turn this task's explicit cancel into a query.
        linked_actions = (
            [selected.text]
            if selected is not None
            else list(snapshot.linked_inputs.values())
        )
        validate_request_semantics(
            output.resolved_text,
            [snapshot.original_text, *linked_actions],
        )
        spans = sorted((b.span.start, b.span.end) for b in output.bindings)
        if any(left[1] > right[0] for left, right in pairwise(spans)):
            raise UnderstandingError("source_invalid", True)
        edits = SequenceMatcher(
            None, snapshot.original_text, output.resolved_text, autojunk=False
        ).get_opcodes()
        for binding in output.bindings:
            if binding.value in {t.task_id for t in snapshot.pending_requests}:
                raise UnderstandingError("source_invalid", True)
            if binding.span.end > len(snapshot.original_text):
                raise UnderstandingError("source_invalid", True)
            fragment = snapshot.original_text[binding.span.start : binding.span.end]
            if not contains_entity(fragment, binding.value) and not re.search(
                r"它|他|她|这|那|其|该|刚才|之前|上述|前者|后者|我的|订单|商品|继续",
                fragment,
            ):
                raise UnderstandingError("source_invalid", True)
            # Bind the Python range to its actual replacement, not to an
            # occurrence of the same entity elsewhere in the rewritten text.
            adjacent = re.match(
                r"(?:的)?(?:退款|订单|商品|包裹|运单)",
                snapshot.original_text[binding.span.end :],
            )
            replacement_end = binding.span.end + (adjacent.end() if adjacent else 0)
            replacements = " ".join(
                output.resolved_text[j1:j2]
                for tag, i1, i2, j1, j2 in edits
                if tag != "equal"
                and (
                    i1 < replacement_end
                    and binding.span.start < i2
                    or i1 == i2
                    and binding.span.start <= i1 <= replacement_end
                )
            )
            if not contains_entity(fragment, binding.value) and not contains_entity(
                replacements, binding.value
            ):
                raise UnderstandingError("source_invalid", True)
            # A later correction for another goal cannot change an earlier
            # reference. Current explicit order mentions apply forward only.
            mentions = [
                m.group()
                for m in BUSINESS_ID.finditer(
                    snapshot.original_text[: binding.span.start]
                )
                if re.search(
                    r"(?:订单|换成|改成)\s*$",
                    snapshot.original_text[max(0, m.start() - 10) : m.start()],
                )
            ]
            if (
                binding.entity_type == "order"
                and mentions
                and binding.value != mentions[-1]
            ):
                raise UnderstandingError("source_invalid", True)
            entity_proved = False
            for ref in binding.evidence_refs:
                evidence = self.context.verify(ref, snapshot, source_loader)
                proves_entity = contains_entity(ref.quote, binding.value)
                if ref.kind == "task":
                    task = next(
                        t
                        for t in snapshot.pending_requests
                        if t.task_id == ref.source_id
                    )
                    if ref.json_path[0] not in {"text", "bindings"} or (
                        ref.json_path == ["text"]
                        and ref.quote == binding.value
                        and not BUSINESS_ID.fullmatch(binding.value)
                        and not any(b.value == binding.value for b in task.bindings)
                    ):
                        proves_entity = False
                entity_proved |= proves_entity
                verified[(evidence.kind, evidence.source_id)] = SourceFingerprint(
                    kind=evidence.kind,
                    source_id=evidence.source_id,
                    source_hash=evidence.source_hash,
                )
                if ref.kind == "summary":
                    proof = self.verify_summary_entity(
                        binding.value, snapshot, source_loader
                    )
                    verified[(proof.kind, proof.source_id)] = proof
            if not entity_proved:
                raise UnderstandingError("source_invalid", True)
            if not contains_entity(output.resolved_text, binding.value):
                raise UnderstandingError("source_invalid", True)
        for unresolved in output.unresolved_references:
            if unresolved.span.end > len(snapshot.original_text):
                raise UnderstandingError("source_invalid", True)
            for ref in unresolved.candidate_refs:
                self.context.verify(ref, snapshot, source_loader)
        if (
            output.continuation_selection is not None
            and output.continuation_selection
            not in {t.task_id for t in snapshot.pending_requests}
        ):
            raise UnderstandingError("continuation_conflict", True)
        allowed = (
            snapshot.original_text + " " + " ".join(snapshot.linked_inputs.values())
        )
        allowed += " " + " ".join(b.value for b in output.bindings)
        allowed += " " + " ".join(t.task_id for t in snapshot.pending_requests)
        if any(
            not contains_entity(allowed, id)
            for id in BUSINESS_ID.findall(output.resolved_text)
        ):
            raise UnderstandingError("source_invalid", True)
        return verified

    def verify_summary_entity(self, value, snapshot, loader):
        for item in snapshot.source_fingerprints:
            if item.kind not in {"message", "action"}:
                continue
            ref = EvidenceRef(
                kind=item.kind,
                source_id=item.source_id,
                span=None,
                json_path=[],
                quote=value,
            )
            evidence = loader(ref)
            if evidence is None or evidence.source_hash != item.source_hash:
                continue
            if evidence.text is not None:
                found = contains_entity(evidence.text, value)
            else:

                def values(node):
                    if isinstance(node, dict):
                        return any(values(v) for v in node.values())
                    if isinstance(node, list):
                        return any(values(v) for v in node)
                    return contains_entity(str(node), value)

                found = values(evidence.json_value)
            if found:
                return item
        raise UnderstandingError("source_invalid", True)
