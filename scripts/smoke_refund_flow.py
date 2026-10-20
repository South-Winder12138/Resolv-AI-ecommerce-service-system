"""Explicit R2 gates on refund-owned resources; no customer or old Gate writes."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
from datetime import UTC, datetime
from hashlib import sha256

from scripts.refund_gate_support import (
    ROOT,
    WORKSPACE,
    finish_refund_call,
    reserve_refund_call,
)

from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_submission import ReplyMetadataV3

CASES = "tests/fixtures/refund_flow/semantic_cases.json"
POLICY_COMMIT = "abf6c077a15c6055de39edf6a4d20e045a39fcea"
POLICY_FIXTURE = "tests/fixtures/refund_flow/returns-policy.md"
POLICY_SHA256 = "fd9543748df085444df5ffddec2191500aa26ba7260038c02c177f954d11870f"
CASES_SHA256 = "051ed01496c2ca5bede6d2e2f5d66b41a12c365a1133c499044a65302cbb405a"
NOW = datetime(2026, 10, 16, 4, tzinfo=UTC)
# Frozen source set: later unrelated file additions do not alter prior gate evidence.
SOURCES = (
    "scripts/refund_native_regression.py",
    "scripts/refund_acceptance_v2.py",
    "src/resolv/core/refund_native.py",
    "tests/test_refund_native_return.py",
    "tests/test_refund_native_application.py",
    "tests/test_refund_native_regression.py",
    "tests/test_refund_acceptance_v2.py",
    "tests/fixtures/refund_flow/semantic_cases-v2.json",
    "tests/fixtures/refund_flow/explicit-time-policy.md",
    "src/resolv/core/refund_response.py",
    "src/resolv/core/refund_diagnostics.py",
    "scripts/refund_failure_storage.py",
    "tests/test_refund_model_diagnostics.py",
    "tests/test_refund_failure_storage.py",
    "tests/test_refund_output_protocol.py",
    "tests/test_refund_context_identity.py",
    "tests/test_refund_output_regression.py",
    "scripts/refund_round_regression.py",
    "tests/test_refund_round_regression.py",
    "tests/test_refund_round_policy.py",
    "scripts/retry_refund_flow_c3.py",
    "src/resolv/core/context_budget.py",
    "scripts/refund_continuation.py",
    "scripts/refund_semantic_continuation.py",
    "tests/refund_fixtures.py",
    "scripts/smoke_refund_flow.py",
    "scripts/refund_gate_support.py",
    "tests/fixtures/refund_flow/semantic_cases.json",
    "tests/fixtures/refund_flow/returns-policy.md",
    "pyproject.toml",
    "uv.lock",
    "knowledge/file-config.formal.toml",
    "src/resolv/__init__.py",
    "src/resolv/after_sales.py",
    "src/resolv/agent_events.py",
    "src/resolv/agent_loop.py",
    "src/resolv/api/__init__.py",
    "src/resolv/api/agent.py",
    "src/resolv/api/chat.py",
    "src/resolv/api/conversations.py",
    "src/resolv/api/refunds.py",
    "src/resolv/api/sse.py",
    "src/resolv/api/understandings.py",
    "src/resolv/approval.py",
    "src/resolv/approval_store.py",
    "src/resolv/chat.py",
    "src/resolv/conversation.py",
    "src/resolv/conversation_graph.py",
    "src/resolv/core/__init__.py",
    "src/resolv/core/business_model.py",
    "src/resolv/core/config.py",
    "src/resolv/core/hybrid_knowledge_config.py",
    "src/resolv/core/knowledge_branch_config.py",
    "src/resolv/core/knowledge_branch_model.py",
    "src/resolv/core/knowledge_branch_prompts.py",
    "src/resolv/core/llm.py",
    "src/resolv/core/model_stream.py",
    "src/resolv/core/prompts.py",
    "src/resolv/core/refund_config.py",
    "src/resolv/core/refund_model.py",
    "src/resolv/core/refund_prompts.py",
    "src/resolv/core/understanding_model.py",
    "src/resolv/core/understanding_prompts.py",
    "src/resolv/db/__init__.py",
    "src/resolv/db/base.py",
    "src/resolv/db/checkpoints.py",
    "src/resolv/db/conversation.py",
    "src/resolv/db/conversation_base.py",
    "src/resolv/db/engine.py",
    "src/resolv/db/exact_json.py",
    "src/resolv/db/init_db.py",
    "src/resolv/db/knowledge.py",
    "src/resolv/db/knowledge_base.py",
    "src/resolv/db/knowledge_hybrid.py",
    "src/resolv/db/knowledge_index.py",
    "src/resolv/db/milvus.py",
    "src/resolv/knowledge.py",
    "src/resolv/knowledge_routing_graph.py",
    "src/resolv/main.py",
    "src/resolv/models/__init__.py",
    "src/resolv/models/conversation.py",
    "src/resolv/models/knowledge.py",
    "src/resolv/models/knowledge_branch.py",
    "src/resolv/models/knowledge_hybrid.py",
    "src/resolv/models/order.py",
    "src/resolv/models/refund.py",
    "src/resolv/models/refund_application.py",
    "src/resolv/models/refund_flow.py",
    "src/resolv/models/shipment.py",
    "src/resolv/models/tool_execution.py",
    "src/resolv/models/understanding.py",
    "src/resolv/refund_routing_graph.py",
    "src/resolv/repositories/__init__.py",
    "src/resolv/repositories/conversation.py",
    "src/resolv/repositories/knowledge.py",
    "src/resolv/repositories/knowledge_branch.py",
    "src/resolv/repositories/knowledge_hybrid.py",
    "src/resolv/repositories/knowledge_hybrid_milvus.py",
    "src/resolv/repositories/knowledge_milvus.py",
    "src/resolv/repositories/milvus.py",
    "src/resolv/repositories/order.py",
    "src/resolv/repositories/refund.py",
    "src/resolv/repositories/refund_application.py",
    "src/resolv/repositories/refund_flow.py",
    "src/resolv/repositories/refund_submission.py",
    "src/resolv/repositories/shipment.py",
    "src/resolv/repositories/tool_execution.py",
    "src/resolv/repositories/understanding.py",
    "src/resolv/routing_graph.py",
    "src/resolv/schemas/__init__.py",
    "src/resolv/schemas/after_sales.py",
    "src/resolv/schemas/branch_execution.py",
    "src/resolv/schemas/conversation.py",
    "src/resolv/schemas/hybrid_knowledge.py",
    "src/resolv/schemas/knowledge.py",
    "src/resolv/schemas/knowledge_branch.py",
    "src/resolv/schemas/knowledge_gate_cases.py",
    "src/resolv/schemas/knowledge_release.py",
    "src/resolv/schemas/refund.py",
    "src/resolv/schemas/refund_flow.py",
    "src/resolv/schemas/refund_submission.py",
    "src/resolv/schemas/understanding.py",
    "src/resolv/services/__init__.py",
    "src/resolv/services/branch_evaluation.py",
    "src/resolv/services/branch_replies.py",
    "src/resolv/services/business_execution.py",
    "src/resolv/services/business_policy.py",
    "src/resolv/services/business_summary.py",
    "src/resolv/services/context.py",
    "src/resolv/services/conversation.py",
    "src/resolv/services/conversation_routing.py",
    "src/resolv/services/embedding.py",
    "src/resolv/services/hybrid_authority.py",
    "src/resolv/services/hybrid_embedding.py",
    "src/resolv/services/hybrid_fingerprints.py",
    "src/resolv/services/hybrid_indexing.py",
    "src/resolv/services/hybrid_manifest.py",
    "src/resolv/services/hybrid_ranking.py",
    "src/resolv/services/hybrid_reranker.py",
    "src/resolv/services/hybrid_retrieval.py",
    "src/resolv/services/hybrid_runtime.py",
    "src/resolv/services/hybrid_windows.py",
    "src/resolv/services/intent_classification.py",
    "src/resolv/services/knowledge.py",
    "src/resolv/services/knowledge_branch.py",
    "src/resolv/services/knowledge_branch_answer.py",
    "src/resolv/services/knowledge_branch_check.py",
    "src/resolv/services/knowledge_branch_evidence.py",
    "src/resolv/services/knowledge_branch_query.py",
    "src/resolv/services/knowledge_branch_runtime.py",
    "src/resolv/services/knowledge_branch_summary.py",
    "src/resolv/services/knowledge_chat.py",
    "src/resolv/services/knowledge_confidence.py",
    "src/resolv/services/knowledge_embedding.py",
    "src/resolv/services/knowledge_gate_validation.py",
    "src/resolv/services/knowledge_indexing.py",
    "src/resolv/services/knowledge_release.py",
    "src/resolv/services/knowledge_reply_projection.py",
    "src/resolv/services/knowledge_retrieval.py",
    "src/resolv/services/order.py",
    "src/resolv/services/reference_resolution.py",
    "src/resolv/services/refund.py",
    "src/resolv/services/refund_adapters.py",
    "src/resolv/services/refund_context.py",
    "src/resolv/services/refund_decision.py",
    "src/resolv/services/refund_evidence.py",
    "src/resolv/services/refund_flow.py",
    "src/resolv/services/refund_offer.py",
    "src/resolv/services/refund_query.py",
    "src/resolv/services/refund_registration.py",
    "src/resolv/services/refund_reply_projection.py",
    "src/resolv/services/refund_revalidation.py",
    "src/resolv/services/refund_runtime.py",
    "src/resolv/services/refund_scope.py",
    "src/resolv/services/refund_submission.py",
    "src/resolv/services/refund_tools.py",
    "src/resolv/services/retrieval.py",
    "src/resolv/services/routed_runtime.py",
    "src/resolv/services/shipment.py",
    "src/resolv/services/summarization.py",
    "src/resolv/services/tool_execution.py",
    "src/resolv/services/understanding.py",
    "src/resolv/services/understanding_context.py",
    "src/resolv/services/understanding_evaluation.py",
    "src/resolv/services/understanding_execution.py",
    "src/resolv/services/understanding_runtime.py",
    "src/resolv/services/understanding_safety.py",
    "src/resolv/tool_calling.py",
    "src/resolv/tools/__init__.py",
    "src/resolv/tools/cancel.py",
    "src/resolv/tools/executor.py",
    "src/resolv/tools/logistics.py",
    "src/resolv/tools/order.py",
    "src/resolv/tools/refund.py",
    "src/resolv/tools/registry.py",
    "src/resolv/understanding_graph.py",
    "src/resolv/understanding_rules.py",
)
FORMAL_QUERIES = (
    "普通商品完好可二次销售，签收两天无理由退货的适用条件和例外是什么？",
    "生鲜商品非质量问题能否无理由退货？",
    "无理由退货和商品质量问题退货的退回运费分别由谁承担？",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_cases(value):
    require(
        value.get("schema_version") == 1
        and value.get("policy_commit") == POLICY_COMMIT
        and value.get("clock") == NOW.isoformat().replace("+00:00", "Z")
        and value.get("timezone") == "Asia/Shanghai"
        and [c["case_id"] for c in value.get("cases", [])]
        == [f"F{i}" for i in range(1, 9)],
        "Fixed cases/clock/policy are required",
    )
    raw = (ROOT / POLICY_FIXTURE).read_bytes()
    require(sha256(raw).hexdigest() == POLICY_SHA256, "Frozen policy fixture changed")
    policies = raw.decode("utf-8")
    require(
        set(value.get("policy_quotes", {}))
        == {"ordinary", "exception", "warranty", "shipping"}
        and all(q and q in policies for q in value["policy_quotes"].values()),
        "Expectations require exact committed policy quotes",
    )
    required = {
        "verdict",
        "next_actions",
        "offer_count",
        "policy_quotes",
        "required_fact_fields",
        "reason_groups",
        "forbidden_patterns",
    }
    for case in value["cases"]:
        require(
            all(
                k in case
                for k in (
                    "question",
                    "order_id",
                    "subject_ref",
                    "facts",
                    "money",
                    "existing_application",
                    "expected",
                )
            )
            and bool(case["question"])
            and case["order_id"] in case["question"]
            and set(case["expected"]) == required
            and all(q in policies for q in case["expected"]["policy_quotes"]),
            "Case facts and independent expectations must be complete",
        )
    return value


def load_cases():
    raw = (ROOT / CASES).read_bytes()
    require(sha256(raw).hexdigest() == CASES_SHA256, "Fixed case bytes changed")
    return validate_cases(json.loads(raw))


class BudgetedRefundClient:
    """Reserve before the SDK; retain unknown usage even without a terminal audit."""

    def __init__(
        self, base, ownership, case_id, attempt, source_hash, *, continuation=None
    ):
        self.base, self.ownership, self.case_id, self.attempt, self.source_hash = (
            base,
            ownership,
            case_id,
            attempt,
            source_hash,
        )
        self.outputs, self.audits = [], []
        self.profiles = {}
        self.continuation = continuation

    def invoke(self, *, purpose, invoke, audit_hook=lambda _: None, input_hash=None):
        from time import monotonic

        begin = monotonic()
        reservation = reserve_refund_call(
            self.ownership,
            purpose=purpose,
            case_id=self.case_id,
            attempt=self.attempt,
            source_hash=self.source_hash,
            **({"continuation": self.continuation} if self.continuation else {}),
        )
        last = None
        started = False

        def record(audit):
            nonlocal last, started
            if audit.outcome == "started":
                require(not started, "One reservation cannot hide a second call")
                started = True
            last = (
                audit.model_dump(mode="json")
                | self.profiles.get(purpose, {})
                | {
                    "purpose": purpose,
                    "reservation_id": reservation.invocation_id,
                    "input_hash": getattr(audit, "input_hash", None) or input_hash,
                    "elapsed_seconds": monotonic() - begin,
                }
            )
            self.audits.append(last)
            audit_hook(audit)

        try:
            output = invoke(audit_hook=record)
            value = output[0] if isinstance(output, tuple) else output
            if hasattr(value, "model_dump"):
                self.outputs.append(
                    {
                        "purpose": purpose,
                        "output": value.model_dump(mode="json", exclude={"audit"}),
                    }
                )
            return output
        finally:
            last = last or {}
            finish_refund_call(
                reservation,
                outcome=last.get("outcome")
                if last.get("outcome") in {"succeeded", "failed"}
                else "unknown",
                input_tokens=last.get("input_tokens"),
                output_tokens=last.get("output_tokens"),
                input_hash=last.get("input_hash") or input_hash,
                error_code=last.get("error_code"),
            )

    def prepare(self, messages, *, profile, audit_hook):
        from resolv.services.hybrid_fingerprints import canonical_hash

        return self.invoke(
            purpose="refund_query",
            audit_hook=audit_hook,
            input_hash=canonical_hash([m.model_dump(mode="json") for m in messages]),
            invoke=lambda audit_hook: self.base.prepare(
                messages, profile=profile, audit_hook=audit_hook
            ),
        )

    def decide(self, messages, *, audit_hook, **kwargs):
        from resolv.services.hybrid_fingerprints import canonical_hash

        return self.invoke(
            purpose="refund_decision",
            audit_hook=audit_hook,
            input_hash=canonical_hash([m.model_dump(mode="json") for m in messages]),
            invoke=lambda audit_hook: self.base.decide(
                messages, audit_hook=audit_hook, **kwargs
            ),
        )

    @property
    def counter_factory(self):
        return self.base.counter_factory


def check_semantic_result(result, case, *, manual_review=None):
    failures = []

    def check(ok, label):
        if not ok:
            failures.append(label)

    expected = case["expected"]
    try:
        from resolv.schemas.refund_flow import RefundRunRecord
        from resolv.services.refund_decision import validate_refund_decision

        record = RefundRunRecord.model_validate_json(
            canonical_json(result["refund_record"])
        )
        check(
            result["binding"] == record.binding.model_dump(mode="json")
            and result["decision"] == record.decision.model_dump(mode="json")
            and result["observations"]
            == [o.model_dump(mode="json") for o in record.observations],
            "exact_saved_judgement_proof",
        )
        validate_refund_decision(
            record.decision,
            request=record.request,
            binding=record.binding,
            evidence=record.evidence,
            observations=record.observations,
            profile=record.profile,
        )
        meta = ReplyMetadataV3.model_validate_json(canonical_json(result["metadata"]))
        refund = meta.refund
        check(
            result["status"] == "completed"
            and meta.branch == "refund_flow"
            and refund is not None,
            "completed_refund_branch",
        )
        if refund is None:
            raise ValueError("Missing refund")
        check(
            refund.reason == record.decision.reason
            and refund.operation == record.decision.operation
            and refund.policy_citations == record.decision.policy_citations
            and refund.evidence == record.evidence
            and refund.reason in result["answer"],
            "same_model_response_and_evidence",
        )
        check(refund.order_id == case["order_id"], "current_order")
        check(refund.verdict == expected["verdict"], "verdict")
        check(refund.next_action in expected["next_actions"], "next_action")
        check(bool(refund.offer_id) == bool(expected["offer_count"]), "offer_count")
        check(result["legacy_tool_calls"] == 0, "no_legacy_tools")
        binding = result.get("binding") or {}
        check(binding.get("order_id") == case["order_id"], "bound_order")
        observations = result.get("observations", [])
        check(
            bool(observations)
            and all(
                o.get("source") == binding.get("order_source")
                and o.get("binding_hash") == binding.get("binding_hash")
                for o in observations
            ),
            "observed_current_order",
        )
        check(
            {"query_refund_order", "query_refund_status"}
            <= {o.get("tool_name") for o in observations},
            "required_reads",
        )
        for quote in expected["policy_quotes"]:
            check(
                any(quote in c.quote for c in refund.policy_citations),
                "required_policy_quote",
            )
        for group in expected["reason_groups"]:
            check(any(t in result["answer"] for t in group), "reason_facts")
        check(
            not any(
                re.search(p, result["answer"]) for p in expected["forbidden_patterns"]
            ),
            "no_funds_promise",
        )
        fields = {
            r["field"]
            for rule in (result.get("decision") or {}).get("rules", [])
            for r in rule.get("fact_refs", [])
        }
        check(set(expected["required_fact_fields"]) <= fields, "rule_fact_coverage")
        if case["case_id"] in {"F3", "F8"}:
            check(
                bool(refund.unverified_conditions) and bool(refund.missing_information),
                "user_claims_remain_unverified",
            )
        if case["case_id"] == "F7":
            check(
                any(
                    case["other_order_id"] in t.get("text", "")
                    and t.get("queue_status") == "pending"
                    for t in result.get("pending_requests", [])
                ),
                "other_order_remains_pending",
            )
        check(not result.get("unapproved_transport_attempts"), "only_approved_purposes")
        if case["case_id"] == "F6":
            check(
                any(
                    o.get("tool_name") == "query_refund_payment"
                    and o.get("fields", {})
                    .get("refundable_money", {})
                    .get("availability")
                    == "not_supported"
                    for o in observations
                ),
                "unsupported_money_observed",
            )
        if case["case_id"] == "F4":
            check(
                refund.application is not None
                and refund.application.application_id
                == case["existing_application"]["application_id"]
                and refund.application.status == "rejected",
                "original_rejected_application",
            )
        if expected["offer_count"]:
            check(
                result["application_count_before"] == 0, "no_application_before_click"
            )
            submitted = result.get("submission") or {}
            check(
                result["application_count_after"] == 1
                and submitted.get("receipt", {}).get("status") == "pending"
                and bool(submitted.get("receipt", {}).get("application_id")),
                "actual_pending_application",
            )
            check(
                result.get("duplicate_receipt") == submitted.get("receipt"),
                "same_duplicate_receipt",
            )
        else:
            check(
                result["application_count_after"] == result["application_count_before"],
                "no_new_application",
            )
    except (ValueError, KeyError, TypeError, AttributeError, RefundFlowError):
        failures.append("incomplete_or_invalid_public_proof")
    machine_passed = not failures
    reviewed = bool(
        manual_review
        and manual_review.get("case_id") == case["case_id"]
        and manual_review.get("approved") is True
        and manual_review.get("checks")
        and all(manual_review["checks"].values())
        and manual_review.get("notes")
    )
    return {
        "passed": machine_passed and reviewed,
        "machine_passed": machine_passed,
        "manual_review_required": not reviewed,
        "failures": failures,
    }


def validate_forward_receipt(r):
    require(
        r.get("model_class")
        == "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaForSequenceClassification"
        and r.get("device") in {"mps", "mps:0"}
        and r.get("dtype") == "torch.float32"
        and type(r.get("batch_size")) is int
        and 1 <= r["batch_size"] <= 4
        and r.get("shape") == [r["batch_size"], 1]
        and r.get("finite") is True
        and all(
            re.fullmatch("[0-9a-f]{64}", r.get(k, ""))
            for k in ("input_hash", "output_hash")
        ),
        "Real MPS forward receipt required",
    )


def validate_retrieval_gate(report):
    require(
        report.get("passed")
        and report.get("source_fingerprints")
        and report.get("formal_preserved")
        and report.get("collection_identity_verified"),
        "Incomplete R2 retrieval authority",
    )
    queries = report.get("queries", [])
    require(
        len(queries) == 3 and [q["query"] for q in queries] == list(FORMAL_QUERIES),
        "Fixed formal queries required",
    )
    for q in queries:
        d = q["diagnostics"]
        require(
            all(
                d.get(k, 0) > 0
                for k in (
                    "dense_count",
                    "bm25_count",
                    "center_count",
                    "scored_count",
                    "selected_count",
                )
            )
            and q["evidence_tokens"] <= 3000
            and q["evidence_tokens"] > 0,
            "Actual closed hybrid evidence required",
        )
        receipts = q.get("forward_receipts", [])
        require(
            receipts and sum(r["batch_size"] for r in receipts) == d["scored_count"],
            "Forward count differs from scored windows",
        )
        for receipt in receipts:
            validate_forward_receipt(receipt)
    require(
        set(report.get("probes", {}))
        == {"one_lane_empty", "bm25_fault", "authority_mismatch"}
        and all(p.get("passed") for p in report["probes"].values()),
        "All owned boundary probes required",
    )


class IsolatedRefundFacts:
    """Only fixed owned test identities; never installed by the application."""

    def __init__(self, case, owner):
        from resolv.schemas.refund import AdapterDescriptor

        self.case, self.calls = case, []
        self.descriptor = AdapterDescriptor(
            namespace="refund-gate-" + owner[:16],
            implementation_version="fixed-cases-v1",
            authorization_version="fixed-cases-v1",
            connected=True,
        )

    def _observation(self, name, binding, call_id, fields):
        from uuid import uuid4

        from resolv.schemas.refund import RefundObservation
        from resolv.services.hybrid_fingerprints import canonical_hash

        require(
            binding.order_source == self.descriptor.namespace
            and binding.order_id == self.case["order_id"],
            "Fixture cannot read another order",
        )
        self.calls.append(
            {"tool": name, "order_id": binding.order_id, "call_id": call_id}
        )
        fields = {
            k: v
            if isinstance(v, dict) and "availability" in v
            else {"availability": "available", "value": v}
            for k, v in fields.items()
        }
        return RefundObservation.create(
            observation_id=uuid4().hex,
            tool_call_id=call_id,
            tool_name=name,
            binding_hash=binding.binding_hash,
            source=binding.order_source,
            observed_at=NOW,
            provider_version="fixed-cases-v1",
            arguments_hash=canonical_hash({}),
            availability="available",
            fields=fields,
        )

    def authorize(self, access, binding):
        require(
            access.subject_ref == self.case["subject_ref"]
            and access.namespace == self.descriptor.namespace,
            "Fixture identity mismatch",
        )
        return self._observation(
            "query_refund_order", binding, "authorization", {"authorized": True}
        )

    def query_order(self, access, binding, call_id):
        return self._observation(
            "query_refund_order",
            binding,
            call_id,
            {"order_id": binding.order_id, **self.case["facts"]},
        )

    def query_payment(self, access, binding, call_id):
        return self._observation(
            "query_refund_payment",
            binding,
            call_id,
            {"refundable_money": self.case["money"]},
        )

    def query_delivery(self, access, binding, call_id):
        return self._observation(
            "query_refund_delivery",
            binding,
            call_id,
            {k: self.case["facts"][k] for k in ("delivered_at",)},
        )

    def query_status(self, access, binding, call_id):
        return self._observation("query_refund_status", binding, call_id, {})


def ensure_services(ownership, target):
    """One explicitly approved start of the existing stopped Milvus container."""
    import subprocess
    import time

    from scripts.refund_gate_support import save_json, validate_manifest

    validate_manifest(ownership)
    receipt = Path(ownership).with_name("milvus-start.json")

    def inspect():
        p = subprocess.run(
            [
                "docker",
                "inspect",
                "resolv-rag-standalone-1",
                "--format",
                "{{json .State}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        require(p.returncode == 0, "Existing Milvus container is required")
        return json.loads(p.stdout)

    state = inspect()
    if not state.get("Running"):
        require(
            not receipt.exists() and state.get("ExitCode") == 80,
            "Approved single Milvus start unavailable",
        )
        save_json(
            receipt,
            {
                "attempted": True,
                "container": "resolv-rag-standalone-1",
                "old_exit_code": 80,
            },
        )
        p = subprocess.run(
            ["docker", "compose", "-f", "compose.milvus.yaml", "start", "standalone"],
            cwd=ROOT,
            capture_output=True,
            timeout=45,
            check=False,
        )
        require(p.returncode == 0, "Existing Milvus start failed")
    begin = time.monotonic()
    while time.monotonic() - begin < 120:
        state = inspect()
        if state.get("Running") and state.get("Health", {}).get("Status") == "healthy":
            result = {
                "gate": "R0",
                "passed": True,
                "container": "resolv-rag-standalone-1",
                "started_by_this_task": receipt.exists(),
                "health": "healthy",
                "elapsed_seconds": time.monotonic() - begin,
            }
            save_json(target / "R0.json", result)
            return result
        if not state.get("Running"):
            break
        time.sleep(2)
    save_json(
        target / "R0.json",
        {
            "gate": "R0",
            "passed": False,
            "error_code": "milvus_not_healthy",
            "exit_code": state.get("ExitCode"),
        },
    )
    raise ValueError("Milvus not healthy after the approved bounded start")


def owned_policy(obj):
    from resolv.schemas.hybrid_knowledge import HybridIndexProfile
    from resolv.schemas.knowledge import Document
    from resolv.services.knowledge_release import (
        _committed_manifest,
        _decode,
        release_corpus,
    )

    manifest, blobs, _ = _committed_manifest(POLICY_COMMIT)
    documents, specs = [], []
    release_id = "refund_20261016_" + obj["owner"][:16]
    for spec in manifest.documents:
        raw = blobs[spec.source]
        if spec.original_path == "knowledge/returns-policy.md":
            require(
                raw == (ROOT / POLICY_FIXTURE).read_bytes(),
                "Frozen policy differs from Git",
            )
        source = (
            "knowledge/releases/" + release_id + "/" + Path(spec.original_path).name
        )
        did = sha256(source.encode()).hexdigest()
        documents.append(Document(document_id=did, source=source, text=_decode(raw)))
        specs.append(spec.model_copy(update={"source": source, "document_id": did}))
    manifest = manifest.model_copy(
        update={
            "release_id": release_id,
            "index_version": obj["collection"],
            "collection": obj["collection"],
            "index_profile": HybridIndexProfile(collection_name=obj["collection"]),
            "documents": tuple(specs),
        }
    )
    corpus = release_corpus(manifest, tuple(documents))
    require(
        len(documents) == 6 and sum(len(d.chunks) for d in corpus.documents) == 51,
        "Six fixed documents / 51 chunks required",
    )
    return manifest, tuple(documents), corpus


def prepare_owned(ownership, target, *, continue_intent=None):
    from contextlib import ExitStack

    from scripts.refund_gate_support import (
        manifest_lock,
        owned_refund_resources,
        save_json,
    )

    from resolv.core.config import MilvusSettings
    from resolv.db.knowledge_hybrid import hybrid_index_lock
    from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, milvus_client
    from resolv.repositories.knowledge_hybrid_milvus import HybridMilvusStore
    from resolv.services.hybrid_embedding import HybridEmbeddingAdapter
    from resolv.services.hybrid_fingerprints import (
        canonical_hash,
        manifest_hash,
        profile_hash,
    )
    from resolv.services.hybrid_indexing import HybridKnowledgeIndexer
    from resolv.services.knowledge import KnowledgeIngestionService
    from resolv.services.knowledge_release import (
        _PreflightStore,
        preflight_release,
        staging_profile,
    )

    with owned_refund_resources(ownership) as resources, ExitStack() as stack:
        obj = resources.identity
        manifest, documents, corpus = owned_policy(obj)
        client = stack.enter_context(milvus_client(MilvusSettings()))
        require(
            not client.has_collection(
                collection_name=obj["collection"], timeout=MILVUS_TIMEOUT_SECONDS
            ),
            "Refusing an existing unclaimed collection",
        )
        with manifest_lock(ownership) as current:
            if continue_intent is None:
                require(
                    "collection_build" not in current,
                    "An attempted build is not automatically repeated",
                )
            else:
                require(
                    current.get("collection_build") == "started"
                    and current.get("prepare_continuation", {}).get("prior_report_hash")
                    == continue_intent,
                    "Empty-intent continuation receipt required",
                )
            current["collection_build"] = "started"
            save_json(ownership, current)
        lock = stack.enter_context(
            hybrid_index_lock(resources.knowledge_engine, manifest.index_version)
        )
        preflight_release(
            resources.knowledge_factory,
            manifest,
            documents,
            _PreflightStore(client, manifest.collection),
        )
        ingestion = KnowledgeIngestionService(resources.knowledge_factory)
        for doc, spec in zip(documents, manifest.documents, strict=True):
            ingestion.stage_document(
                doc,
                spec.file_config,
                staging_profile(manifest),
                max_chars=spec.max_chars,
                overlap=spec.overlap,
            )
        dense = HybridEmbeddingAdapter(manifest.index_profile)
        stack.callback(dense.close)
        build = HybridKnowledgeIndexer(
            resources.knowledge_factory,
            dense,
            HybridMilvusStore(client, manifest.index_profile),
            manifest.index_profile,
            lock,
        )
        build.prepare(manifest.index_version, corpus)
        result = build.run(manifest.index_version)
        require(
            result.summary.state == "ready" and result.summary.done_count == 51,
            "Owned index not ready",
        )
        desc = client.describe_collection(
            collection_name=obj["collection"], timeout=MILVUS_TIMEOUT_SECONDS
        )
        require(desc.get("collection_id") is not None, "Collection ID is required")
        HybridMilvusStore(client, manifest.index_profile).validate_existing()
        with manifest_lock(ownership) as current:
            current.update(
                collection_build="ready",
                collection_id=str(desc["collection_id"]),
                collection_profile_hash=profile_hash(manifest.index_profile),
                collection_manifest_hash=manifest_hash(corpus),
                release_manifest_hash=canonical_hash(manifest.model_dump(mode="json")),
            )
            save_json(ownership, current)
        return {
            "passed": True,
            "documents": 6,
            "chunks": 51,
            "build": result.model_dump(mode="json"),
            "collection_id": str(desc["collection_id"]),
            "dense_indexing_batch_count": result.batch_count,
            "policy_commit": POLICY_COMMIT,
        }


def owned_selection(resources):
    from resolv.schemas.knowledge_release import ReleaseSelection
    from resolv.services.hybrid_fingerprints import (
        canonical_hash,
        manifest_hash,
        profile_hash,
    )

    manifest, _, corpus = owned_policy(resources.identity)
    obj = resources.identity
    require(
        obj.get("collection_build") == "ready"
        and obj.get("collection_id")
        and obj.get("collection_profile_hash") == profile_hash(manifest.index_profile)
        and obj.get("collection_manifest_hash") == manifest_hash(corpus)
        and obj.get("release_manifest_hash")
        == canonical_hash(manifest.model_dump(mode="json")),
        "Owned collection receipt differs from fixed policy",
    )
    return ReleaseSelection(
        status="configured",
        release_id=manifest.release_id,
        index_version=manifest.index_version,
        collection=manifest.collection,
        profile=manifest.index_profile,
        profile_hash=profile_hash(manifest.index_profile),
        manifest_hash=manifest_hash(corpus),
    )


def verify_collection(pool, obj, selection):
    pool.validate_selection(selection)
    desc = pool._client.describe_collection(
        collection_name=selection.collection, timeout=15
    )
    require(
        str(desc.get("collection_id")) == obj["collection_id"]
        and pool._factory.kw["bind"].url.database == obj["knowledge_database"],
        "Owned collection ID or database changed",
    )


class ForwardRecorder:
    def __init__(self, ranker):
        self.records = []
        self.handle = ranker._model.register_forward_hook(self.record, with_kwargs=True)

    def record(self, model, args, kwargs, output):
        import resource

        import torch

        torch.mps.synchronize()
        tokens, logits = kwargs["input_ids"], output.logits
        parameter = next(model.parameters())
        self.records.append(
            {
                "model_class": type(model).__module__ + "." + type(model).__name__,
                "device": str(parameter.device),
                "dtype": str(parameter.dtype),
                "batch_size": int(tokens.shape[0]),
                "shape": list(logits.shape),
                "input_shape": list(tokens.shape),
                "input_hash": sha256(
                    tokens.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "output_hash": sha256(
                    logits.detach().float().cpu().numpy().tobytes()
                ).hexdigest(),
                "finite": bool(torch.isfinite(logits).all().item()),
                "process_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_bytes": torch.mps.driver_allocated_memory(),
            }
        )

    def close(self):
        self.handle.remove()


def make_pool(url):
    from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
    from resolv.schemas.hybrid_knowledge import HybridRetrievalProfile
    from resolv.services.knowledge_branch_runtime import KnowledgeRetrieverPool

    return KnowledgeRetrieverPool(
        knowledge_settings=KnowledgeDatabaseSettings(
            _env_file=None,
            knowledge_database_url=url.render_as_string(hide_password=False),
        ),
        milvus_settings=MilvusSettings(),
        retrieval_profile=HybridRetrievalProfile(),
    )


def budgeted_retrieve(ownership, label, source_hash, function, *, continuation=None):
    from scripts.refund_gate_support import (
        finish_refund_retrieval,
        reserve_refund_retrieval,
    )

    reserve_refund_retrieval(
        ownership,
        label=label,
        source_hash=source_hash,
        **({"continuation": continuation} if continuation else {}),
    )
    outcome = "failed"
    try:
        result = function()
        outcome = "succeeded"
        return result
    finally:
        finish_refund_retrieval(ownership, label=label, outcome=outcome)


def retrieval_gate(ownership, target, source_hash):
    from contextlib import ExitStack

    from scripts.refund_gate_support import (
        configured_urls,
        owned_refund_resources,
        save_json,
    )
    from sqlalchemy import select

    from resolv.models.knowledge import KnowledgeDocument
    from resolv.schemas.hybrid_knowledge import (
        HybridIndexProfile,
        HybridRetrievalError,
    )
    from resolv.schemas.knowledge_release import ReleaseSelection
    from resolv.services.context import TokenCounter
    from resolv.services.knowledge_branch_evidence import select_knowledge_evidence

    formal = ReleaseSelection(
        status="configured",
        release_id="knowledge-20261015-v1",
        index_version="knowledge_20261015_v1",
        collection="resolv_knowledge_20261015_v1",
        profile=HybridIndexProfile(collection_name="resolv_knowledge_20261015_v1"),
        profile_hash="ad0679960b397fdac70efcbc80933957dd6fc4c05fe9c5fb4a8fa6269a5a8846",
        manifest_hash="9e6aef90f108bfaa0fba6836bbacf96ba0e969cbe9213c0c9c669461d2d06c1c",
    )
    report = {
        "passed": False,
        "queries": [],
        "probes": {},
        "formal_identity": formal.model_dump(mode="json"),
    }
    # Save each query before moving on; an inference failure is never retried.
    try:
        with owned_refund_resources(ownership) as resources, ExitStack() as stack:
            k, _ = configured_urls()
            pool = make_pool(k)
            stack.callback(pool.close)
            pool.validate_selection(formal)
            recorder = ForwardRecorder(pool._ranker())
            stack.callback(recorder.close)
            counter = TokenCounter("o200k_base")
            for i, query in enumerate(FORMAL_QUERIES, 1):
                begin = len(recorder.records)
                result = budgeted_retrieve(
                    ownership,
                    f"formal{i}",
                    source_hash,
                    lambda q=query: pool.retrieve(formal, q, bm25_text=q),
                )
                bundle = select_knowledge_evidence(
                    result, counter=counter, max_tokens=3000
                )
                row = {
                    "query": query,
                    "result": result.model_dump(mode="json"),
                    "diagnostics": result.diagnostics.model_dump(mode="json"),
                    "evidence_tokens": bundle.input_tokens,
                    "evidence": bundle.model_dump(mode="json"),
                    "forward_receipts": recorder.records[begin:],
                }
                report["queries"].append(row)
                save_json(target / f"formal{i}.json", row)
            dense_model = pool._dense._service._service._model.model
            parameter = next(dense_model.parameters())
            require(
                str(parameter.device) == "cpu"
                and str(parameter.dtype) == "torch.float32",
                "Dense execution profile changed",
            )
            report["dense_inference"] = {
                "device": str(parameter.device),
                "dtype": str(parameter.dtype),
                "model_class": type(dense_model).__module__
                + "."
                + type(dense_model).__name__,
                "model_revision": formal.profile.model_revision,
            }
            owned = make_pool(k.set(database=resources.identity["knowledge_database"]))
            stack.callback(owned.close)
            owned._dense, owned._reranker = pool._dense, pool._reranker
            selection = owned_selection(resources)
            verify_collection(owned, resources.identity, selection)
            report["collection_identity_verified"] = True
            result = budgeted_retrieve(
                ownership,
                "boundary1",
                source_hash,
                lambda: owned.retrieve(
                    selection,
                    "普通商品完好可二次销售无理由退货",
                    bm25_text="zz" + resources.identity["owner"] + "qz",
                ),
            )
            report["probes"]["one_lane_empty"] = {
                "passed": result.diagnostics.bm25_count == 0
                and result.diagnostics.dense_count > 0
                and result.diagnostics.scored_count > 0,
                "diagnostics": result.diagnostics.model_dump(mode="json"),
            }
            _, store, _ = owned._versions[selection.fingerprint]
            original = store.search_bm25

            def fail(*args, **kwargs):
                raise TimeoutError("Controlled owned BM25 fault")

            store.search_bm25 = fail
            try:
                budgeted_retrieve(
                    ownership,
                    "boundary2",
                    source_hash,
                    lambda: owned.retrieve(
                        selection, "普通商品退货条件", bm25_text="无理由退货"
                    ),
                )
            except HybridRetrievalError as exc:
                report["probes"]["bm25_fault"] = {
                    "passed": True,
                    "error_code": exc.code,
                    "partial_result_returned": False,
                }
            finally:
                store.search_bm25 = original
            with resources.knowledge_factory() as session:
                originals = {
                    row.document_id: row.raw_text
                    for row in session.scalars(select(KnowledgeDocument))
                }
            require(len(originals) == 6, "Unexpected owned authority rows")
            try:
                with resources.knowledge_factory.begin() as session:
                    for did, raw in originals.items():
                        session.get(KnowledgeDocument, did).raw_text = (
                            raw + "\nrefund-owned-authority-mismatch"
                        )
                result = budgeted_retrieve(
                    ownership,
                    "boundary3",
                    source_hash,
                    lambda: owned.retrieve(
                        selection, "普通商品退货条件", bm25_text="无理由退货"
                    ),
                )
                report["probes"]["authority_mismatch"] = {
                    "passed": result.diagnostics.invalid_count > 0
                    and result.diagnostics.center_count == 0
                    and not result.windows,
                    "diagnostics": result.diagnostics.model_dump(mode="json"),
                    "owned_only": True,
                }
            finally:
                with resources.knowledge_factory.begin() as session:
                    for did, raw in originals.items():
                        session.get(KnowledgeDocument, did).raw_text = raw
            report["passed"] = True
    finally:
        save_json(target / "retrieval-observed.json", report)
    return report


class BudgetedUnderstanding:
    def __init__(self, base, budget):
        self.base, self.budget = base, budget

    def invoke(
        self, stage, messages, output_type, estimate, reserved, audit_hook, **kwargs
    ):
        from resolv.services.hybrid_fingerprints import canonical_hash

        return self.budget.invoke(
            purpose=stage,
            input_hash=canonical_hash([m.model_dump(mode="json") for m in messages]),
            audit_hook=audit_hook,
            invoke=lambda audit_hook: self.base.invoke(
                stage, messages, output_type, estimate, reserved, audit_hook, **kwargs
            ),
        )


class DeniedModel:
    def __init__(self, model_name="denied-model"):
        self.attempts = []
        self.model_name = model_name

    def _deny(self, *args, **kwargs):
        self.attempts.append("unapproved_model_call")
        raise ValueError("This Gate authorizes only understanding and refund calls")

    bind = bind_tools = invoke = stream = for_run = create_candidate = _deny


def make_semantic_service(resources, case, budget, target, cleanup):
    """Explicitly compose real production graph/services with owned settings."""
    from functools import partial

    from scripts.refund_gate_support import configured_urls, save_json

    from resolv.core.business_model import make_business_config
    from resolv.core.config import (
        ContextSettings,
        ConversationDatabaseSettings,
        Settings,
        SummaryModelSettings,
        UnderstandingSettings,
    )
    from resolv.core.context_budget import apply_context_budget, apply_summary_budget
    from resolv.core.knowledge_branch_config import runtime_knowledge_branch_profile
    from resolv.core.understanding_model import (
        JsonStageModel,
        get_understanding_model,
        understanding_model_profile,
    )
    from resolv.core.understanding_prompts import (
        INTENT_PROMPT_V1,
        INTENT_VERSION,
        RESOLUTION_PROMPT_V1,
        RESOLUTION_VERSION,
    )
    from resolv.db.checkpoints import checkpoint_session
    from resolv.schemas.refund import RefundAccessContext
    from resolv.services.business_summary import AuditedSummaryService, summary_profile
    from resolv.services.context import ContextBuilder, TokenCounter
    from resolv.services.conversation import ConversationService
    from resolv.services.hybrid_fingerprints import canonical_hash
    from resolv.services.intent_classification import IntentClassifier
    from resolv.services.knowledge_release import selector_payload
    from resolv.services.reference_resolution import ReferenceResolver
    from resolv.services.refund_runtime import create_refund_runtime
    from resolv.services.routed_runtime import business_context_settings, close_model
    from resolv.services.tool_execution import WriteExecutionService
    from resolv.services.understanding import UnderstandingService
    from resolv.services.understanding_context import UnderstandingContext

    settings, context, options, summary = (
        Settings(),
        apply_context_budget(ContextSettings()),
        UnderstandingSettings(),
        apply_summary_budget(SummaryModelSettings()),
    )
    _, c = configured_urls()
    ck = ConversationDatabaseSettings(
        _env_file=None,
        conversation_database_url=c.set(
            database=resources.identity["conversation_database"]
        ).render_as_string(hide_password=False),
    )
    checkpoint = partial(checkpoint_session, ck)
    counter = TokenCounter(context.tokenizer_profile)
    understanding_context = UnderstandingContext(
        context,
        counter,
        model_name=settings.openai_model,
        api_address_fingerprint=canonical_hash(settings.openai_base_url),
        resolution_prompt=RESOLUTION_PROMPT_V1,
        resolution_version=RESOLUTION_VERSION,
        intent_prompt=INTENT_PROMPT_V1,
        intent_version=INTENT_VERSION,
        overhead_override=options.provider_input_overhead_tokens,
        **understanding_model_profile(settings, options.reasoning_effort),
    )
    raw = get_understanding_model(settings, options.timeout_seconds)
    cleanup.callback(close_model, raw)
    model = BudgetedUnderstanding(JsonStageModel(raw, counter), budget)
    owner = "refund-gate-" + resources.identity["owner"][:16]
    understanding = UnderstandingService(
        resources.conversation_factory,
        checkpoint,
        understanding_context,
        ReferenceResolver(model, understanding_context),
        IntentClassifier(model, understanding_context),
        owner_id=owner,
    )
    business = make_business_config(settings)
    business.summary_profile = summary_profile(summary)
    profile = runtime_knowledge_branch_profile(
        settings,
        understanding_config=understanding_context.config,
        business_config=business,
        summary_settings=summary,
    )
    k, _ = configured_urls()
    pool = make_pool(k.set(database=resources.identity["knowledge_database"]))
    cleanup.callback(pool.close)
    selection = owned_selection(resources)
    verify_collection(pool, resources.identity, selection)
    denied = DeniedModel(settings.openai_model)
    summary_guard = DeniedModel(summary.model_name)
    summary_guard.attempts = denied.attempts
    selector = target / "owned-selector.json"
    save_json(selector, selector_payload(selection))
    service = ConversationService(
        resources.conversation_factory,
        checkpoint,
        owner_id=owner,
        understanding=understanding,
        orchestration_version="routed-v4",
        business_run_config=business,
        knowledge_profile=profile,
        knowledge_clients={
            p: denied
            for p in ("knowledge_query", "knowledge_check", "knowledge_answer")
        },
        knowledge_retrievers=pool,
        release_selector_path=selector,
        context_builder=ContextBuilder(
            business_context_settings(business, context), counter
        ),
        summary_service=AuditedSummaryService(summary_guard, counter, summary),
        model=denied,
        max_iterations=business.max_agent_steps,
        write_service=WriteExecutionService(
            resources.conversation_factory, resources.business_factory
        ),
        run_config={
            "prompt_version": business.prompt_version,
            "state_schema_version": 4,
        },
    )
    facts = IsolatedRefundFacts(case, resources.identity["owner"])

    class Identity:
        def resolve(self, server_context):
            require(
                server_context == {"case_id": case["case_id"]},
                "Only a server-owned fixture context is accepted",
            )
            return RefundAccessContext(
                subject_ref=case["subject_ref"],
                namespace=facts.descriptor.namespace,
                authorization_version=facts.descriptor.authorization_version,
                trusted=True,
            )

    identity = Identity()
    runtime = create_refund_runtime(
        service,
        settings=settings,
        identity=identity,
        facts=facts,
        business_factory=resources.business_factory,
    )
    cleanup.callback(runtime.close)
    frozen = understanding_context.config
    for purpose in ("resolution", "intents"):
        budget.profiles[purpose] = {
            "model_name": frozen.model_name,
            "protocol": frozen.model_api_mode,
            "profile_fingerprint": canonical_hash(frozen.model_dump(mode="json")),
            "input_limit": frozen.input_limit,
            "output_limit": frozen.reserved_output_tokens,
        }
    for frozen in (runtime.profile.query_model, runtime.profile.decision_model):
        budget.profiles[frozen.purpose] = {
            "model_name": frozen.model_name,
            "protocol": frozen.model_api_mode,
            "profile_fingerprint": frozen.fingerprint,
            "input_limit": frozen.input_limit,
            "output_limit": frozen.reserved_output_tokens,
        }
    budget.base = runtime.client
    runtime.flow.client = budget
    runtime.submissions.clock = lambda: NOW
    service.refund_runtime, service.refund_profile = runtime, runtime.profile
    service.refund_identity_context = {"case_id": case["case_id"]}
    service.refund_clock = lambda: NOW
    return service, facts, denied


def seed_rejected(resources, case, source):
    from tests.refund_fixtures import binding

    from resolv.models.refund_application import RefundApplication
    from resolv.repositories.refund_application import RefundApplicationRepository
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund_submission import (
        RefundReceipt,
        RefundRegistrationCommand,
    )
    from resolv.services.refund_registration import begin_refund_write

    b = binding(case["order_id"], source)
    command = RefundRegistrationCommand(
        command_id="seed-F4",
        offer_id="seed-F4",
        request_hash="a" * 64,
        binding=b,
        subject_ref=case["subject_ref"],
        money=case["money"],
        reason=case["existing_application"]["reason"],
        decision_hash="b" * 64,
        evidence_hash="c" * 64,
        revalidation_hash="d" * 64,
        snapshots=FrozenJSON.from_value({"fixture_seed": True}),
        created_at=NOW,
    )
    with resources.business_factory() as session:
        repo = RefundApplicationRepository(session)
        existing = repo.get(source, case["order_id"])
        if existing is not None:
            saved = repo.receipt(command.command_id)
            require(
                existing.application_id
                == case["existing_application"]["application_id"]
                and existing.status == "rejected"
                and existing.reason == command.reason
                and existing.money == case["money"]
                and existing.subject_ref == case["subject_ref"]
                and saved is not None
                and saved.command_hash == command.fingerprint
                and repo.validated_receipt(saved).application_id
                == existing.application_id,
                "Existing F4 fixture differs; never reset it",
            )
            return
        begin_refund_write(session)
        # A pre-existing fixture has a stable ID before any model call. This
        # fixture-only row is never evidence of a live application's creation.
        application = RefundApplication(
            application_id=case["existing_application"]["application_id"],
            order_source=source,
            order_id=case["order_id"],
            subject_ref=case["subject_ref"],
            status="rejected",
            reason=command.reason,
            money=case["money"],
            origin="offer",
            offer_id=command.offer_id,
            command_id=command.command_id,
            decision_hash=command.decision_hash,
            evidence_hash=command.evidence_hash,
            revalidation_hash=command.revalidation_hash,
            snapshots=command.snapshots.value,
            created_at=NOW,
            legacy_refund_id=None,
        )
        session.add(application)
        session.flush()
        receipt = RefundReceipt(
            application_id=application.application_id,
            order_source=source,
            order_id=case["order_id"],
            status="rejected",
            created=True,
            money=command.money.public(),
            reason=command.reason,
            created_at=NOW,
        )
        RefundApplicationRepository(session).add_receipt(command, application, receipt)
        session.commit()


def semantic_submission_request_id(case_id, offer_id):
    return "submit-" + case_id + "-" + offer_id


def run_semantic_round(cases, *, invoke):
    results, stopped = [], False
    for case in cases:
        if stopped:
            result = {
                "case_id": case["case_id"],
                "question": case["question"],
                "expected": case["expected"],
                "status": "not_executed",
                "machine_passed": False,
                "passed": False,
            }
        else:
            result = invoke(case)
            stopped = not result.get("machine_passed", False)
        results.append(result)
    return results


def semantic_case(
    resources,
    case,
    ownership,
    source_hash,
    target,
    *,
    attempt=1,
    continuation=None,
    diagnostic_hook=None,
):
    from contextlib import ExitStack

    from scripts.refund_gate_support import reserve_refund_case_attempt, save_json
    from sqlalchemy import func, select

    from resolv.models.conversation import ConversationAction
    from resolv.models.refund_application import RefundApplication
    from resolv.repositories.refund_flow import RefundFlowRepository
    from resolv.schemas.refund_submission import RefundSubmitInput

    reserve_refund_case_attempt(
        ownership,
        case_id=case["case_id"],
        attempt=attempt,
        source_hash=source_hash,
        **({"continuation": continuation} if continuation else {}),
    )
    budget = BudgetedRefundClient(
        None,
        ownership,
        case["case_id"],
        attempt,
        source_hash,
        **({"continuation": continuation} if continuation else {}),
    )
    result = {
        "case_id": case["case_id"],
        "attempt": attempt,
        "question": case["question"],
        "clock": NOW.isoformat(),
        "expected": case["expected"],
        "passed": False,
    }
    try:
        with ExitStack() as cleanup:
            service, facts, denied = make_semantic_service(
                resources, case, budget, target, cleanup
            )
            runtime_client = service.refund_runtime.client
            runtime_client.diagnostic_hook = diagnostic_hook
            source = facts.descriptor.namespace
            if case["existing_application"]:
                seed_rejected(resources, case, source)

            def count():
                with resources.business_factory() as session:
                    return session.scalar(
                        select(func.count())
                        .select_from(RefundApplication)
                        .where(
                            RefundApplication.order_source == source,
                            RefundApplication.order_id == case["order_id"],
                        )
                    )

            result["application_count_before"] = count()
            c = service.create()
            result["conversation_id"] = c.conversation_id
            run, _ = service.accept(
                c.conversation_id, case["case_id"], case["question"], "agent"
            )
            result.update(conversation_id=c.conversation_id, run_id=run.run_id)
            original_retrieve = service.knowledge_retrievers.retrieve
            service.knowledge_retrievers.retrieve = lambda *a, **kw: budgeted_retrieve(
                ownership,
                case["case_id"] + ":" + str(attempt),
                source_hash,
                lambda: original_retrieve(*a, **kw),
                **({"continuation": continuation} if continuation else {}),
            )
            events = list(service.run(run.run_id))
            saved = service.get_run(run.run_id)
            result.update(
                status=saved.status,
                error_code=saved.error_code,
                events=[e.type for e in events],
                unapproved_transport_attempts=denied.attempts,
            )
            result["pending_requests"] = [
                t.model_dump(mode="json")
                for t in service.understanding.list_tasks(c.conversation_id)
            ]
            with resources.conversation_factory() as session:
                record = RefundFlowRepository(session, service.owner_id).get(run.run_id)
                result["legacy_tool_calls"] = session.scalar(
                    select(func.count())
                    .select_from(ConversationAction)
                    .where(ConversationAction.run_id == run.run_id)
                )
            if record:
                result.update(
                    binding=record.binding.model_dump(mode="json")
                    if record.binding
                    else None,
                    observations=[
                        o.model_dump(mode="json") for o in record.observations
                    ],
                    decision=record.decision.model_dump(mode="json")
                    if record.decision
                    else None,
                    refund_record=record.model_dump(mode="json"),
                )
            if saved.status == "completed":
                result["metadata"] = service.reply_metadata(run.run_id)
                detail = service.detail(c.conversation_id)
                result["answer"] = next(
                    m.text
                    for m in detail.messages
                    if m.message_id == saved.answer_message_id
                )
                require(
                    next(r.result_json for r in detail.runs if r.run_id == run.run_id)
                    == result["metadata"],
                    "Refresh projection differs",
                )
                public = result["metadata"].get("refund")
                if public and public.get("offer_id"):
                    require(
                        case["case_id"] in {"F1", "F7"},
                        "Unexpected offer is never submitted",
                    )
                    view = service.refund_runtime.submissions.submit(
                        c.conversation_id,
                        public["offer_id"],
                        RefundSubmitInput(
                            request_id=semantic_submission_request_id(
                                case["case_id"], public["offer_id"]
                            ),
                            confirm=True,
                        ),
                        service.refund_identity_context,
                    )
                    result["submission"] = view.model_dump(mode="json")
                    again = service.refund_runtime.submissions.submit(
                        c.conversation_id,
                        public["offer_id"],
                        RefundSubmitInput(
                            request_id=semantic_submission_request_id(
                                case["case_id"], public["offer_id"]
                            ),
                            confirm=True,
                        ),
                        service.refund_identity_context,
                    )
                    result["duplicate_receipt"] = (
                        again.receipt.model_dump(mode="json") if again.receipt else None
                    )
            result.update(application_count_after=count(), adapter_calls=facts.calls)
            result.update(check_semantic_result(result, case))
    except Exception as exc:  # noqa: BLE001 — safe gate metadata, never driver/provider details
        result.update(
            passed=False,
            error_code=getattr(exc, "code", "refund_case_failed"),
            error_type=type(exc).__name__,
        )
    finally:
        result.update(audits=budget.audits, model_outputs=budget.outputs)
        save_json(target / (case["case_id"] + ".json"), result)
    return result


def empty_intent_probe(ownership):
    from scripts.refund_gate_support import owned_refund_resources
    from sqlalchemy import text

    from resolv.core.config import MilvusSettings
    from resolv.db.milvus import milvus_client

    with owned_refund_resources(ownership) as resources:
        with resources.knowledge_engine.connect() as connection:
            counts = {
                name: connection.execute(
                    text("SELECT COUNT(*) FROM `" + name + "`")
                ).scalar_one()
                for name in (
                    "knowledge_documents",
                    "knowledge_chunks",
                    "knowledge_index_versions",
                    "knowledge_chunk_indexes",
                )
            }
        with milvus_client(MilvusSettings()) as client:
            exists = client.has_collection(
                collection_name=resources.identity["collection"], timeout=15
            )
    return counts, exists


def validate_empty_intent_continuation(
    ownership, prior_path, fingerprints, *, counts, collection_exists
):
    from scripts.refund_gate_support import budget_ledger, validate_manifest

    from resolv.services.hybrid_fingerprints import canonical_hash

    public = ROOT / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    require(
        Path(prior_path).resolve() == (public / "R2-prepare-correction.json").resolve()
        and not Path(prior_path).is_symlink(),
        "Only the retained attribute failure is eligible",
    )
    prior, obj = json.loads(Path(prior_path).read_text()), validate_manifest(ownership)
    old = prior.get("resource_identity", {})
    require(
        prior.get("passed") is False
        and prior.get("error_type") == "AttributeError"
        and prior.get("source_fingerprints") != fingerprints
        and obj.get("collection_build") == old.get("collection_build") == "started"
        and not obj.get("prepare_continuation")
        and not obj.get("retrievals")
        and not obj.get("semantic_attempts")
        and all(c["case_id"] == "R1" for c in budget_ledger(ownership))
        and not (public / "R2-prepare-continuation.json").exists(),
        "Only one explicit empty-intent continuation is allowed",
    )
    require(
        set(counts)
        == {
            "knowledge_documents",
            "knowledge_chunks",
            "knowledge_index_versions",
            "knowledge_chunk_indexes",
        }
        and all(type(v) is int and v == 0 for v in counts.values())
        and collection_exists is False,
        "Any index data or collection forbids this continuation",
    )
    require(
        all(
            old.get(k) == obj.get(k)
            for k in (
                "kind",
                "owner",
                "server_fingerprint",
                "business_db",
                "business_identity",
                "knowledge_database",
                "conversation_database",
                "collection",
            )
        ),
        "Continuation ownership changed",
    )
    return canonical_hash(prior)


def validate_preflight_correction(ownership, prior_path, fingerprints):
    from scripts.refund_gate_support import budget_ledger, validate_manifest

    from resolv.services.hybrid_fingerprints import canonical_hash

    public = ROOT / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    require(
        Path(prior_path).resolve() == (public / "R2-prepare.json").resolve()
        and not Path(prior_path).is_symlink(),
        "Only the retained initial prepare report is eligible",
    )
    prior = json.loads(Path(prior_path).read_text())
    obj = validate_manifest(ownership)
    old = prior.get("resource_identity", {})
    require(
        prior.get("gate") == "R2-prepare"
        and prior.get("passed") is False
        and prior.get("error_type") == "TypeError"
        and fingerprints != prior.get("source_fingerprints")
        and "collection_build" not in obj
        and "collection_build" not in old
        and not obj.get("retrievals")
        and all(c["case_id"] == "R1" for c in budget_ledger(ownership))
        and not (public / "R2-prepare-correction.json").exists(),
        "Only one corrected zero-work preflight is allowed",
    )
    require(
        all(
            old.get(k) == obj.get(k)
            for k in (
                "kind",
                "owner",
                "server_fingerprint",
                "business_db",
                "business_identity",
                "knowledge_database",
                "conversation_database",
                "collection",
            )
        ),
        "Correction resource identity changed",
    )
    return canonical_hash(prior)


def select_case_attempts(case_ids, *, attempt):
    fixed = load_cases()["cases"]
    ids = tuple(case_ids)
    if attempt == 1:
        require(
            ids == tuple(c["case_id"] for c in fixed),
            "Initial Gate requires all eight fixed cases",
        )
    else:
        require(
            attempt == 2
            and 1 <= len(ids) <= 2
            and len(set(ids)) == len(ids)
            and all(i in {c["case_id"] for c in fixed} for i in ids),
            "At most two distinct fixed corrected cases",
        )
    return [next(c for c in fixed if c["case_id"] == i) for i in ids]


def run_refund_gate(
    *,
    ownership,
    case_ids,
    report_dir,
    prepare_report=None,
    attempt=1,
    fix_commit=None,
    continuation=None,
    approval_commit=None,
):
    from scripts.refund_gate_support import (
        authorize_case_retest,
        committed_sources,
        new_report_dir,
        owned_refund_resources,
    )

    from resolv.services.hybrid_fingerprints import canonical_hash

    cases = select_case_attempts(case_ids, attempt=attempt)
    target = new_report_dir(ownership, report_dir)
    if continuation:
        if continuation in {"C2", "C3"}:
            from scripts.refund_semantic_continuation import (
                fingerprints as continuation_fingerprints,
            )
        else:
            from scripts.refund_continuation import (
                fingerprints as continuation_fingerprints,
            )
        commit, fingerprints = continuation_fingerprints()
    else:
        commit, fingerprints = committed_sources(SOURCES)
    prerequisites = ROOT / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    initial = None
    if attempt == 2 and not continuation:
        import subprocess

        initial = json.loads((prerequisites / "R2-semantic.json").read_text())
        require(
            fix_commit == commit
            and re.fullmatch("[0-9a-f]{40}", fix_commit or "")
            and re.fullmatch("[0-9a-f]{40}", initial.get("candidate_commit", "")),
            "A committed fix must be explicit",
        )
        changed = set(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "diff",
                    "--name-only",
                    initial["candidate_commit"],
                    fix_commit,
                    "--",
                ],
                text=True,
            ).splitlines()
        )
        require(
            changed & set(SOURCES)
            and all(
                any(
                    old["case_id"] == c["case_id"]
                    and old["question"] == c["question"]
                    and old["expected"] == c["expected"]
                    for old in initial["cases"]
                )
                for c in cases
            ),
            "A relevant source fix and identical fixed cases are required",
        )
        for case in cases:
            authorize_case_retest(
                ownership,
                case_id=case["case_id"],
                source_hash=canonical_hash(fingerprints),
            )
    prepared = (
        Path(prepare_report) if prepare_report else prerequisites / "R2-prepare.json"
    )
    require(
        prepared.resolve()
        in {
            (prerequisites / name).resolve()
            for name in (
                "R2-prepare.json",
                "R2-prepare-correction.json",
                "R2-prepare-continuation.json",
            )
        },
        "Explicit prepare receipt required",
    )
    if prepared.name == "R2-prepare-correction.json":
        require(
            json.loads(prepared.read_text()).get("preflight_correction_of")
            == canonical_hash(
                json.loads((prerequisites / "R2-prepare.json").read_text())
            ),
            "Correction receipt lineage differs",
        )
    if prepared.name == "R2-prepare-continuation.json":
        require(
            json.loads(prepared.read_text()).get("prepare_continuation_of")
            == canonical_hash(
                json.loads((prerequisites / "R2-prepare-correction.json").read_text())
            ),
            "Continuation receipt lineage differs",
        )
    for mode, receipt in (
        ("prepare", prepared),
        ("retrieval", prerequisites / "R2-retrieval.json"),
    ):
        prior = json.loads(receipt.read_text())
        require(prior.get("passed") is True, "R2 prerequisite did not pass: " + mode)
    with owned_refund_resources(ownership) as resources:
        owned_selection(resources)
        if continuation:
            source_hash = canonical_hash(fingerprints)
            if continuation in {"C2", "C3"}:
                from scripts.refund_semantic_continuation import (
                    authorize_c2,
                    authorize_c3,
                    case_attempts,
                    finish_case,
                )

                attempts = case_attempts(continuation)
                authorize = authorize_c3 if continuation == "C3" else authorize_c2
                authorize(
                    ownership, approval_commit=approval_commit, source_hash=source_hash
                )
            else:
                from scripts.refund_continuation import ATTEMPTS, finish_case, grant
                from scripts.refund_gate_support import authorize_refund_continuation

                require(continuation == "C1", "Unknown continuation")
                authorize_refund_continuation(
                    ownership, approval_commit=approval_commit, source_hash=source_hash
                )
                r1 = json.loads((prerequisites / "R1-C1.json").read_text())
                require(
                    r1.get("passed") is True
                    and r1.get("source_fingerprints") == fingerprints,
                    "Source-matching R1-C1 required",
                )
                require(
                    grant(resources.identity, source_hash)["r1_passed"],
                    "C1 R1 not passed",
                )
                attempts = ATTEMPTS

            def invoke(case):
                result = semantic_case(
                    resources,
                    case,
                    ownership,
                    source_hash,
                    target,
                    attempt=attempts[case["case_id"]],
                    **(
                        {"continuation": continuation}
                        if continuation in {"C2", "C3"}
                        else {}
                    ),
                )
                finish_case(
                    ownership,
                    source_hash=source_hash,
                    case_id=case["case_id"],
                    passed=result.get("machine_passed", False),
                    **(
                        {"continuation": continuation}
                        if continuation in {"C2", "C3"}
                        else {}
                    ),
                )
                return result

            results = run_semantic_round(cases, invoke=invoke)
        else:
            results = [
                semantic_case(
                    resources,
                    case,
                    ownership,
                    canonical_hash(fingerprints),
                    target,
                    attempt=attempt,
                )
                for case in cases
            ]
    return {
        "gate": "R2-" + continuation
        if continuation
        else "R2-semantic"
        if attempt == 1
        else "R2-semantic-retest",
        "passed": False,
        "machine_passed": all(c.get("machine_passed") for c in results),
        "manual_review_required": True,
        "candidate_commit": commit,
        "source_fingerprints": fingerprints,
        "cases": results,
        "initial_semantic_report_hash": canonical_hash(initial)
        if initial is not None
        else None,
        "attempt": attempt,
        **(
            {"r1_prerequisite_deferred": True, "formal_migration_eligible": False}
            if continuation in {"C2", "C3"}
            else {}
        ),
    }


def review_semantics(original, reviews):
    from resolv.services.hybrid_fingerprints import canonical_hash

    require(
        reviews.get("observed_report_hash") == canonical_hash(original)
        and reviews.get("reviewer")
        and isinstance(reviews.get("case_reviews"), list),
        "Manual review must bind the immutable observed report",
    )
    fixed = load_cases()["cases"]
    require(
        [c["case_id"] for c in original.get("cases", [])]
        == [c["case_id"] for c in fixed]
        and [r["case_id"] for r in reviews["case_reviews"]]
        == [c["case_id"] for c in fixed],
        "All eight cases require an explicit review",
    )
    results = []
    for observed, case, review in zip(
        original["cases"], fixed, reviews["case_reviews"], strict=True
    ):
        require(
            review.get("result_hash") == canonical_hash(observed)
            and set(review.get("checks", {}))
            == {
                "scope",
                "facts",
                "policy",
                "exceptions",
                "time",
                "reason",
                "citations",
                "status",
                "submission",
                "no_funds_promise",
                "unverified",
            }
            and all(type(v) is bool for v in review["checks"].values())
            and review.get("notes"),
            "Explicit case review checklist and source hash required",
        )
        results.append(
            {
                "case_id": case["case_id"],
                **check_semantic_result(observed, case, manual_review=review),
                "review": review,
            }
        )
    return {
        "gate": original["gate"] + "-reviewed"
        if original.get("gate") in {"R2-C2", "R2-C3"}
        else "R2-reviewed",
        "passed": all(c["passed"] for c in results),
        "observed_report_hash": canonical_hash(original),
        "reviewer": reviews["reviewer"],
        "cases": results,
        "candidate_commit": original["candidate_commit"],
        "source_fingerprints": original["source_fingerprints"],
        "formal_preserved": original.get("formal_preserved") is True,
        **(
            {"r1_prerequisite_deferred": True, "formal_migration_eligible": False}
            if original.get("gate") in {"R2-C2", "R2-C3"}
            else {}
        ),
    }


def main(argv=None):
    from time import monotonic

    from scripts.refund_gate_support import (
        budget_ledger,
        committed_sources,
        manifest_lock,
        new_report_dir,
        save_json,
        snapshot_formal_resources,
        validate_manifest,
    )

    from resolv.services.hybrid_fingerprints import canonical_hash

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "retrieval", "semantic"), required=True
    )
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cases", default="F1,F2,F3,F4,F5,F6,F7,F8")
    parser.add_argument("--preflight-correction-of", type=Path)
    parser.add_argument("--continue-empty-intent-from", type=Path)
    parser.add_argument("--prepare-report", type=Path)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fix-commit")
    parser.add_argument("--continuation", choices=("C1", "C2", "C3"))
    parser.add_argument("--approval-plan-commit")
    parser.add_argument(
        "--review",
        type=Path,
        help="Review the retained semantic report without provider calls",
    )
    args = parser.parse_args(argv)
    require(
        bool(args.continuation) == bool(args.approval_plan_commit)
        and (
            not args.continuation
            or args.mode == "semantic"
            and args.attempt == 1
            and not args.fix_commit
        ),
        "Continuations require an explicit fixed semantic round",
    )
    require(
        (args.attempt == 1 and args.fix_commit is None)
        or (
            args.attempt == 2
            and args.mode == "semantic"
            and args.fix_commit
            and not args.review
            and not args.preflight_correction_of
            and not args.continue_empty_intent_from
        ),
        "Retest requires an explicit committed semantic fix",
    )
    begin = monotonic()
    validate_manifest(args.ownership)
    target = new_report_dir(args.ownership, args.report_dir)
    public = ROOT / "docs/superpowers/acceptance/2026-10-16-refund-flow"
    destination = public / (
        (
            f"R2-{args.continuation}-review.json"
            if args.review
            else f"R2-{args.continuation}.json"
        )
        if args.continuation
        else "R2-retest.json"
        if args.attempt == 2
        else "R2-prepare-continuation.json"
        if args.continue_empty_intent_from
        else "R2-prepare-correction.json"
        if args.preflight_correction_of
        else "R2.json"
        if args.review
        else "R2-" + args.mode + ".json"
    )
    require(not destination.exists(), "Refusing to overwrite R2 evidence")
    before = None
    report = {"gate": "R2-" + args.mode, "passed": False}
    if args.continuation in {"C2", "C3"}:
        report.update(
            gate="R2-" + args.continuation,
            r1_prerequisite_deferred=True,
            formal_migration_eligible=False,
        )
    try:
        if args.continuation:
            if args.continuation in {"C2", "C3"}:
                from scripts.refund_semantic_continuation import (
                    fingerprints as continuation_fingerprints,
                )
            else:
                from scripts.refund_continuation import (
                    fingerprints as continuation_fingerprints,
                )
            commit, fingerprints = continuation_fingerprints()
        else:
            commit, fingerprints = committed_sources(SOURCES)
        load_cases()
        report.update(candidate_commit=commit, source_fingerprints=fingerprints)
        if args.continue_empty_intent_from:
            require(
                args.mode == "prepare"
                and not args.preflight_correction_of
                and not args.review,
                "Continuation is only for the retained empty prepare intent",
            )
            counts, exists = empty_intent_probe(args.ownership)
            previous = validate_empty_intent_continuation(
                args.ownership,
                args.continue_empty_intent_from,
                fingerprints,
                counts=counts,
                collection_exists=exists,
            )
            with manifest_lock(args.ownership) as current:
                require(
                    not current.get("prepare_continuation")
                    and current.get("collection_build") == "started",
                    "Continuation already claimed",
                )
                current["prepare_continuation"] = {
                    "prior_report_hash": previous,
                    "source_hash": canonical_hash(fingerprints),
                    "counts": counts,
                    "collection_exists": exists,
                }
                save_json(args.ownership, current)
            report.update(
                prepare_continuation_of=previous,
                prepare_zero_work_probe={"counts": counts, "collection_exists": exists},
            )
        if args.preflight_correction_of:
            require(
                args.mode == "prepare" and not args.review,
                "Correction is only for prepare",
            )
            report["preflight_correction_of"] = validate_preflight_correction(
                args.ownership, args.preflight_correction_of, fingerprints
            )
        if args.review:
            require(
                args.mode == "semantic"
                and args.review.resolve().is_relative_to((ROOT / WORKSPACE).resolve()),
                "Review must be an explicit owned artifact",
            )
            observed = json.loads(
                (
                    public
                    / (
                        f"R2-{args.continuation}.json"
                        if args.continuation
                        else "R2-semantic.json"
                    )
                ).read_text()
            )
            report = review_semantics(observed, json.loads(args.review.read_text()))
            report["passed"] = report["passed"] and report["formal_preserved"]
        else:
            if args.mode == "prepare":
                ensure_services(args.ownership, target)
            before = snapshot_formal_resources()
            save_json(target / "preservation-before.json", before)
            if args.mode == "prepare":
                report.update(
                    prepare_owned(
                        args.ownership,
                        target,
                        continue_intent=report["prepare_continuation_of"],
                    )
                    if args.continue_empty_intent_from
                    else prepare_owned(args.ownership, target)
                )
            elif args.mode == "retrieval":
                report.update(
                    retrieval_gate(args.ownership, target, canonical_hash(fingerprints))
                )
            else:
                report.update(
                    run_refund_gate(
                        ownership=args.ownership,
                        case_ids=tuple(args.cases.split(",")),
                        report_dir=target / "cases",
                        prepare_report=args.prepare_report,
                        attempt=args.attempt,
                        fix_commit=args.fix_commit,
                        continuation=args.continuation,
                        approval_commit=args.approval_plan_commit,
                    )
                )
            after = snapshot_formal_resources(layout=before)
            report["formal_preserved"] = before == after
            save_json(target / "preservation-after.json", after)
            require(report["formal_preserved"], "Formal resources changed")
            if args.mode == "retrieval":
                validate_retrieval_gate(report)
            report["ledger"] = budget_ledger(args.ownership)
    except Exception as exc:  # noqa: BLE001 — do not log provider/driver credentials
        report.update(
            passed=False,
            error_code=getattr(exc, "code", "refund_gate_failed"),
            error_type=type(exc).__name__,
        )
        if args.mode == "retrieval" and (target / "retrieval-observed.json").exists():
            report.update(
                json.loads((target / "retrieval-observed.json").read_text()),
                passed=False,
            )
        if args.mode == "semantic" and not report.get("cases"):
            report["cases"] = [
                {
                    "case_id": c["case_id"],
                    "question": c["question"],
                    "expected": c["expected"],
                    "status": "not_executed",
                    "passed": False,
                    "error_code": report["error_code"],
                }
                for c in load_cases()["cases"]
            ]
        if before is not None:
            try:
                after = snapshot_formal_resources(layout=before)
                report["formal_preserved"] = before == after
                save_json(target / "preservation-after.json", after)
            except Exception:  # noqa: BLE001 — missing preservation proof is failure
                report["formal_preserved"] = False
    report.update(
        elapsed_seconds=monotonic() - begin,
        resource_identity=validate_manifest(args.ownership),
        ledger=budget_ledger(args.ownership),
    )
    save_json(target / "report.json", report)
    require(not destination.exists(), "R2 report was concurrently created")
    save_json(destination, report)
    if args.mode == "semantic" and not args.review:
        suffix = (
            "-" + args.continuation
            if args.continuation
            else "-retest"
            if args.attempt == 2
            else ""
        )
        save_json(
            public / ("real-invocation-ledger" + suffix + ".json"),
            budget_ledger(args.ownership),
        )
        save_json(
            public / ("preservation-R2" + suffix + ".json"),
            {
                "before": before,
                "formal_preserved": report.get("formal_preserved", False),
            },
        )
    print(
        canonical_json(
            {
                "passed": report["passed"],
                "gate": report["gate"],
                "report": str(target / "report.json"),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
