"""Explicit positive policy + native return reaches only a local pending application."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from scripts.refund_acceptance_v2 import explicit_policy
from sqlalchemy import func, select

from resolv.core.refund_config import default_refund_profile
from resolv.core.refund_model import RefundModelClient
from resolv.models.conversation import ConversationRun
from resolv.models.refund_application import RefundApplication
from resolv.repositories.refund_flow import RefundFlowRepository
from resolv.schemas.knowledge_branch import canonical_json
from resolv.schemas.refund_flow import RefundDecision
from resolv.schemas.refund_submission import RefundSubmitInput
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.refund_offer import finalize_refund_reply
from resolv.services.refund_submission import RefundSubmissionService
from tests.refund_fixtures import NOW, observation
from tests.refund_flow_fixtures import drive, flow_case
from tests.refund_submission_fixtures import LocalLocks
from tests.test_refund_model import SDK, tool_chunks


@pytest.mark.parametrize(
    "cross_deadline,crash_point",
    [
        (False, None),
        (True, None),
        (False, "refund_after_model_request_saved"),
        (False, "refund_after_decision_saved"),
    ],
)
def test_explicit_elapsed_policy_native_flow_creates_pending_or_reassesses(
    tmp_path, cross_deadline, crash_point
):
    with flow_case(tmp_path) as h:
        policy_text = explicit_policy().split("\n\n", 1)[1].strip()
        quote = policy_text.split("；", 1)[0]
        original = h.flow.retrievers.retrieve

        def retrieve(*args, **kwargs):
            result = original(*args, **kwargs)
            return result.model_copy(
                update={
                    "chunks": {
                        cid: body.model_copy(
                            update={
                                "chunk": body.chunk.model_copy(
                                    update={
                                        "answer": policy_text,
                                        "text": "退款\n" + policy_text,
                                    }
                                )
                            }
                        )
                        for cid, body in result.chunks.items()
                    }
                }
            )

        h.flow.retrievers.retrieve = retrieve
        paid_at = (NOW - timedelta(hours=168) + timedelta(seconds=10)).isoformat()
        payment = h.facts.query_payment

        def paid(access, bound, call_id):
            old = payment(access, bound, call_id)
            return observation(
                "query_refund_payment",
                bound=bound,
                call_id=call_id,
                fields={
                    **{k: f.value for k, f in old.fields.items()},
                    "paid_at": paid_at,
                },
            )

        h.facts.query_payment = paid
        p = default_refund_profile(
            understanding_config=h.profile.query_model,
            business_config=h.profile.decision_model,
            summary_profile=h.profile.summary_profile.value,
            adapter_descriptor=h.profile.adapter,
        )
        model = p.decision_model.model_copy(
            update={
                "api_address_fingerprint": canonical_hash("https://offline.invalid/v1")
            }
        )
        p = p.model_copy(
            update={"decision_model": model, "selection": h.profile.selection}
        )
        with h.factory.begin() as session:
            session.get(ConversationRun, h.run.run_id).refund_config = p.model_dump(
                mode="json"
            )

        class NativeSDK(SDK):
            def stream(self, messages):
                self.calls += 1
                record = h.flow._get(h.run.run_id)
                latest = {o.tool_name: o for o in record.observations}
                choice = self.bound[-1].get("tool_choice")
                if choice:
                    name = choice["function"]["name"]
                else:
                    name = (
                        "query_refund_payment"
                        if "query_refund_payment" not in latest
                        else "return_refund_decision"
                    )
                args = "{}"
                if name == "return_refund_decision":
                    citation = {
                        "citation_number": 1,
                        "window_id": record.evidence.windows[0].window_id,
                        "quote": quote,
                    }
                    refs = [
                        {"observation_id": latest[tool].observation_id, "field": field}
                        for tool, field in (
                            ("query_refund_order", "activated"),
                            ("query_refund_payment", "paid_at"),
                        )
                    ]
                    value = RefundDecision.model_validate_json(
                        canonical_json(
                            {
                                "operation": "apply",
                                "verdict": "eligible",
                                "order_binding_hash": record.binding.binding_hash,
                                "evidence_hash": record.evidence.evidence_hash,
                                "observation_refs": tuple(
                                    o.observation_id for o in record.observations
                                ),
                                "policy_citations": [citation],
                                "rules": [
                                    {
                                        "rule": "未激活且支付后168小时内",
                                        "policy_citations": [citation],
                                        "fact_refs": refs,
                                    }
                                ],
                                "reason": "符合隔离政策，提交后等待审核。",
                                "critical_fields": refs,
                                "recheck_conditions": [
                                    {
                                        "fact_ref": refs[1],
                                        "policy_window_id": citation["window_id"],
                                        "policy_quote": quote,
                                        "operator": "elapsed_seconds_lte",
                                        "expected": 168 * 3600,
                                    }
                                ],
                                "next_action": "show_offer",
                            }
                        )
                    )
                    args = value.model_dump_json()
                chunks = tool_chunks(name=name, args=args)
                chunk = chunks[0]
                chunk.tool_call_chunks[0]["id"] = "native-" + str(self.calls)
                chunk.tool_calls[0]["id"] = "native-" + str(self.calls)
                yield chunk

        sdk = NativeSDK([], [])
        client = RefundModelClient(
            SimpleNamespace(
                openai_model=model.model_name,
                openai_base_url="https://offline.invalid/v1",
                openai_api_key="offline",
            ),
            client_factory=sdk,
        )
        h.flow.client = {"refund_query": h.client, "refund_decision": client}
        h.service.refund_clock = lambda: NOW
        if crash_point:

            def crash(name):
                if name == crash_point and sdk.calls == 4:
                    h.service.fault_hook = None
                    raise RuntimeError("native terminal saved crash")

            h.service.fault_hook = crash
            with pytest.raises(RuntimeError, match="native terminal saved crash"):
                drive(h)
            assert sdk.calls == 4
        record = drive(h)
        assert p.config_version == 7 and record.decision.verdict == "eligible"
        assert [p["tool_choice"]["function"]["name"] for p in sdk.bound[:2]] == [
            "query_refund_order",
            "query_refund_status",
        ]
        assert len(record.model_requests) == 4 and len(record.observations) == 3
        assert not record.model_requests[-1].tool_calls
        finalize_refund_reply(h.service, h.run.run_id, h.execution, record)
        h.service.mark_completed(h.run.run_id, h.execution.epoch)
        with h.factory() as session:
            offer = RefundFlowRepository(session, "local").offer_for_run(h.run.run_id)
        clock = SimpleNamespace(
            now=NOW + timedelta(seconds=11 if cross_deadline else 0)
        )
        submitter = RefundSubmissionService(
            h.factory,
            h.business,
            owner_id="local",
            identity=h.flow.identity,
            facts=h.facts,
            selection_reader=lambda: p.selection,
            profile=p,
            clock=lambda: clock.now,
            execution_lock_factory=LocalLocks(),
        )
        req = RefundSubmitInput(request_id="explicit-positive", confirm=True)
        first = submitter.submit(offer.conversation_id, offer.offer_id, req, {})
        with h.business() as session:
            count = session.scalar(select(func.count()).select_from(RefundApplication))
        if cross_deadline:
            assert (
                first.status == "failed"
                and first.error_code == "refund_needs_reassessment"
                and count == 0
            )
        else:
            assert (
                first.status == "completed"
                and first.receipt.status == "pending"
                and count == 1
            )
            again = submitter.submit(offer.conversation_id, offer.offer_id, req, {})
            assert canonical_json(
                again.receipt.model_dump(mode="json")
            ) == canonical_json(first.receipt.model_dump(mode="json"))
            assert sdk.calls == 4
