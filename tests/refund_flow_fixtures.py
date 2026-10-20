"""Explicit isolated dependency injection; no production fixture selection."""

from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage, messages_to_dict

from resolv.db.base import Base
from resolv.db.conversation_base import ConversationBase
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.models.conversation import ConversationRun
from resolv.models.refund_flow import ConversationRefundRun  # noqa: F401
from resolv.models.understanding import ConversationUnderstanding  # noqa: F401
from resolv.schemas.knowledge_branch import FrozenJSON, canonical_json
from resolv.schemas.refund import RefundAccessContext, RefundMoney
from resolv.schemas.refund_flow import (
    RefundDecision,
    RefundModelAudit,
    RefundModelResponse,
    RefundQueryOutput,
    RefundToolCall,
)
from resolv.services.hybrid_fingerprints import canonical_hash
from tests.knowledge_branch_fixtures import hybrid_result
from tests.refund_fixtures import NOW, observation, refund_case
from tests.test_refund_evidence import harness as retrieval_harness
from tests.test_refund_evidence import selected


@contextmanager
def flow_case(
    tmp_path,
    *,
    mode="eligible",
    policy=True,
    text="订单A001申请整单退款，因为商品未激活",
):
    from resolv.schemas.knowledge_release import ReleaseSelection
    from resolv.services.refund_flow import RefundFlowService
    from resolv.services.refund_registration import RefundRegistrationService

    chat_engine = create_db_engine(f"sqlite:///{tmp_path / 'chat.db'}")
    business_engine = create_db_engine(f"sqlite:///{tmp_path / 'business.db'}")
    ConversationBase.metadata.create_all(chat_engine)
    Base.metadata.create_all(business_engine)
    factory, business = (
        create_session_factory(chat_engine),
        create_session_factory(business_engine),
    )
    counts = Counter()

    def outside():
        assert chat_engine.pool.checkedout() == business_engine.pool.checkedout() == 0

    with refund_case(factory, text=text) as h:
        prof = h.profile.model_copy(
            update={
                "adapter": h.profile.adapter.model_copy(update={"connected": True}),
                "selection": selected()
                if policy
                else ReleaseSelection(status="disabled"),
            }
        )
        with factory.begin() as s:
            s.get(ConversationRun, h.run.run_id).refund_config = prof.model_dump(
                mode="json"
            )

        class Identity:
            def resolve(self, context):
                outside()
                counts["identity"] += 1
                return RefundAccessContext(
                    subject_ref="fixture-subject",
                    namespace=prof.adapter.namespace,
                    authorization_version=prof.adapter.authorization_version,
                    trusted=True,
                )

        class Facts:
            descriptor = prof.adapter
            fail = None

            def authorize(self, access, binding):
                outside()
                counts["authorize"] += 1
                return observation(
                    bound=binding, fields={"authorized": True}, call_id=uuid4().hex
                )

            def read(self, name, binding, call_id, fields):
                outside()
                counts[name] += 1
                if self.fail:
                    raise self.fail
                return observation(name, bound=binding, call_id=call_id, fields=fields)

            def query_order(self, access, binding, call_id):
                return self.read(
                    "query_refund_order",
                    binding,
                    call_id,
                    {
                        "order_id": binding.order_id,
                        "status": "paid",
                        "activated": False,
                        "product_type": "software",
                    },
                )

            def query_payment(self, access, binding, call_id):
                return self.read(
                    "query_refund_payment",
                    binding,
                    call_id,
                    {
                        "refundable_money": RefundMoney(
                            currency="CNY", minor_units=100, unit_precision=2
                        ).model_dump(mode="json")
                    },
                )

            def query_delivery(self, access, binding, call_id):
                return self.read("query_refund_delivery", binding, call_id, {})

            def query_status(self, access, binding, call_id):
                return self.read("query_refund_status", binding, call_id, {})

        class Reader:
            def lookup(self, binding):
                with business() as s:
                    return RefundRegistrationService(s).lookup(binding)

        class Tokens:
            def count(self, messages, schemas):
                return (
                    sum(len(str(m.content)) for m in messages) // 50
                    + len(str(schemas)) // 50
                )

            def count_text(self, text):
                return max(1, len(text) // 50)

        class Client:
            counter_factory = staticmethod(lambda _: Tokens())

            def __init__(self):
                self.mode = mode
                self.fail = None

            def audit(self, messages, profile, hook, step=0):
                a = RefundModelAudit(
                    invocation_id=uuid4().hex,
                    purpose=profile.purpose,
                    profile_fingerprint=profile.fingerprint,
                    input_hash=canonical_hash(messages_to_dict(messages)),
                    input_estimate=7000,
                    model_step=step,
                    started_at=NOW,
                    outcome="started",
                )
                hook(a)
                if self.fail:
                    raise self.fail
                a = a.model_copy(
                    update={
                        "outcome": "succeeded",
                        "finished_at": NOW,
                        "input_tokens": 1,
                        "output_tokens": 1,
                    }
                )
                hook(a)
                return a

            def prepare(self, messages, *, profile, audit_hook):
                outside()
                counts["prepare"] += 1
                self.audit(messages, profile, audit_hook)
                return RefundQueryOutput(
                    standard="软件整单退款条件",
                    expanded=(
                        "退款",
                        "激活",
                    ),
                )

            def decide(
                self,
                messages,
                *,
                profile,
                model_step,
                attempt,
                require_tool,
                audit_hook,
            ):
                import json

                outside()
                counts["decide"] += 1
                audit = self.audit(messages, profile, audit_hook, model_step)
                obs = [
                    json.loads(m.content)
                    for m in messages
                    if isinstance(m, ToolMessage)
                ]
                current = json.loads(
                    next(
                        m.content
                        for m in reversed(messages)
                        if m.type == "human" and m.content.startswith('{"binding":')
                    )
                )
                operation = current["request"]["operation"]
                if self.mode == "loop" or not obs:
                    names = (
                        ["query_refund_order"]
                        if self.mode == "loop"
                        else ["query_refund_order", "query_refund_status"]
                    )
                    if self.mode == "eligible":
                        names.append("query_refund_payment")
                    calls = tuple(
                        RefundToolCall(
                            id=f"s{model_step}-{i}",
                            name=name,
                            args=FrozenJSON.from_value({}),
                        )
                        for i, name in enumerate(names)
                    )
                    d = None
                else:
                    evidence = next(
                        m.content
                        for m in messages
                        if m.type == "human" and m.content.startswith("本轮参考资料")
                    )
                    payload = json.loads(evidence.split("\n", 1)[1])[0]
                    eligible = self.mode == "eligible"
                    application = next(
                        (
                            o["fields"].get("local_application", {}).get("value")
                            for o in obs
                            if o["tool_name"] == "query_refund_status"
                        ),
                        None,
                    )
                    status_reason = (
                        f"本地申请{application['application_id']}状态为{application['status']}，仅代表本地已知进度。"
                        if application
                        else "未发现本地退款申请，外部进度尚未接入。"
                    )
                    fields = {
                        "operation": operation,
                        "verdict": None
                        if operation == "status"
                        else "eligible"
                        if eligible
                        else "undetermined",
                        "order_binding_hash": current["binding"]["binding_hash"],
                        "evidence_hash": canonical_hash(
                            {"rule": "knowledge-evidence-v1", "text": evidence}
                        ),
                        "observation_refs": [o["observation_id"] for o in obs],
                        "reason": status_reason
                        if self.mode in {"status", "existing"}
                        else "符合未激活商品的申请条件。"
                        if eligible
                        else "缺少实际政策信息，暂时无法判断。",
                        "next_action": "show_offer"
                        if eligible
                        else "show_status"
                        if self.mode in {"status", "existing"}
                        else "request_information",
                    }
                    if eligible:
                        c = payload["citations"][0]
                        cid = c["chunk_ids"][0]
                        quote = payload["chunks"][cid]["chunk"]["answer"]
                        citation = {
                            "citation_number": c["citation_number"],
                            "window_id": c["window_id"],
                            "quote": quote,
                        }
                        order = next(
                            o for o in obs if o["tool_name"] == "query_refund_order"
                        )
                        ref = {
                            "observation_id": order["observation_id"],
                            "field": "activated",
                        }
                        fields.update(
                            policy_citations=[citation],
                            critical_fields=[ref],
                            rules=[
                                {
                                    "rule": "未激活的软件可申请",
                                    "policy_citations": [citation],
                                    "fact_refs": [ref],
                                }
                            ],
                        )
                    elif operation != "status" and self.mode != "existing":
                        fields["missing_information"] = ["明确政策条件"]
                    calls = ()
                    d = RefundDecision.model_validate_json(canonical_json(fields))
                return RefundModelResponse(
                    model_step=model_step,
                    tool_calls=calls,
                    decision=d,
                    input_hash=audit.input_hash,
                    output_hash=canonical_hash(
                        {
                            "tool_calls": [c.model_dump(mode="json") for c in calls],
                            "decision": d.model_dump(mode="json") if d else None,
                        }
                    ),
                    audit=audit,
                )

        result = hybrid_result((3.0,))
        result = result.model_copy(
            update={
                "chunks": {
                    cid: body.model_copy(
                        update={
                            "chunk": body.chunk.model_copy(
                                update={
                                    "answer": "软件商品未激活可以申请整单退款。激活后不支持无理由退款。",
                                    "text": "退货\n软件商品未激活可以申请整单退款。激活后不支持无理由退款。",
                                }
                            )
                        }
                    )
                    for cid, body in result.chunks.items()
                }
            }
        )
        retrievers = retrieval_harness(result)
        retrievers.count_query_tokens = len
        original_retrieve = retrievers.retrieve

        def retrieve(*a, **kw):
            outside()
            counts["retrieve"] += 1
            return original_retrieve(*a, **kw)

        retrievers.retrieve = retrieve
        client, facts = Client(), Facts()
        service = RefundFlowService(
            h.service,
            client=client,
            retrievers=retrievers,
            identity=Identity(),
            facts=facts,
            registration=Reader(),
        )
        try:
            yield SimpleNamespace(
                **(
                    vars(h)
                    | {
                        "flow": service,
                        "client": client,
                        "facts": facts,
                        "counts": counts,
                        "business": business,
                        "profile": prof,
                    }
                )
            )
        finally:
            chat_engine.dispose()
            business_engine.dispose()


@pytest.fixture
def flow_harness(tmp_path):
    with flow_case(tmp_path) as h:
        yield h


def drive(h):
    rid, e = h.run.run_id, h.execution
    for method in (h.flow.bind, h.flow.prepare_query, h.flow.retrieve):
        method(rid, e)
    for _ in range(24):
        record = h.flow.main_agent(rid, e)
        if record.decision is not None or record.step_limit or record.scope_reply:
            return record
        record = h.flow.read_next(rid, e)
    raise AssertionError("Unbounded refund loop")
