"""Actual LangGraph and repository integration, with isolated model boundaries."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from resolv.models.conversation import (
    Conversation,
    ConversationAction,
    ConversationRun,
)
from resolv.models.refund_application import RefundApplication
from resolv.models.refund_flow import ConversationRefundOffer
from tests.knowledge_branch_fixtures import branch_profile
from tests.refund_flow_fixtures import flow_case


@contextmanager
def harness(tmp_path, *, mode="agent", **kwargs):
    with flow_case(tmp_path, **kwargs) as h:
        h.service.orchestration_version = "routed-v4"
        h.service.refund_runtime = SimpleNamespace(flow=h.flow, profile=h.profile)
        h.service.refund_profile = h.profile
        h.service.knowledge_profile = branch_profile()
        h.service.release_selector_path = tmp_path / "absent-selector.json"
        with h.factory.begin() as s:
            row = s.get(ConversationRun, h.run.run_id)
            row.mode = mode
            from resolv.repositories.conversation import fingerprint
            from resolv.schemas.branch_execution import RoutingInput

            row.routing_input = RoutingInput().model_dump(mode="json")
            row.request_fingerprint = fingerprint(
                {
                    "question": h.run.question,
                    "mode": mode,
                    "routing_input": row.routing_input,
                }
            )
            if mode == "chat":
                from resolv.repositories.conversation import fingerprint

                row.tool_schema_fingerprint = fingerprint([])
    try:
        yield h
    finally:
        h.factory.kw["bind"].dispose()
        h.business.kw["bind"].dispose()


def events(h, resume=False):
    c = h.service
    result = list(
        c.resume(h.run.conversation_id, h.run.run_id) if resume else c.run(h.run.run_id)
    )
    return result


@pytest.mark.parametrize("mode", ["agent", "chat"])
def test_v4_refund_and_chat_zero_tools(tmp_path, mode):
    with harness(tmp_path, mode=mode) as h:
        result = events(h)
        assert result[-1].type == "done", [(e.type, e.data) for e in result]
        run = h.service.get_run(h.run.run_id)
        assert run.result_json["schema_version"] == 3
        assert (
            run.orchestration_version == "routed-v4" and run.state_schema_version == 4
        )
        with h.factory() as s:
            assert s.scalar(select(func.count()).select_from(ConversationAction)) == 0
            assert s.scalar(
                select(func.count()).select_from(ConversationRefundOffer)
            ) == (mode == "agent")
        with h.business() as s:
            assert s.scalar(select(func.count()).select_from(RefundApplication)) == 0
        assert not any(e.type == "approval_required" for e in result)
        if mode == "chat":
            assert (
                h.counts["decide"]
                == h.counts["query_refund_order"]
                == h.counts["retrieve"]
                == 0
            )
        else:
            assert h.counts["decide"] == 2 and h.counts["retrieve"] == 1
        snapshot = h.service.checkpoint_factory.saver.get_tuple(
            {"configurable": {"thread_id": run.conversation_id}}
        )
        assert snapshot.checkpoint["channel_values"]["phase"] == "end"


@pytest.mark.parametrize(
    "text", ["订单A001申请保修", "订单A001要换货", "订单A001申请维修"]
)
def test_v4_manual_draft_never_creates_offer_or_queries_facts(tmp_path, text):
    with harness(tmp_path, text=text) as h:
        result = events(h)
        assert result[-1].type == "done", [(e.type, e.data) for e in result]
        saved = h.service.get_run(h.run.run_id)
        assert saved.result_json["suggested_actions"][0]["preview_only"] is True
        assert h.counts["decide"] == h.counts["query_refund_order"] == 0


@pytest.mark.parametrize("retryable", [False, True])
def test_v4_other_branches_and_terminal_errors(tmp_path, retryable):
    from resolv.schemas.refund import RefundFlowError

    with harness(tmp_path) as h:
        h.flow.main_agent = lambda *a: (_ for _ in ()).throw(
            RefundFlowError("refund_model_budget_exceeded", retryable)
        )
        result = events(h)
        assert result[-1].type == "error"
        run = h.service.get_run(h.run.run_id)
        assert run.status == ("interrupted" if retryable else "failed")
        with h.factory() as s:
            assert (
                s.get(Conversation, run.conversation_id).current_run_id == run.run_id
            ) is retryable


def test_v4_loop_limit_uses_six_decisions_and_no_sixth_reads(tmp_path):
    with harness(tmp_path, mode="agent") as h:
        h.client.mode = "loop"
        result = events(h)
        assert result[-1].type == "done", [(e.type, e.data) for e in result]
        assert h.counts["decide"] == 6 and h.counts["query_refund_order"] == 5
        assert h.service.get_run(h.run.run_id).result_json["reply_kind"] == "step_limit"
