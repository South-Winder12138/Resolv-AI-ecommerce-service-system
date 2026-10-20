"""Faults straddle external work, stage commits and final commit receipts."""

import pytest
from sqlalchemy import func, select

from resolv.models.conversation import ConversationMessage, ConversationRun
from resolv.models.knowledge_branch import ConversationKnowledgeReviewItem
from resolv.repositories.knowledge_branch import KnowledgeBranchRepository
from resolv.schemas.knowledge_branch import KnowledgeBranchError
from tests.conversation_fixtures import MemoryCheckpoints
from tests.knowledge_branch_fixtures import (
    hybrid_result,
    knowledge_factory,
    make_knowledge_service,
)
from tests.test_knowledge_branch_service import advance

__all__ = ["knowledge_factory"]


def fail_at(name):
    def fail(point):
        if point == name:
            raise RuntimeError("injected:" + point)

    return fail


@pytest.mark.parametrize(
    "stage", ["query", "retrieval", "check", "answer_input", "answer"]
)
def test_resume_after_saved_stage_does_not_repeat_saved_work(knowledge_factory, stage):
    h = make_knowledge_service(knowledge_factory, MemoryCheckpoints())
    h.fault_hook = fail_at("knowledge_after_" + stage + "_saved")
    with (
        h.conversation.execution(h.run.run_id) as execution,
        pytest.raises(RuntimeError, match="injected"),
    ):
        advance(h, execution)
        h.service.answer(h.run.run_id, execution)
    before = dict(h.counts)
    h.fault_hook = None
    with h.conversation.execution(h.run.run_id) as execution:
        advance(h, execution)
        h.service.answer(h.run.run_id, execution)
    for purpose, count in before.items():
        assert h.counts[purpose] == count
    assert h.counts["knowledge_answer"] == 1


def test_before_stage_save_can_repeat_external_call_with_audit_retained(
    knowledge_factory,
):
    h = make_knowledge_service(knowledge_factory, MemoryCheckpoints())
    h.fault_hook = fail_at("knowledge_before_query_saved")
    with (
        h.conversation.execution(h.run.run_id) as execution,
        pytest.raises(RuntimeError),
    ):
        h.service.prepare_query(h.run.run_id, execution)
    h.fault_hook = None
    with h.conversation.execution(h.run.run_id) as execution:
        record = h.service.prepare_query(h.run.run_id, execution)
    assert h.counts["knowledge_query"] == 2 and len(record.audits) == 4


def test_unknown_external_call_window_is_retained_when_preparation_repeats(
    knowledge_factory,
):
    h = make_knowledge_service(
        knowledge_factory,
        MemoryCheckpoints(),
        query_reply=KnowledgeBranchError("knowledge_model_unavailable", True),
    )
    h.fault_hook = fail_at("knowledge_before_query_saved")
    with (
        h.conversation.execution(h.run.run_id) as execution,
        pytest.raises(RuntimeError),
    ):
        h.service.prepare_query(h.run.run_id, execution)
    h.fault_hook = None
    h.responses["knowledge_query"] = {"standard": "退货运费由谁承担？", "expanded": []}
    with h.conversation.execution(h.run.run_id) as execution:
        record = h.service.prepare_query(h.run.run_id, execution)
    assert h.counts["knowledge_query"] == 2
    assert [a.outcome for a in record.audits] == [
        "started",
        "unknown",
        "started",
        "succeeded",
    ]


@pytest.mark.parametrize(
    "point,committed",
    [
        ("knowledge_before_finalize_commit", False),
        ("knowledge_after_finalize_commit", True),
    ],
)
def test_review_reply_and_metadata_commit_atomically(
    knowledge_factory, point, committed
):
    h = make_knowledge_service(
        knowledge_factory, MemoryCheckpoints(), hybrid_result=hybrid_result((-10.0,))
    )
    with h.conversation.execution(h.run.run_id) as execution:
        advance(h, execution)
        reply = h.service.fallback(h.run.run_id, execution)
        h.fault_hook = fail_at(point)
        if committed:
            h.service.finalize(h.run.run_id, execution, reply)
        else:
            with pytest.raises(KnowledgeBranchError):
                h.service.finalize(h.run.run_id, execution, reply)
        h.fault_hook = None
    with knowledge_factory() as session:
        answer = session.get(ConversationMessage, h.run.answer_message_id)
        review_count = session.scalar(
            select(func.count()).select_from(ConversationKnowledgeReviewItem)
        )
        run = session.get(ConversationRun, h.run.run_id)
        assert bool(answer) == bool(review_count) == bool(run.result_json) == committed
        if not committed:
            assert (
                KnowledgeBranchRepository(session, "local").get(h.run.run_id).error_code
                == "knowledge_finalize_store_failed"
            )
    with h.conversation.execution(h.run.run_id) as execution:
        h.service.finalize(h.run.run_id, execution, reply)
    with knowledge_factory() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(ConversationKnowledgeReviewItem)
            )
            == 1
        )
