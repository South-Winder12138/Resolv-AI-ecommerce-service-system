"""Persisted gates, model/tool call counts and atomic publication."""

import pytest
from sqlalchemy import func, select

from resolv.models.conversation import ConversationMessage, ConversationRun
from resolv.models.knowledge_branch import ConversationKnowledgeReviewItem
from resolv.repositories.knowledge_branch import KnowledgeBranchRepository
from resolv.schemas.knowledge_branch import KnowledgeBranchError
from resolv.services.knowledge_branch import knowledge_answer_reply
from tests.conversation_fixtures import MemoryCheckpoints
from tests.knowledge_branch_fixtures import (
    hybrid_result,
    knowledge_factory,
    make_knowledge_service,
)

__all__ = ["knowledge_factory"]


def advance(h, execution):
    h.service.prepare_query(h.run.run_id, execution)
    h.service.retrieve(h.run.run_id, execution)
    return h.service.check(h.run.run_id, execution)


def test_success_stages_are_reused_and_do_not_call_business_tools(knowledge_factory):
    h = make_knowledge_service(knowledge_factory, MemoryCheckpoints())
    with h.conversation.execution(h.run.run_id) as execution:
        record = advance(h, execution)
        answer = h.service.answer(h.run.run_id, execution)
        h.service.finalize(
            h.run.run_id, execution, knowledge_answer_reply(record, answer)
        )
        assert h.conversation.get_run(h.run.run_id).status == "finalizing"
        advance(h, execution)
        assert h.service.answer(h.run.run_id, execution) == answer
    assert dict(h.counts) == {
        "knowledge_query": 1,
        "retrieval": 1,
        "knowledge_check": 1,
        "knowledge_answer": 1,
    }
    with knowledge_factory() as session:
        record = KnowledgeBranchRepository(session, "local").get(h.run.run_id)
        assert len(record.stages) == 5 and len(record.audits) == 6
        assert (
            session.scalar(
                select(func.count()).select_from(ConversationKnowledgeReviewItem)
            )
            == 0
        )


@pytest.mark.parametrize("low", [True, False])
def test_insufficient_evidence_has_no_answer_or_summary_and_one_review(
    knowledge_factory, low
):
    kwargs = (
        {"hybrid_result": hybrid_result((-10.0,))}
        if low
        else {
            "check_reply": {
                "useful": False,
                "supporting_window_ids": [],
                "missing_information": ["运费"],
                "reason": "缺关键条款",
            }
        }
    )
    h = make_knowledge_service(knowledge_factory, MemoryCheckpoints(), **kwargs)
    with h.conversation.execution(h.run.run_id) as execution:
        record = advance(h, execution)
        assert not record.check.evidence_strong
        reply = h.service.fallback(h.run.run_id, execution)
        h.service.finalize(h.run.run_id, execution, reply)
        h.service.finalize(h.run.run_id, execution, reply)
    assert h.counts["knowledge_query"] == h.counts["retrieval"] == 1
    assert h.counts["knowledge_check"] == (0 if low else 1)
    assert (
        h.counts["knowledge_answer"]
        == h.counts["summary"]
        == h.counts["business_tools"]
        == 0
    )
    assert reply.metadata.suggested_actions[0].preview_only is True
    with knowledge_factory() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(ConversationKnowledgeReviewItem)
            )
            == 1
        )
        assert session.get(
            ConversationMessage, h.run.answer_message_id
        ).text.startswith("目前的知识资料还不足")


def test_retrieval_fault_is_not_low_confidence_or_review_item(knowledge_factory):
    h = make_knowledge_service(knowledge_factory, MemoryCheckpoints())
    h.retrievers.error = KnowledgeBranchError("retrieval_failure", True)
    with h.conversation.execution(h.run.run_id) as execution:
        h.service.prepare_query(h.run.run_id, execution)
        with pytest.raises(KnowledgeBranchError, match="retrieval_failure"):
            h.service.retrieve(h.run.run_id, execution)
    with knowledge_factory() as session:
        record = KnowledgeBranchRepository(session, "local").get(h.run.run_id)
        assert record.error_code == "retrieval_failure" and record.retrieval is None
        assert (
            session.scalar(
                select(func.count()).select_from(ConversationKnowledgeReviewItem)
            )
            == 0
        )


def test_saved_profile_tampering_blocks_reuse(knowledge_factory):
    h = make_knowledge_service(knowledge_factory, MemoryCheckpoints())
    with h.conversation.execution(h.run.run_id) as execution:
        h.service.prepare_query(h.run.run_id, execution)
        with knowledge_factory.begin() as session:
            session.get(ConversationRun, h.run.run_id).knowledge_config = {}
        with pytest.raises(KnowledgeBranchError):
            h.service.prepare_query(h.run.run_id, execution)
    assert h.counts["knowledge_query"] == 1


@pytest.mark.parametrize("purpose", ["check", "answer"])
def test_selfcheck_or_generation_failure_never_creates_review_item(
    knowledge_factory, purpose
):
    from langchain_core.messages import AIMessage

    kwargs = (
        {"check_reply": KnowledgeBranchError("knowledge_model_unavailable", True)}
        if purpose == "check"
        else {
            "answer_reply": AIMessage(
                content="无引用的回答", response_metadata={"finish_reason": "stop"}
            )
        }
    )
    h = make_knowledge_service(knowledge_factory, MemoryCheckpoints(), **kwargs)
    with h.conversation.execution(h.run.run_id) as execution:
        h.service.prepare_query(h.run.run_id, execution)
        h.service.retrieve(h.run.run_id, execution)
        with pytest.raises(KnowledgeBranchError):
            h.service.check(h.run.run_id, execution)
            h.service.answer(h.run.run_id, execution)
    with knowledge_factory() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(ConversationKnowledgeReviewItem)
            )
            == 0
        )
        assert session.get(ConversationMessage, h.run.answer_message_id) is None
