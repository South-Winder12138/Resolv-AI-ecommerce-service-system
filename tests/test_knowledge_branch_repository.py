"""Real offline SQL transactions, identity fences and immutable stage writes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from resolv.models.conversation import ConversationMessage, ConversationRun
from resolv.models.knowledge_branch import ConversationKnowledgeReviewItem
from resolv.repositories.conversation import ConversationRepository
from resolv.repositories.knowledge_branch import KnowledgeBranchRepository
from resolv.schemas.conversation import ConversationError
from resolv.schemas.knowledge_branch import (
    KnowledgeBranchError,
    KnowledgeReviewItem,
    PreparedKnowledgeQuery,
)
from resolv.services.hybrid_fingerprints import canonical_hash
from tests.knowledge_branch_fixtures import bundle, knowledge_factory, knowledge_owned

__all__ = ["knowledge_factory", "knowledge_owned"]


def create(h):
    with h.factory.begin() as session:
        return KnowledgeBranchRepository(session, "local").create(
            h.run.run_id, h.execution.epoch, h.primary, h.profile
        )


def query(h, **changes):
    data = {
        "standard": h.primary.primary.text,
        "expanded": (),
        "bm25_text": h.primary.primary.text,
        "status": "prepared",
        "input_hash": canonical_hash(h.primary.model_dump(mode="json")),
    }
    return PreparedKnowledgeQuery(**(data | changes))


def test_create_is_owned_frozen_and_idempotent(knowledge_owned):
    h = knowledge_owned
    record = create(h)
    assert record == create(h)
    assert record.primary_task_id == h.primary.primary.task_id
    with h.factory() as session:
        assert KnowledgeBranchRepository(session, "other").get(h.run.run_id) is None
    assert "knowledge_config" not in h.service.get_run(h.run.run_id).model_dump()
    with (
        pytest.raises((ConversationError, KnowledgeBranchError)),
        h.factory.begin() as session,
    ):
        KnowledgeBranchRepository(session, "other").create(
            h.run.run_id, h.execution.epoch, h.primary, h.profile
        )


@pytest.mark.parametrize(
    "change", ["epoch", "profile", "understanding", "task", "text", "snapshot"]
)
def test_write_rejects_changed_identity(knowledge_owned, change):
    h = knowledge_owned
    create(h)
    with h.factory.begin() as session:
        row = session.get(ConversationRun, h.run.run_id)
        if change == "profile":
            row.knowledge_config = {}
        if change == "understanding":
            row.understanding_id = None
    p = h.primary
    if change in {"task", "text"}:
        p = p.model_copy(
            update={
                "primary": p.primary.model_copy(
                    update={"task_id" if change == "task" else "text": "changed"}
                )
            }
        )
    if change == "snapshot":
        p = p.model_copy(update={"snapshot_hash": "0" * 64})
    with (
        pytest.raises((ConversationError, KnowledgeBranchError)),
        h.factory.begin() as session,
    ):
        KnowledgeBranchRepository(session, "local").create(
            h.run.run_id, h.execution.epoch - (change == "epoch"), p, h.profile
        )


def test_same_stage_is_idempotent_but_changed_payload_or_input_is_rejected(
    knowledge_owned,
):
    h = knowledge_owned
    record = create(h)
    q = query(h)
    for _ in range(2):
        with h.factory.begin() as session:
            saved = KnowledgeBranchRepository(session, "local").save_stage(
                h.run.run_id,
                h.execution.epoch,
                "query",
                q,
                expected_input_hash=record.primary_hash,
            )
    assert saved.query == q and len(saved.stages) == 1
    for q2, input_hash in [
        (query(h, standard="其他", bm25_text="其他"), record.primary_hash),
        (q, "0" * 64),
    ]:
        with pytest.raises(KnowledgeBranchError), h.factory.begin() as session:
            KnowledgeBranchRepository(session, "local").save_stage(
                h.run.run_id,
                h.execution.epoch,
                "query",
                q2,
                expected_input_hash=input_hash,
            )


def test_missing_predecessor_is_rejected(knowledge_owned):
    h = knowledge_owned
    create(h)
    from resolv.schemas.hybrid_knowledge import FrozenFileConfig
    from resolv.schemas.knowledge_branch import KnowledgeRetrievalStage
    from tests.knowledge_branch_fixtures import hybrid_result

    result = KnowledgeRetrievalStage.from_result(
        hybrid_result(),
        bundle=bundle(),
        file_configs=(("d" * 64, FrozenFileConfig(content_type="policy")),),
    )
    with pytest.raises(KnowledgeBranchError), h.factory.begin() as session:
        KnowledgeBranchRepository(session, "local").save_stage(
            h.run.run_id,
            h.execution.epoch,
            "retrieval",
            result,
            expected_input_hash="a" * 64,
        )


@pytest.mark.parametrize(
    "field,value",
    [("primary_hash", "0" * 64), ("owner_id", "other"), ("query_input_hash", "0" * 64)],
)
def test_stored_identity_or_partial_stage_hash_corruption_is_rejected(
    knowledge_owned, field, value
):
    from resolv.models.knowledge_branch import ConversationKnowledgeRun

    h = knowledge_owned
    create(h)
    with h.factory.begin() as session:
        setattr(session.get(ConversationKnowledgeRun, h.run.run_id), field, value)
    with pytest.raises(KnowledgeBranchError), h.factory() as session:
        KnowledgeBranchRepository(session, "local").get(h.run.run_id)


def test_two_sessions_first_save_keeps_one_consistent_result(knowledge_owned):
    h = knowledge_owned
    record = create(h)
    q = query(h)

    def save(_):
        with h.factory.begin() as session:
            return KnowledgeBranchRepository(session, "local").save_stage(
                h.run.run_id,
                h.execution.epoch,
                "query",
                q,
                expected_input_hash=record.primary_hash,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(save, range(2)))
    assert records[0] == records[1] and len(records[0].stages) == 1


def test_review_and_reply_rollback_together(knowledge_owned):
    h = knowledge_owned
    create(h)
    item = KnowledgeReviewItem(
        run_id=h.run.run_id,
        conversation_id=h.run.conversation_id,
        understanding_id=h.run.understanding_id,
        primary_task_id=h.primary.primary.task_id,
        original_question=h.run.question,
        primary_text=h.primary.primary.text,
        standard=h.primary.primary.text,
        source="retrieval_low_conf",
        total=0.0,
        self_check_reason=None,
        evidence=bundle(()),
        created_at=datetime.now(UTC),
    )
    with pytest.raises(RuntimeError, match="rollback"), h.factory.begin() as session:
        KnowledgeBranchRepository(session, "local").add_review(
            h.run.run_id, h.execution.epoch, item
        )
        # Task10 adds v3 metadata support; this step verifies the existing transaction boundary.
        from resolv.schemas.branch_execution import ReplyMetadata

        meta = ReplyMetadata(
            reply_kind="answer",
            understanding_id=h.run.understanding_id,
            primary_task_id=h.primary.primary.task_id,
            branch="knowledge",
        )
        ConversationRepository(session).commit_answer(
            h.run.run_id, h.execution.epoch, "兜底", metadata=meta
        )
        raise RuntimeError("rollback")
    with h.factory() as session:
        assert session.get(ConversationMessage, h.run.answer_message_id) is None
        assert (
            session.scalar(
                select(func.count()).select_from(ConversationKnowledgeReviewItem)
            )
            == 0
        )


def test_audit_is_idempotent_and_requires_owner_epoch(knowledge_owned):
    from resolv.schemas.knowledge_branch import KnowledgeModelAudit

    h = knowledge_owned
    create(h)
    audit = KnowledgeModelAudit(
        invocation_id="invoke1",
        purpose="knowledge_query",
        profile_fingerprint=h.profile.query_model.fingerprint,
        input_hash="a" * 64,
        input_estimate=1,
        started_at=datetime.now(UTC),
        outcome="started",
    )
    for _ in range(2):
        with h.factory.begin() as session:
            KnowledgeBranchRepository(session, "local").record_audit(
                h.run.run_id, h.execution.epoch, audit
            )
    with h.factory() as session:
        assert KnowledgeBranchRepository(session, "local").get(h.run.run_id).audits == (
            audit,
        )
    with pytest.raises(ConversationError), h.factory.begin() as session:
        KnowledgeBranchRepository(session, "local").record_audit(
            h.run.run_id, h.execution.epoch - 1, audit
        )


@pytest.mark.parametrize(
    "microsecond,rounded", [(200191, False), (800191, False), (800191, True)]
)
def test_task_identity_accepts_only_valid_mysql_second_precision_forms(
    microsecond, rounded
):
    from datetime import timedelta

    from resolv.repositories.knowledge_branch import same_task_identity
    from tests.knowledge_branch_fixtures import primary

    original = primary().primary
    original = original.model_copy(
        update={"created_at": original.created_at.replace(microsecond=microsecond)}
    )
    timestamp = original.created_at.replace(microsecond=0) + (
        timedelta(seconds=1) if rounded else timedelta()
    )
    persisted = original.model_copy(update={"created_at": timestamp})
    assert same_task_identity(persisted, original)
    if microsecond < 500000:
        assert not same_task_identity(
            persisted.model_copy(
                update={"created_at": timestamp + timedelta(seconds=1)}
            ),
            original,
        )
    assert not same_task_identity(
        persisted.model_copy(update={"text": "changed primary"}), original
    )
    assert not same_task_identity(
        persisted.model_copy(update={"created_at": timestamp + timedelta(seconds=2)}),
        original,
    )
    assert not same_task_identity(
        persisted.model_copy(update={"created_at": timestamp.replace(microsecond=1)}),
        original,
    )
