"""Coordination of independent understanding and conversation records."""

from sqlalchemy import select

from resolv.models.conversation import Conversation, ConversationRun
from resolv.schemas.conversation import ConversationError


def validate_understanding_store(factory, owner_id, understanding):
    if understanding is None or understanding.owner_id != owner_id:
        raise ValueError("routed understanding owner required")
    chat_bind, understanding_bind = (
        factory.kw.get("bind"),
        understanding.session_factory.kw.get("bind"),
    )
    if (
        chat_bind is None
        or understanding_bind is None
        or chat_bind.url != understanding_bind.url
        or (
            chat_bind.url.database in {None, ":memory:"}
            and chat_bind is not understanding_bind
        )
    ):
        raise ValueError("understanding and chat must share the logical store")


def accept_understanding(
    session,
    *,
    owner_id,
    run_id,
    conversation_id,
    question,
    routing_input,
    understanding,
):
    from resolv.repositories.understanding import UnderstandingRepository
    from resolv.schemas.understanding import UnderstandingError, UnderstandingInput

    if (
        understanding is None
        or understanding.owner_id != owner_id
        or session.get_bind().url != understanding.session_factory.kw["bind"].url
    ):
        raise ConversationError("understanding_unavailable", True, run_id)
    request = UnderstandingInput(
        request_id="chat:" + run_id, text=question,
        **routing_input.model_dump(exclude={"reassesses_refund_offer_id"})
    )
    try:
        record, _ = UnderstandingRepository(session, owner_id).create(
            conversation_id,
            request,
            lambda sources: understanding.context.freeze(sources, request),
        )
        return record
    except UnderstandingError as error:
        raise ConversationError(error.code, error.retryable, run_id) from None


def load_primary(run, view, *, snapshot_hash):
    from resolv.schemas.branch_execution import PrimaryExecution
    from resolv.understanding_rules import INTENT_BRANCHES_V1

    p = view.primary_request
    if (
        run.orchestration_version not in {"routed-v2", "routed-v3", "routed-v4"}
        or run.understanding_id != view.understanding_id
        or run.conversation_id != view.conversation_id
        or run.question != view.original_text
        or view.status != "ready"
        or not view.route_ready
        or p is None
        or p.conversation_id != run.conversation_id
        or p.understanding_id != run.understanding_id
        or p.role != "primary"
        or not p.task_id
        or not p.evidence_refs
        or p.intent is None
        or INTENT_BRANCHES_V1.get(p.intent) != view.branch
        or p.branch != view.branch
        or p.ambiguities
        or p.unresolved_references
        or not snapshot_hash
    ):
        raise ConversationError("understanding_state_conflict", run_id=run.run_id)
    return PrimaryExecution(
        conversation_id=run.conversation_id,
        understanding_id=view.understanding_id,
        snapshot_hash=snapshot_hash,
        primary=p,
    )


def run_understanding(run, understanding):
    from resolv.schemas.understanding import UnderstandingError

    if (
        understanding is None
        or run.orchestration_version not in {"routed-v2", "routed-v3", "routed-v4"}
        or not run.understanding_id
    ):
        raise ConversationError("understanding_unavailable", True, run.run_id)
    try:
        record = understanding.record(run.conversation_id, run.understanding_id)
        understanding.compatible(record)
        if record.original_text != run.question:
            raise ConversationError("understanding_state_conflict", run_id=run.run_id)
        view = understanding.run(run.conversation_id, run.understanding_id)
    except UnderstandingError as error:
        raise ConversationError(error.code, error.retryable, run.run_id) from None
    if view.status not in {"ready", "needs_clarification"}:
        raise ConversationError("understanding_not_ready", True, run.run_id)
    if view.route_ready:
        load_primary(run, view, snapshot_hash=record.snapshot.snapshot_hash)
    elif (
        view.conversation_id != run.conversation_id
        or view.understanding_id != run.understanding_id
        or view.original_text != run.question
        or not view.clarification_question
    ):
        raise ConversationError("understanding_state_conflict", run_id=run.run_id)
    return view


def assert_no_legacy_inflight(session, *, owner_id: str) -> None:
    identity = session.scalar(
        select(ConversationRun.run_id)
        .join(Conversation)
        .where(
            Conversation.owner_id == owner_id,
            ConversationRun.orchestration_version == "legacy-v1",
            ConversationRun.status.not_in(["completed", "failed"]),
        )
        .limit(1)
    )
    if identity is not None:
        raise ConversationError("legacy_run_requires_resolution", run_id=identity)
