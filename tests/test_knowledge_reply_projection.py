"""One persisted whitelist drives JSON, SSE and refresh, without internal profiles."""

import json

import pytest

from resolv.api.agent import response_for_events
from resolv.api.sse import conversation_sse
from resolv.repositories.knowledge_branch import KnowledgeBranchRepository
from resolv.schemas.branch_execution import ReplyMetadata, parse_reply_metadata
from resolv.schemas.conversation import ConversationError, public_reply_metadata
from resolv.services.knowledge_branch import knowledge_answer_reply
from resolv.services.knowledge_reply_projection import public_knowledge_reply
from tests.conversation_fixtures import MemoryCheckpoints, make_run
from tests.knowledge_branch_fixtures import (
    hybrid_result,
    knowledge_factory,
    make_knowledge_service,
)
from tests.test_knowledge_branch_service import advance

__all__ = ["knowledge_factory"]


def persisted(factory, *, fallback=False):
    h = make_knowledge_service(
        factory,
        MemoryCheckpoints(),
        **({"hybrid_result": hybrid_result((-10.0,))} if fallback else {}),
    )
    with h.conversation.execution(h.run.run_id) as execution:
        record = advance(h, execution)
        reply = (
            h.service.fallback(h.run.run_id, execution)
            if fallback
            else knowledge_answer_reply(
                record, h.service.answer(h.run.run_id, execution)
            )
        )
        h.service.finalize(h.run.run_id, execution, reply)
        h.conversation.mark_completed(h.run.run_id, execution.epoch)
    with factory() as session:
        record = KnowledgeBranchRepository(session, "local").get(h.run.run_id)
    return h, record


@pytest.mark.parametrize("fallback", [False, True])
def test_json_sse_and_detail_have_identical_persisted_references(
    knowledge_factory, fallback
):
    h, record = persisted(knowledge_factory, fallback=fallback)
    events = list(h.conversation.snapshot_events(h.run.run_id))
    http = response_for_events(h.conversation, h.run.run_id, iter(events)).model_dump(
        mode="json"
    )
    frames = list(conversation_sse(iter(events)))
    sse = next(
        json.loads(frame.split("data: ", 1)[1])
        for frame in frames
        if frame.startswith("event: message\n")
    )
    detail = h.conversation.detail(h.run.conversation_id)
    meta = next(r.result_json for r in detail.runs if r.run_id == h.run.run_id)
    assert http["knowledge"] == sse["knowledge"] == meta["knowledge"]
    assert meta["schema_version"] == 2
    assert "knowledge_config" not in detail.model_dump_json()
    assert all(
        key not in canonical_keys(meta)
        for key in ("system_prompt", "profile", "api_key", "self_check", "error_code")
    )
    if fallback:
        assert http["knowledge"]["citations"] == http["knowledge"]["chunks"] == []
        assert record.retrieval.bundle.windows
    else:
        assert http["knowledge"]["used_citation_numbers"] == [1]


def canonical_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(canonical_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(canonical_keys(v) for v in value))
    return set()


@pytest.mark.parametrize(
    "change",
    [
        "missing_hash",
        "wrong_hash",
        "unknown_citation",
        "release",
        "unfinished",
        "cross_run",
    ],
)
def test_projection_rejects_incomplete_or_cross_run_evidence(knowledge_factory, change):
    h, record = persisted(knowledge_factory)
    run = h.conversation.get_run(h.run.run_id)
    data = json.loads(json.dumps(run.result_json))
    if change == "missing_hash":
        del data["knowledge"]["evidence_hash"]
    elif change == "wrong_hash":
        data["knowledge"]["evidence_hash"] = "0" * 64
    elif change == "unknown_citation":
        data["knowledge"]["used_citation_numbers"] = [99]
    elif change == "release":
        data["knowledge"]["release_id"] = "other"
    elif change == "unfinished":
        run = run.model_copy(update={"status": "finalizing"})
    else:
        record = record.model_copy(update={"run_id": "foreign"})
    with pytest.raises(ConversationError, match="run_state_conflict"):
        public_knowledge_reply(run.model_copy(update={"result_json": data}), record)


def test_legacy_v1_keeps_its_exact_contract():
    run = make_run(orchestration_version="legacy-v1")
    assert "knowledge" not in public_reply_metadata(run)
    meta = ReplyMetadata(
        reply_kind="answer",
        understanding_id="u",
        primary_task_id="p",
        branch="business",
    )
    assert type(parse_reply_metadata(meta.model_dump(mode="json"))) is ReplyMetadata


def test_nonknowledge_v3_has_explicit_null_knowledge():
    run = make_run(
        status="completed",
        orchestration_version="routed-v3",
        understanding_id="u",
        result_json={
            "schema_version": 2,
            "reply_kind": "answer",
            "understanding_id": "u",
            "primary_task_id": "p",
            "branch": "business",
            "clarifies_understanding_id": None,
            "suggested_actions": [],
            "knowledge": None,
        },
    )
    assert public_knowledge_reply(run, None)["knowledge"] is None
