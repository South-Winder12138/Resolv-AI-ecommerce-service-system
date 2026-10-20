"""Independent-review regressions; recorded provider output is evidence of a miss."""

import json
from pathlib import Path

import pytest

from resolv.services.knowledge_branch_runtime import read_release_selection
from tests.knowledge_branch_fixtures import knowledge_factory
from tests.test_knowledge_routing_graph import harness

__all__ = ["knowledge_factory"]


def test_non_utf8_selector_is_invalid_configuration(tmp_path):
    path = tmp_path / "selector.json"
    path.write_bytes(b"\xff\xfe")
    assert read_release_selection(path).status == "invalid"


@pytest.mark.parametrize("error_code", ["query_too_long", "pair_too_long"])
def test_uneditable_length_failure_releases_conversation_without_fallback(
    knowledge_factory, tmp_path, error_code
):
    from resolv.schemas.hybrid_knowledge import HybridRetrievalError

    h = harness(knowledge_factory, tmp_path)
    if error_code == "query_too_long":
        h.retrievers.count_query_tokens = lambda text: 513
    else:

        def fail(*a, **kw):
            raise HybridRetrievalError(error_code)

        h.retrievers.retrieve = fail
    events = list(h.conversation.run(h.run.run_id))
    assert events[-1].type == "error" and events[-1].data["code"] == error_code
    saved = h.conversation.get_run(h.run.run_id)
    assert saved.status == "failed" and saved.retryable is False
    assert h.conversation.detail(h.run.conversation_id).current_run is None
    assert saved.result_json is None and h.counts["knowledge_answer"] == 0
    new, created = h.conversation.accept(
        h.run.conversation_id, "shortened", "退货政策", "agent"
    )
    assert created and new.run_id != saved.run_id


def test_recorded_refund_time_answer_cannot_pass_full_question_gate():
    from scripts.smoke_knowledge_branch import validate_formal_case

    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root
            / "docs/superpowers/acceptance/2026-10-15-knowledge-branch/reports/K2.json"
        ).read_text()
    )
    case = next(
        c for c in report["reports"][1]["cases"] if c["case_id"] == "refund-policy-time"
    )
    expected = next(
        c
        for c in json.loads(
            (root / "tests/fixtures/knowledge_branch/formal_cases.json").read_text()
        )
        if c["case_id"] == case["case_id"]
    )
    with pytest.raises(ValueError):
        validate_formal_case(case, expected)


@pytest.mark.parametrize("gate", ["snapshot_python", "snapshot_frontend"])
def test_activation_rejects_placeholder_snapshot_checks(gate):
    from types import SimpleNamespace

    from resolv.services.knowledge_release import validate_gate_report

    candidate = "a" * 40
    with pytest.raises(ValueError):
        validate_gate_report(
            gate,
            {"revision": candidate, "tree": "b" * 40, "checks": {"test": 0}},
            SimpleNamespace(candidate_commit=candidate),
        )


def test_activation_rejects_review_hash_without_actual_document():
    from types import SimpleNamespace

    from resolv.services.knowledge_release import validate_gate_report

    candidate = "a" * 40
    with pytest.raises(ValueError):
        validate_gate_report(
            "review",
            {
                "candidate_commit": candidate,
                "important_count": 0,
                "reviewer_id": "fixture",
                "review_document_sha256": "f" * 64,
            },
            SimpleNamespace(candidate_commit=candidate),
        )


def test_formal_gate_rejects_missing_approved_cases():
    from resolv.services.knowledge_gate_validation import validate_knowledge_gate
    from tests.test_smoke_knowledge_branch import positive_report

    r = positive_report()
    r["knowledge_scope"] = "formal-readonly"
    with pytest.raises(ValueError):
        validate_knowledge_gate(r)


@pytest.mark.parametrize(
    "raw", [b"\xff\xfe", '{"enabled":false,"config_version":1}'.encode("utf-16")]
)
@pytest.mark.parametrize("intent", ["chitchat", "product_consultation"])
def test_invalid_selector_only_fails_knowledge_after_accept(
    knowledge_factory, tmp_path, raw, intent
):
    from langchain_core.messages import AIMessage

    from tests.understanding_fixtures import make_classification, make_resolution

    h = harness(knowledge_factory, tmp_path)
    h.conversation.release_selector_path.write_bytes(raw)
    question = "你好" if intent == "chitchat" else "退货政策"
    h.conversation.understanding.resolver.model.client.replies.extend(
        [
            AIMessage(content=x.model_dump_json())
            for x in (
                make_resolution(resolved_text=question),
                make_classification([question], [intent]),
            )
        ]
    )
    c = h.conversation.create()
    run, _ = h.conversation.accept(
        c.conversation_id, "invalid-selector", question, "agent"
    )
    assert run.knowledge_config["selection"]["status"] == "invalid"
    events = list(h.conversation.run(run.run_id))
    if intent == "chitchat":
        assert (
            events[-1].type == "done"
            and h.conversation.get_run(run.run_id).status == "completed"
        )
    else:
        assert (
            events[-1].type == "error"
            and events[-1].data["code"] == "knowledge_release_selector_invalid"
        )
        assert h.conversation.get_run(run.run_id).status == "failed"
    assert (
        h.counts["knowledge_query"]
        == h.counts["retrieval"]
        == h.counts["knowledge_answer"]
        == 0
    )


def test_frozen_evidence_budget_failure_releases_conversation(
    knowledge_factory, tmp_path
):
    from types import SimpleNamespace

    h = harness(knowledge_factory, tmp_path)
    h.service._counter = lambda profile: SimpleNamespace(count_text=lambda _: 3001)
    events = list(h.conversation.run(h.run.run_id))
    assert (
        events[-1].type == "error"
        and events[-1].data["code"] == "evidence_budget_exceeded"
    )
    assert h.conversation.get_run(h.run.run_id).status == "failed"
    assert h.conversation.detail(h.run.conversation_id).current_run is None
    assert h.counts["knowledge_answer"] == 0
    assert h.conversation.accept(
        h.run.conversation_id, "shorter", "请简述退货", "agent"
    )[1]


def test_formal_fixture_matches_frozen_case_contract():
    from resolv.schemas.knowledge_gate_cases import FORMAL_CASES

    path = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/knowledge_branch/formal_cases.json"
    )
    assert json.loads(path.read_text()) == list(FORMAL_CASES)


def test_permanent_frozen_answer_budget_error_public_retryability_matches_terminal(
    knowledge_factory, tmp_path
):
    from types import SimpleNamespace

    from resolv.schemas.knowledge_branch import KnowledgeBranchError

    h = harness(knowledge_factory, tmp_path)

    def fail(*a, **kw):
        raise KnowledgeBranchError("answer_context_budget_exceeded", True)

    h.service.clients["knowledge_answer"] = SimpleNamespace(
        invoke_answer=fail,
        counter_factory=h.service.clients["knowledge_answer"].counter_factory,
    )
    events = list(h.conversation.run(h.run.run_id))
    assert events[-1].data["code"] == "answer_context_budget_exceeded", [
        (e.type, e.data) for e in events
    ]
    assert h.conversation.get_run(h.run.run_id).status == "failed"
    assert events[-1].data["retryable"] is False


def test_fixed_answer_context_overflow_uses_actual_builder_before_memory(
    knowledge_factory, tmp_path
):
    from resolv.models.conversation import ConversationRun

    h = harness(knowledge_factory, tmp_path)
    smaller = h.profile.model_copy(
        update={
            "answer_model": h.profile.answer_model.model_copy(
                update={"app_max_input_tokens": 7000}
            )
        }
    )
    with knowledge_factory.begin() as session:
        session.get(
            ConversationRun, h.run.run_id
        ).knowledge_config = smaller.model_dump(mode="json")
    # Real ContextBuilder and prepare_memory, real fixture counter; no injected budget exception.
    events = list(h.conversation.run(h.run.run_id))
    assert (
        events[-1].type == "error"
        and events[-1].data["code"] == "answer_context_budget_exceeded"
    ), [(e.type, e.data) for e in events]
    saved = h.conversation.get_run(h.run.run_id)
    assert saved.status == "failed" and not saved.memory_prepared
    assert events[-1].data["retryable"] is False and h.counts["knowledge_answer"] == 0
    assert h.conversation.detail(h.run.conversation_id).current_run is None
    assert h.conversation.accept(
        h.run.conversation_id, "new-question", "请介绍会员权益", "agent"
    )[1]


def test_memory_failure_with_fitting_fixed_context_remains_recoverable(
    knowledge_factory, tmp_path
):
    from resolv.services.context import ContextBudgetExceeded

    h = harness(knowledge_factory, tmp_path)

    def unavailable(*a, **kw):
        raise ContextBudgetExceeded()

    h.conversation.prepare_memory = unavailable
    events = list(h.conversation.run(h.run.run_id))
    assert (
        events[-1].data["code"] == "context_budget_exceeded"
        and events[-1].data["retryable"] is True
    )
    assert h.conversation.get_run(h.run.run_id).status == "interrupted"
    assert h.conversation.detail(h.run.conversation_id).current_run is not None
