"""Freeze complete answer context and reject incomplete or invalid citations."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from resolv.schemas.knowledge_branch import KnowledgeBranchError
from resolv.services.knowledge_branch_answer import (
    build_knowledge_answer_input,
    generate_knowledge_answer,
)
from tests.conversation_fixtures import make_turn
from tests.knowledge_branch_fixtures import branch_profile, bundle, primary
from tests.test_context import Counter
from tests.test_knowledge_branch_query import Client, prepare


def inputs(*, history=(), profile=None, evidence=None):
    p = primary()
    q = prepare(Client({"standard": p.primary.text, "expanded": []}), p)
    b = evidence or bundle((2.0, 1.0))
    prof = profile or branch_profile()
    prof = prof.model_copy(
        update={
            "answer_model": prof.answer_model.model_copy(
                update={"provider_input_overhead_tokens": 0}
            )
        }
    )
    run = SimpleNamespace(
        conversation_id=p.conversation_id,
        understanding_id=p.understanding_id,
        run_id="r",
    )
    frozen = build_knowledge_answer_input(
        run, p, q, b, summary=None, history=history, profile=prof, counter=Counter()
    )
    return frozen, b, prof


class Answers:
    def __init__(self, reply):
        self.reply, self.calls, self.tool_bindings = reply, [], []

    def invoke_answer(self, messages, **kwargs):
        self.calls.append(messages)
        return self.reply


def test_full_evidence_is_preserved_and_input_is_immutable():
    frozen, b, _ = inputs()
    messages = frozen.messages.value
    assert b.serialized in [m["content"] for m in messages]
    assert frozen.evidence_hash == b.evidence_hash
    messages.clear()
    assert frozen.messages.value
    assert frozen.input_estimate <= frozen.input_limit


def test_history_is_removed_in_whole_turns_without_removing_evidence():
    history = [
        make_turn(question="a" * 10000, answer="b" * 10000),
        make_turn(seq=3, question="recent", answer="ok"),
    ]
    frozen, b, _ = inputs(history=history)
    texts = [m["content"] for m in frozen.messages.value]
    assert "a" * 10000 not in texts and "b" * 10000 not in texts
    assert "recent" in texts and "ok" in texts
    assert b.serialized in texts and len(frozen.history_refs) == 2


def test_core_context_cannot_fit_is_error():
    profile = branch_profile()
    profile = profile.model_copy(
        update={
            "answer_model": profile.answer_model.model_copy(
                update={"app_max_input_tokens": 100}
            )
        }
    )
    with pytest.raises(KnowledgeBranchError, match="answer_context_budget_exceeded"):
        inputs(profile=profile)


def test_conditional_answer_keeps_stable_used_citation_numbers():
    frozen, b, profile = inputs()
    client = Answers(
        AIMessage(
            content="质量问题由商家承担[2]，个人原因由用户承担[1]。[2]",
            response_metadata={"finish_reason": "stop"},
        )
    )
    answer = generate_knowledge_answer(
        frozen, b, client=client, profile=profile, audit_hook=lambda _: None
    )
    assert answer.used_citation_numbers == (1, 2)
    assert answer.evidence_hash == frozen.evidence_hash == b.evidence_hash
    assert client.tool_bindings == [] and len(client.calls) == 1


@pytest.mark.parametrize(
    "text", ["缺少引用", "错误[0]", "越界[3]", "前导零[01]", "有效[1]但又越界[100]"]
)
def test_missing_or_invalid_citations_are_technical_errors(text):
    frozen, b, profile = inputs()
    with pytest.raises(KnowledgeBranchError, match="answer_citation_invalid"):
        generate_knowledge_answer(
            frozen,
            b,
            client=Answers(
                AIMessage(content=text, response_metadata={"finish_reason": "stop"})
            ),
            profile=profile,
            audit_hook=lambda _: None,
        )


@pytest.mark.parametrize(
    "reply",
    [
        AIMessage(content=""),
        AIMessage(content="ok[1]", response_metadata={"finish_reason": "length"}),
        AIMessage(
            content="ok[1]",
            tool_calls=[{"name": "query_order", "args": {}, "id": "t"}],
            response_metadata={"finish_reason": "stop"},
        ),
    ],
)
def test_empty_truncated_or_tool_answer_is_never_returned(reply):
    frozen, b, profile = inputs()
    with pytest.raises(KnowledgeBranchError, match="answer_generation_error"):
        generate_knowledge_answer(
            frozen, b, client=Answers(reply), profile=profile, audit_hook=lambda _: None
        )


def test_checked_evidence_hash_cannot_change_before_generation():
    frozen, _, profile = inputs()
    client = Answers(
        AIMessage(content="ok[1]", response_metadata={"finish_reason": "stop"})
    )
    with pytest.raises(KnowledgeBranchError, match="knowledge_answer_input_conflict"):
        generate_knowledge_answer(
            frozen,
            bundle((1.0,)),
            client=client,
            profile=profile,
            audit_hook=lambda _: None,
        )
    assert client.calls == []
