"""Preparation isolates the selected primary and never hides ownership/storage errors."""

import json

import pytest
from pydantic import ValidationError

from resolv.core.knowledge_branch_prompts import KnowledgeQueryOutput
from resolv.schemas.knowledge_branch import KnowledgeBranchError
from resolv.services.knowledge_branch_query import prepare_knowledge_query
from tests.knowledge_branch_fixtures import branch_profile, primary


class Client:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def invoke_json(self, purpose, messages, output_type, **kwargs):
        self.calls.append((purpose, messages, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        try:
            return output_type.model_validate_json(json.dumps(self.reply))
        except ValidationError as exc:
            raise KnowledgeBranchError("knowledge_model_output_invalid", True) from exc


def prepare(client, p=None, count=len):
    return prepare_knowledge_query(
        p or primary(),
        client=client,
        profile=branch_profile(),
        query_token_count=count,
        audit_hook=lambda _: None,
    )


def test_only_primary_and_bindings_are_sent_and_expansions_are_normalized():
    client = Client(
        {
            "standard": " 退货运费由谁承担？ ",
            "expanded": ["邮费", "", "  ", "邮费", "寄回费用"],
        }
    )
    q = prepare(client)
    assert q.standard == "退货运费由谁承担？"
    assert q.expanded == ("邮费", "寄回费用")
    assert q.bm25_text == q.standard + "\n邮费\n寄回费用"
    sent = json.loads(client.calls[0][1][1].content)
    assert sent["primary_text"] == primary().primary.text
    assert set(sent) == {"primary_text", "bindings", "primary_ref"}
    assert q.status == "prepared"


@pytest.mark.parametrize(
    "reply",
    [
        TimeoutError(),
        KnowledgeBranchError("knowledge_model_output_invalid", True),
        {"standard": " ", "expanded": []},
        {"standard": "退货", "expanded": ["a" * 33]},
        {"standard": "退货", "expanded": [str(i) for i in range(9)]},
    ],
)
def test_model_failures_reuse_primary_without_expansions(reply):
    p = primary("普通档用户退货运费谁承担？")
    q = prepare(Client(reply), p)
    assert q.status == "fallback" and q.fallback_reason
    assert q.standard == q.bm25_text == p.primary.text and q.expanded == ()


def test_exact_expansion_limits_are_accepted():
    q = prepare(
        Client({"standard": "退货", "expanded": [str(i) * 32 for i in range(8)]})
    )
    assert q.status == "prepared" and len(q.expanded) == 8


@pytest.mark.parametrize(
    "error",
    [
        KnowledgeBranchError("execution_lost"),
        KnowledgeBranchError("knowledge_model_profile_mismatch"),
        OSError("storage"),
        ValueError("storage hash validation"),
        RuntimeError("owner"),
    ],
)
def test_execution_and_storage_failures_cannot_become_query_fallback(error):
    with pytest.raises(type(error)):
        prepare(Client(error))


def test_query_limit_is_real_count_and_never_silently_truncated():
    with pytest.raises(KnowledgeBranchError, match="query_too_long"):
        prepare(Client(TimeoutError()), primary("原文"), count=lambda _: 513)
    assert (
        prepare(
            Client({"standard": "完整原文", "expanded": []}), count=lambda _: 512
        ).standard
        == "完整原文"
    )


def test_invalid_primary_is_rejected_before_model_call():
    p = primary()
    p.primary.branch = "business"
    client = Client({"standard": "退货", "expanded": []})
    with pytest.raises(KnowledgeBranchError, match="knowledge_primary_invalid"):
        prepare(client, p)
    assert client.calls == []


def test_query_schema_is_strict():
    with pytest.raises(ValueError):
        KnowledgeQueryOutput.model_validate_json(
            '{"standard":"退货","expanded":[],"other":1}'
        )


@pytest.mark.parametrize("raises", [False, True])
def test_overlong_model_rewrite_reuses_legal_primary(raises):
    from resolv.schemas.hybrid_knowledge import HybridRetrievalError

    calls = []

    def count(text):
        calls.append(text)
        if text == "long rewrite":
            if raises:
                raise HybridRetrievalError("query_too_long")
            return 513
        return 12

    p = primary("原主诉求")
    q = prepare(
        Client({"standard": "long rewrite", "expanded": ["expansion"]}), p, count=count
    )
    assert q.status == "fallback" and q.standard == p.primary.text and q.expanded == ()
    assert calls == ["long rewrite", p.primary.text]


def test_tokenizer_fault_does_not_become_overlong_rewrite_fallback():
    from resolv.schemas.hybrid_knowledge import HybridRetrievalError

    calls = []

    def fail(text):
        calls.append(text)
        raise HybridRetrievalError("reranker_token_count_failed")

    with pytest.raises(HybridRetrievalError, match="reranker_token_count_failed"):
        prepare(Client({"standard": "rewrite", "expanded": []}), count=fail)
    assert calls == ["rewrite"]
