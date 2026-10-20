"""No-network provider boundary tests; fake SDKs expose exact request controls."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from resolv.core.knowledge_branch_model import (
    KnowledgeModelClient,
    make_knowledge_model_profile,
)
from resolv.core.knowledge_branch_prompts import KnowledgeQueryOutput
from resolv.schemas.knowledge_branch import KnowledgeBranchError, SufficiencyResult
from tests.knowledge_branch_fixtures import model_profile


class Counter:
    def count(self, messages, schemas):
        assert not schemas
        return 100

    def count_text(self, text):
        return len(text)


class SDK:
    def __init__(self, reply, audits):
        self.reply, self.audits = reply, audits
        self.kwargs, self.bound, self.tool_bindings = [], [], []
        self.calls = 0

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        return self

    def bind(self, **kwargs):
        self.bound.append(kwargs)
        return self

    def bind_tools(self, tools):
        self.tool_bindings.append(tools)
        pytest.fail("Knowledge transport must never bind tools")

    def invoke(self, messages):
        assert self.audits[-1].outcome == "started"
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def stream(self, messages):
        reply = self.invoke(messages)
        yield AIMessageChunk(
            content=reply.content,
            tool_calls=reply.tool_calls,
            usage_metadata=reply.usage_metadata,
            response_metadata=reply.response_metadata,
        )


def response(content='{"standard":"退货运费？","expanded":["邮费"]}', **changes):
    return AIMessage(
        **(
            {
                "content": content,
                "usage_metadata": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
                "response_metadata": {"finish_reason": "stop"},
            }
            | changes
        )
    )


def harness(reply, purpose="knowledge_query"):
    audits = []
    sdk = SDK(reply, audits)
    settings = SimpleNamespace(
        openai_model="qwen3.8-plus",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key="offline",
    )
    p = make_knowledge_model_profile(
        settings, purpose=purpose, reference_profile=model_profile(purpose)
    )
    client = KnowledgeModelClient(
        settings, client_factory=sdk, counter_factory=lambda _: Counter()
    )
    messages = [
        SystemMessage(content=p.system_prompt),
        HumanMessage(content="退货运费？"),
    ]
    return client, sdk, audits, p, messages


def test_json_has_no_tools_and_audits_started_before_call():
    client, sdk, audits, p, messages = harness(response())
    parsed = client.invoke_json(
        "knowledge_query",
        messages,
        KnowledgeQueryOutput,
        profile=p,
        audit_hook=audits.append,
    )
    assert parsed.expanded == ("邮费",)
    assert sdk.tool_bindings == [] and sdk.calls == 1
    assert sdk.kwargs[0]["max_retries"] == 0
    assert [a.outcome for a in audits] == ["started", "succeeded"]
    assert audits[-1].input_tokens == 100
    assert "tools" not in sdk.bound[0]
    assert sdk.bound[0]["response_format"]["json_schema"]["name"] == "knowledge_query"


@pytest.mark.parametrize(
    "content",
    [
        '{"standard":"退货","standard":"会员","expanded":[]}',
        '{"standard":"退货","expanded":[],"extra":1}',
        '{"standard":NaN,"expanded":[]}',
        "",
        '{"standard":"退货"}',
    ],
)
def test_malformed_json_is_a_technical_failure(content):
    client, sdk, audits, p, messages = harness(response(content))
    with pytest.raises(KnowledgeBranchError, match="knowledge_model_output_invalid"):
        client.invoke_json(
            "knowledge_query",
            messages,
            KnowledgeQueryOutput,
            profile=p,
            audit_hook=audits.append,
        )
    assert [a.outcome for a in audits] == ["started", "failed"]
    assert sdk.calls == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"usage_metadata": None},
        {"response_metadata": {"finish_reason": "length"}},
        {"response_metadata": {}},
        {"tool_calls": [{"id": "t1", "name": "query_order", "args": {}}]},
        {
            "usage_metadata": {
                "input_tokens": 100000,
                "output_tokens": 1,
                "total_tokens": 100001,
            }
        },
    ],
)
def test_invalid_completion_usage_or_tool_call_is_rejected(changes):
    client, _, audits, p, messages = harness(response(**changes))
    with pytest.raises(KnowledgeBranchError):
        client.invoke_json(
            "knowledge_query",
            messages,
            KnowledgeQueryOutput,
            profile=p,
            audit_hook=audits.append,
        )
    assert audits[-1].outcome == "failed"


def test_check_uses_its_own_schema_and_purpose():
    client, sdk, audits, p, messages = harness(
        response(
            '{"useful":true,"supporting_window_ids":["'
            + "a" * 64
            + '"],"missing_information":[],"reason":"条款明确"}'
        ),
        purpose="knowledge_check",
    )
    parsed = client.invoke_json(
        "knowledge_check",
        messages,
        SufficiencyResult,
        profile=p,
        audit_hook=audits.append,
    )
    assert parsed.useful and audits[-1].purpose == "knowledge_check"
    assert sdk.tool_bindings == []
    assert sdk.bound[0]["response_format"]["json_schema"]["name"] == "knowledge_check"


def test_streamed_answer_is_aggregated_sanitized_and_tool_free():
    client, sdk, audits, p, messages = harness(
        response("质量问题由商家承担[1]。"), purpose="knowledge_answer"
    )
    reply = client.invoke_answer(messages, profile=p, audit_hook=audits.append)
    assert reply.content == "质量问题由商家承担[1]。" and not reply.tool_calls
    assert sdk.tool_bindings == []
    assert "response_format" not in sdk.bound[0]
    assert audits[-1].outcome == "succeeded"


def test_transport_timeout_is_unknown_with_safe_error_code():
    client, sdk, audits, p, messages = harness(TimeoutError("private provider detail"))
    with pytest.raises(KnowledgeBranchError, match="knowledge_model_unavailable"):
        client.invoke_json(
            "knowledge_query",
            messages,
            KnowledgeQueryOutput,
            profile=p,
            audit_hook=audits.append,
        )
    assert audits[-1].outcome == "unknown" and sdk.calls == 1
    assert "private" not in audits[-1].model_dump_json()


def test_invalid_json_preserves_received_usage_in_failure_audit():
    client, _, audits, p, messages = harness(response("invalid json"))
    with pytest.raises(KnowledgeBranchError):
        client.invoke_json(
            "knowledge_query",
            messages,
            KnowledgeQueryOutput,
            profile=p,
            audit_hook=audits.append,
        )
    assert (audits[-1].input_tokens, audits[-1].output_tokens) == (100, 20)


@pytest.mark.parametrize("when", ["started", "succeeded"])
def test_audit_storage_or_execution_failure_is_never_hidden(when):
    client, sdk, audits, p, messages = harness(response())

    def hook(a):
        audits.append(a)
        if a.outcome == when:
            raise KnowledgeBranchError("execution_lost")

    with pytest.raises(KnowledgeBranchError, match="execution_lost"):
        client.invoke_json(
            "knowledge_query",
            messages,
            KnowledgeQueryOutput,
            profile=p,
            audit_hook=hook,
        )
    assert sdk.calls == (0 if when == "started" else 1)


def test_wrong_purpose_or_profile_schema_cannot_be_sent():
    client, sdk, audits, p, messages = harness(response())
    with pytest.raises(KnowledgeBranchError, match="knowledge_model_profile_mismatch"):
        client.invoke_json(
            "knowledge_check",
            messages,
            SufficiencyResult,
            profile=p,
            audit_hook=audits.append,
        )
    assert sdk.calls == 0


def test_client_close_releases_both_transports_and_refuses_reuse():
    client, sdk, audits, p, messages = harness(response())
    client.invoke_json(
        "knowledge_query",
        messages,
        KnowledgeQueryOutput,
        profile=p,
        audit_hook=audits.append,
    )
    closed = []

    async def close_async():
        closed.append("async")

    sdk.root_client = SimpleNamespace(close=lambda: closed.append("sync"))
    sdk.root_async_client = SimpleNamespace(close=close_async)
    client.close()
    assert closed == ["sync", "async"]
    with pytest.raises(KnowledgeBranchError, match="knowledge_model_client_closed"):
        client.invoke_json(
            "knowledge_query",
            messages,
            KnowledgeQueryOutput,
            profile=p,
            audit_hook=audits.append,
        )
    assert sdk.calls == 1
