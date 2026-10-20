"""Network replaced at SDK boundary; native request assembly and parsing are real."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_flow import RefundModelProfile
from resolv.services.hybrid_fingerprints import canonical_hash


class Counter:
    def count(self, messages, schemas):
        return 100

    def count_text(self, text):
        return max(1, len(text) // 4)


class SDK:
    def __init__(self, chunks, audits):
        self.chunks, self.audits = chunks, audits
        self.kwargs, self.bound, self.tools = [], [], []
        self.calls = 0

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        return self

    def bind(self, **kwargs):
        self.bound.append(kwargs)
        return self

    def bind_tools(self, tools, **kwargs):
        self.tools.append((tools, kwargs))
        return self

    def stream(self, messages):
        assert self.audits[-1].outcome == "started"
        self.calls += 1
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def invoke(self, messages):
        assert self.audits[-1].outcome == "started"
        self.calls += 1
        return self.chunks[0]


def profile(purpose="refund_decision"):
    from resolv.core.understanding_model import compact_schema
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund_flow import RefundQueryOutput
    from resolv.services.refund_tools import refund_tool_schemas

    return RefundModelProfile(
        purpose=purpose,
        model_name="fixture",
        model_api_mode="chat_completions",
        api_address_fingerprint=canonical_hash("https://offline.invalid/v1"),
        system_prompt="退款只读测试",
        prompt_version="test",
        tool_schema_fingerprint=canonical_hash(
            refund_tool_schemas() if purpose == "refund_decision" else []
        ),
        output_schema=FrozenJSON.from_value(
            compact_schema(RefundQueryOutput.model_json_schema())
        )
        if purpose == "refund_query"
        else None,
    )


def tool_chunks(
    *, args="{}", finish="tool_calls", usage=True, name="query_refund_order"
):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": name, "args": args, "id": "tc-1", "index": 0}],
            response_metadata={"finish_reason": finish},
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            }
            if usage
            else None,
        )
    ]


def harness(chunks, purpose="refund_decision"):
    from resolv.core.refund_model import RefundModelClient

    audits = []
    sdk = SDK(chunks, audits)
    p = profile(purpose)
    settings = SimpleNamespace(
        openai_model=p.model_name,
        openai_base_url="https://offline.invalid/v1",
        openai_api_key="offline",
    )
    client = RefundModelClient(
        settings, client_factory=sdk, counter_factory=lambda _: Counter()
    )
    messages = [
        SystemMessage(content=p.system_prompt),
        HumanMessage(content="订单A001退款"),
    ]
    return client, sdk, audits, p, messages


def decide(h, **kwargs):
    client, _, audits, p, messages = h
    return client.decide(
        messages,
        profile=p,
        model_step=1,
        attempt=1,
        require_tool=True,
        audit_hook=audits.append,
        **kwargs,
    )


def test_refund_stream_usage_and_tool_validation():
    chunks = tool_chunks()
    h = harness(chunks)
    response = decide(h)
    _client, sdk, audits, _p, _messages = h
    assert (
        response.tool_calls[0].name == "query_refund_order"
        and response.tool_calls[0].args.value == {}
    )
    assert sdk.kwargs[0]["max_retries"] == 0 and sdk.kwargs[0]["stream_usage"] is True
    assert len(sdk.tools[0][0]) == 4 and not any(
        "response_format" in params for params in sdk.bound
    )
    assert [a.outcome for a in audits] == [
        "started",
        "succeeded",
    ] and response.audit == audits[-1]
    assert audits[-1].input_tokens == 100


@pytest.mark.parametrize(
    "chunks",
    [
        tool_chunks(args='{"x":1,"x":2}'),
        tool_chunks(args="[]"),
        tool_chunks(args="{"),
        tool_chunks(name="apply_refund"),
        tool_chunks(args='{"order_id":"B002"}'),
        tool_chunks(finish="length"),
        tool_chunks(usage=False),
        [AIMessageChunk(content="", response_metadata={"finish_reason": "stop"})],
        [RuntimeError("provider secret")],
    ],
)
def test_invalid_stream_or_usage_is_technical_failure(chunks):
    h = harness(chunks)
    with pytest.raises(RefundFlowError):
        decide(h)
    assert h[2][-1].outcome in {"failed", "unknown"}
    assert "secret" not in h[2][-1].model_dump_json()
    if not getattr(chunks[0], "usage_metadata", None):
        assert h[2][-1].input_tokens is None


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("input_tokens", 20000, "refund_model_input_underestimated"),
        ("output_tokens", 2001, "refund_model_output_budget_exceeded"),
    ],
)
def test_actual_usage_budget_cannot_be_ignored(field, value, code):
    chunks = tool_chunks()
    usage = chunks[0].usage_metadata
    usage[field] = value
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    h = harness(chunks)
    with pytest.raises(RefundFlowError, match=code):
        decide(h)


def test_prepare_has_strict_schema_and_no_tools():
    reply = AIMessage(
        content='{"standard":"整单退款条件","expanded":["退货"]}',
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    h = harness([reply], "refund_query")
    client, sdk, audits, p, messages = h
    output = client.prepare(messages, profile=p, audit_hook=audits.append)
    assert output.expanded == ("退货",) and sdk.tools == []
    assert sdk.bound[0]["response_format"]["json_schema"]["strict"] is True


def test_complete_stream_is_required_before_any_response():
    h = harness([*tool_chunks(), RuntimeError("lost after tool fragment")])
    with pytest.raises(RefundFlowError):
        decide(h)
    assert len(h[2]) == 2 and h[2][-1].outcome == "unknown"


def test_audit_storage_error_is_not_model_fallback():
    client, _, _, p, messages = harness(tool_chunks())

    def broken(_):
        raise OSError("storage")

    with pytest.raises(OSError):
        client.decide(
            messages,
            profile=p,
            model_step=1,
            attempt=1,
            require_tool=True,
            audit_hook=broken,
        )


def test_final_public_json_comes_from_same_decision_response():
    from tests.test_refund_decision import decision

    text = decision().model_dump_json()
    h = harness(
        [
            AIMessageChunk(
                content=text,
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 100,
                    "total_tokens": 200,
                },
            )
        ]
    )
    client, sdk, audits, p, messages = h
    response = client.decide(
        messages,
        profile=p,
        model_step=2,
        attempt=1,
        require_tool=False,
        audit_hook=audits.append,
    )
    assert response.decision.reason == decision().reason and sdk.calls == 1


def test_default_refund_profile_freezes_native_tools_and_output_contracts():
    from resolv.core.refund_config import default_refund_profile
    from resolv.core.refund_model import refund_request_parameters
    from resolv.core.refund_native import (
        RETURN_DECISION,
        decision_tools,
        native_decision_schema,
    )
    from resolv.schemas.refund import AdapterDescriptor
    from resolv.services.refund_tools import TOOLS, refund_tool_schemas

    p = default_refund_profile(
        understanding_config=profile("refund_query"),
        business_config=profile(),
        summary_profile={},
        adapter_descriptor=AdapterDescriptor(
            namespace="fixture", implementation_version="1", authorization_version="1"
        ),
    )
    assert p.decision_model.tool_schema_fingerprint == canonical_hash(
        decision_tools(p.decision_model)
    )
    assert p.config_version == 7
    assert p.decision_model.output_schema.value == native_decision_schema()
    assert len(TOOLS) == 4 and RETURN_DECISION not in TOOLS
    assert decision_tools(p.decision_model, require_tool=True) == refund_tool_schemas()
    assert "policy_citations" in p.decision_model.system_prompt
    assert (
        refund_request_parameters(p.query_model, prepare=True)["response_format"][
            "type"
        ]
        == "json_schema"
    )


def test_saved_response_hash_and_audit_input_are_closed():
    from pydantic import ValidationError

    response = decide(harness(tool_chunks()))
    for change in [{"output_hash": "0" * 64}, {"input_hash": "0" * 64}]:
        with pytest.raises(ValidationError):
            response.model_copy(update=change)
