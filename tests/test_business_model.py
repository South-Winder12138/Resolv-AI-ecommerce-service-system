import importlib

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from resolv.repositories.conversation import fingerprint
from resolv.schemas.branch_execution import BusinessRunConfig
from resolv.schemas.conversation import ConversationError
from tests.test_context import Counter


def api():
    try:
        return importlib.import_module("resolv.core.business_model")
    except ModuleNotFoundError:
        pytest.fail("audited business transport missing")


def profile(**overrides):
    return BusinessRunConfig(
        **(
            {
                "model_name": "qwen3.8-flash",
                "api_address_fingerprint": fingerprint("https://example.test/v1"),
                "system_prompt": "客服",
                "tool_schema_fingerprint": fingerprint(api().business_tool_schemas()),
            }
            | overrides
        )
    )


class Stream:
    def __init__(self, chunks):
        self.chunks, self.bound, self.calls = chunks, [], []

    def bind_tools(self, tools):
        self.bound.append([t.name for t in tools])
        return self

    def stream(self, messages):
        self.calls.append(messages)
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def bind(self, **kwargs):
        self.parameters = kwargs
        return self


def final(text="已查询", *, reason="stop", usage=True):
    return AIMessageChunk(
        content=text,
        response_metadata={"finish_reason": reason},
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        if usage
        else None,
        additional_kwargs={"reasoning_content": "private thought secret"},
    )


def invoke(client, audits, **options):
    return api().invoke_business_model(
        client,
        [HumanMessage(content="查订单 MH001")],
        config=profile(**options),
        counter=Counter(),
        model_step=1,
        attempt=1,
        audit_hook=audits.append,
    )


def test_required_native_tool_choice_rejects_text_that_only_describes_a_call():
    client, audits = Stream([final('[调用工具] query_order(order_id="MH001")')]), []
    with pytest.raises(ConversationError, match="model_tool_call_required"):
        api().invoke_business_model(
            client,
            [HumanMessage(content="查订单 MH001")],
            config=profile(),
            counter=Counter(),
            model_step=1,
            attempt=1,
            audit_hook=audits.append,
            require_tool=True,
        )
    assert client.parameters["tool_choice"] == "required"
    assert audits[-1].error_code == "model_tool_call_required"


def test_business_profile_has_native_independent_parameters(monkeypatch):
    from resolv.core.config import Settings

    captured = []
    monkeypatch.setattr(
        api(), "ChatOpenAI", lambda **kwargs: captured.append(kwargs) or object()
    )
    settings = Settings(
        _env_file=None,
        openai_api_key="secret",
        openai_model="qwen3.8-flash",
        openai_base_url="https://example.test/v1",
    )
    api().get_business_model(settings, profile())
    params = captured[0]
    assert params["use_responses_api"] is False and params["stream_usage"] is True
    assert params["max_retries"] == 0 and params["timeout"] == 60
    assert params["extra_body"] == {"enable_thinking": False}
    assert params["max_completion_tokens"] == 2000 and params["top_p"] == 0.1
    assert {"response_format", "previous_response_id", "truncation"}.isdisjoint(params)


def test_complete_tool_fragments_are_joined_with_usage_and_no_thoughts():
    client = Stream(
        [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "query_order", "id": "tc", "index": 0, "args": '{"order_'}
                ],
            ),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "id": None, "index": 0, "args": 'id":"MH001"}'}
                ],
            ),
            final("", reason="tool_calls"),
        ]
    )
    audits = []
    result = invoke(client, audits)
    assert result.message.tool_calls[0]["args"] == {"order_id": "MH001"}
    assert client.bound == [["query_order", "query_logistics", "cancel_order"]]
    assert result.audit.actual_input == 100 and result.audit.actual_output == 10
    assert (
        result.message.additional_kwargs == {}
        and "private thought" not in result.model_dump_json()
    )
    assert audits == [result.audit]


@pytest.mark.parametrize(
    "args",
    ['{"order_id":"MH001","order_id":"MH002"}', '{"order_id":NaN}', '{"order_id":'],
)
def test_invalid_complete_tool_json_never_returns_decision(args):
    client = Stream(
        [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "query_order", "id": "tc", "index": 0, "args": args}
                ],
            ),
            final("", reason="tool_calls"),
        ]
    )
    audits = []
    with pytest.raises(ConversationError, match="model_response_invalid"):
        invoke(client, audits)
    assert (
        audits[-1].error_code == "model_response_invalid"
        and audits[-1].actual_input == 100
    )


@pytest.mark.parametrize(
    "chunks,code",
    [
        ([final(reason="length")], "model_response_incomplete"),
        ([final("")], "model_response_empty"),
        ([final(usage=False)], "model_usage_unknown"),
        ([RuntimeError("API key secret")], "model_unavailable"),
    ],
)
def test_incomplete_usage_and_provider_failures_are_audited_safely(chunks, code):
    audits = []
    with pytest.raises(ConversationError, match=code) as caught:
        invoke(Stream(chunks), audits)
    assert str(caught.value) == code and audits[-1].error_code == code
    assert "secret" not in audits[-1].model_dump_json()


def test_usage_underestimate_or_output_overflow_is_rejected():
    for usage, code in [
        (
            {"input_tokens": 11999, "output_tokens": 10, "total_tokens": 12009},
            "model_input_underestimated",
        ),
        (
            {"input_tokens": 100, "output_tokens": 2001, "total_tokens": 2101},
            "model_output_budget_exceeded",
        ),
    ]:
        audits = []
        chunk = final().model_copy(update={"usage_metadata": usage})
        with pytest.raises(ConversationError, match=code):
            invoke(Stream([chunk]), audits)
        assert audits[-1].error_code == code


def test_input_budget_refuses_before_provider_call():
    client = Stream([final()])
    audits = []
    with pytest.raises(ConversationError, match="context_budget_exceeded"):
        invoke(client, audits, app_max_input_tokens=100)
    assert client.calls == []


def test_each_call_reapplies_frozen_sampling_and_output_budget():
    client = Stream([final()])
    audits = []
    invoke(client, audits, top_p=0.2, reserved_output_tokens=1500)
    assert client.parameters["top_p"] == 0.2
    assert client.parameters["max_completion_tokens"] == 1500
    assert client.parameters["extra_body"] == {"enable_thinking": False}


def test_each_public_invocation_records_elapsed_time():
    audits = []
    result = invoke(Stream([final()]), audits)
    assert result.audit.elapsed_seconds >= 0
