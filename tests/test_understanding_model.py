import importlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from resolv.core.config import Settings
from resolv.schemas.understanding import ResolutionResult, UnderstandingError
from tests.understanding_fixtures import FakeJsonModel, make_resolution


def module():
    try:
        return importlib.import_module("resolv.core.understanding_model")
    except ModuleNotFoundError:
        pytest.fail("independent model missing")


def invoke(reply, reserved=2000):
    model = module().JsonStageModel(FakeJsonModel([reply]))
    audits = []
    output, audit = model.invoke(
        "resolution",
        [HumanMessage(content="查订单")],
        ResolutionResult,
        6000,
        reserved,
        audits.append,
    )
    return model, output, audit, audits


@pytest.mark.parametrize(
    "reply",
    [
        AIMessage(content='{"resolved_text":"查订单"}'),
        AIMessage(content='{"resolved_text":"a","resolved_text":"b"}'),
        AIMessage(content="```json\n{}\n```"),
        AIMessage(content=[{"type": "image", "data": "bad"}]),
        AIMessage(content="", tool_calls=[{"id": "tc", "name": "refund", "args": {}}]),
    ],
)
def test_invalid_response_never_becomes_other_or_clarification(reply):
    with pytest.raises(UnderstandingError, match="model_output_invalid"):
        invoke(reply)


def test_text_block_usage_audit_does_not_save_reasoning():
    text = make_resolution().model_dump_json()
    model, output, audit, audits = invoke(
        AIMessage(
            content=[
                {"type": "reasoning", "summary": []},
                {"type": "text", "text": text},
            ],
            usage_metadata={
                "input_tokens": 123,
                "output_tokens": 100,
                "total_tokens": 223,
            },
        )
    )
    assert output.resolved_text == "查订单" and audit.input_tokens == 123
    assert [a.outcome for a in audits] == ["started", "succeeded"]
    assert audit.output_tokens == 100
    assert model.calls_by_stage["resolution"] == 1


def test_failed_call_has_safe_error_and_unknown_usage():
    fake = FakeJsonModel([RuntimeError("secret credential must not be exposed")])
    model = module().JsonStageModel(fake)
    audits = []
    with pytest.raises(UnderstandingError, match="model_unavailable") as caught:
        model.invoke(
            "resolution",
            [HumanMessage(content="hi")],
            ResolutionResult,
            6000,
            2000,
            audits.append,
        )
    assert "secret" not in str(caught.value)
    assert [a.outcome for a in audits] == ["started", "failed"]
    assert audits[-1].input_tokens is None


def test_invalid_public_output_still_preserves_confirmed_usage():
    reply = AIMessage(
        content=[{"type": "image", "data": "invalid"}],
        usage_metadata={"input_tokens": 321, "output_tokens": 20, "total_tokens": 341},
    )
    model = module().JsonStageModel(FakeJsonModel([reply]))
    audits = []
    with pytest.raises(UnderstandingError, match="model_output_invalid"):
        model.invoke(
            "resolution",
            [HumanMessage(content="hi")],
            ResolutionResult,
            6000,
            2000,
            audits.append,
        )
    assert audits[-1].outcome == "failed"
    assert audits[-1].input_tokens == 321
    assert audits[-1].output_tokens == 20


def test_provider_output_over_reserved_budget_is_rejected():
    with pytest.raises(UnderstandingError, match="model_output_budget_exceeded"):
        invoke(
            AIMessage(
                content=make_resolution().model_dump_json(),
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 2001,
                    "total_tokens": 2101,
                },
            )
        )


def test_real_input_usage_above_estimate_blocks_stage_success():
    with pytest.raises(UnderstandingError, match="model_input_underestimated"):
        invoke(
            AIMessage(
                content=make_resolution().model_dump_json(),
                usage_metadata={
                    "input_tokens": 6001,
                    "output_tokens": 100,
                    "total_tokens": 6101,
                },
            )
        )


def test_pinned_sdk_payload_has_actual_output_limit_and_no_tools(monkeypatch):
    import httpx

    mod = module()
    settings = Settings(
        _env_file=None,
        openai_api_key="fake",
        openai_model="test-model",
        openai_base_url="https://example.invalid/v1",
    )
    client = mod.get_understanding_model(settings, 60)
    seen = []

    def transport(request):
        seen.append(json.loads(request.content))
        reply_text = make_resolution().model_dump_json()
        response = {
            "id": "resp_fake",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "test-model",
            "output": [
                {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": reply_text, "annotations": []}
                    ],
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
        if seen[-1].get("stream"):
            events = [
                {
                    "type": "response.created",
                    "response": response,
                    "sequence_number": 0,
                },
                {
                    "type": "response.output_text.delta",
                    "delta": reply_text,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_fake",
                    "sequence_number": 1,
                },
                {
                    "type": "response.output_text.done",
                    "text": reply_text,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_fake",
                    "sequence_number": 2,
                },
                {
                    "type": "response.completed",
                    "response": response,
                    "sequence_number": 3,
                },
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join(
                    "data: " + json.dumps(event) + "\n\n" for event in events
                ),
            )
        return httpx.Response(200, json=response)

    http = httpx.Client(transport=httpx.MockTransport(transport))
    client.root_client._client = http
    wrapped = mod.JsonStageModel(client)
    wrapped.invoke(
        "resolution",
        [HumanMessage(content="查订单")],
        ResolutionResult,
        6000,
        777,
        lambda _: None,
        reasoning_effort="low",
    )
    assert seen[0].get("stream") is True
    assert seen[0]["max_output_tokens"] == 777
    assert seen[0]["reasoning"] == {"effort": "low"}
    assert seen[0]["truncation"] == "disabled" and "tools" not in seen[0]
    assert "previous_response_id" not in seen[0] and client.max_retries == 0
    http.close()


def test_qwen_compatible_sdk_uses_nonstream_chat_with_real_output_limit():
    import httpx

    mod = module()
    settings = Settings(
        _env_file=None,
        openai_api_key="fake",
        openai_model="qwen3.8-flash",
        openai_base_url="https://llm-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    client = mod.get_understanding_model(settings, 60)
    seen = []

    def transport(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": make_resolution().model_dump_json(),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(transport))
    client.root_client._client = http
    output, audit = mod.JsonStageModel(client).invoke(
        "resolution",
        [HumanMessage(content="查订单，JSON")],
        ResolutionResult,
        6000,
        777,
        lambda _: None,
        enable_thinking=True,
        thinking_budget=512,
        top_p=0.1,
        output_format="json_schema",
        output_schema=ResolutionResult.model_json_schema(),
    )
    assert output.resolved_text == "查订单" and audit.input_tokens == 100
    path, payload = seen[0]
    assert path.endswith("/chat/completions") and payload["stream"] is False
    assert (
        payload["max_completion_tokens"] == 777 and payload["enable_thinking"] is True
    )
    assert payload["thinking_budget"] == 512
    assert payload["top_p"] == 0.1
    assert (
        "tools" not in payload
        and "previous_response_id" not in payload
        and "truncation" not in payload
    )
    assert "reasoning_effort" not in payload
    assert payload["response_format"]["type"] == "json_schema"
    schema_format = payload["response_format"]["json_schema"]
    assert schema_format["strict"] is True
    assert set(schema_format["schema"]["required"]) == set(ResolutionResult.model_fields)
    assert client.max_retries == 0
    http.close()
