"""Classify and replay failed public responses without retaining private SDK data."""

import json
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessageChunk

from resolv.schemas.refund import RefundFlowError
from tests.test_refund_model import harness, tool_chunks


def text_chunk(text, **kwargs):
    return AIMessageChunk(
        content=text,
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        **kwargs,
    )


def call(h):
    client, _, audits, profile, messages = h
    return client.decide(
        messages,
        profile=profile,
        model_step=5,
        attempt=1,
        require_tool=False,
        audit_hook=audits.append,
    )


@pytest.mark.parametrize(
    "chunks,stage,kind,path",
    [
        ([text_chunk('{"operation":')], "json", "JSONDecodeError", None),
        (
            [text_chunk('{"operation":"apply"}')],
            "contract",
            "ValidationError",
            ["verdict"],
        ),
        (
            [text_chunk('{"operation":"apply","operation":"status"}')],
            "json",
            "ValueError",
            None,
        ),
        (tool_chunks(finish="stop"), "protocol", "RefundFlowError", None),
        (tool_chunks(args='{"order_id":"other"}'), "tools", "RefundFlowError", None),
    ],
)
def test_failure_hook_identifies_stage_and_preserves_replayable_public_payload(
    chunks, stage, kind, path
):
    h = harness(chunks)
    rows = []
    h[0].diagnostic_hook = rows.append
    with pytest.raises(RefundFlowError, match="refund_model_output_invalid"):
        call(h)
    assert len(rows) == 1
    d = rows[0]
    assert d["stage"] == stage and d["exception_type"] == kind
    assert d["invocation_id"] == h[2][-1].invocation_id
    assert d["input_hash"] == h[2][-1].input_hash
    assert d["error_code"] == "refund_model_output_invalid"
    if path:
        assert any(x["loc"] == path and x["type"] == "missing" for x in d["issues"])
    if kind == "JSONDecodeError":
        assert d["json_error"]["position"] == 13
    assert "input" not in json.dumps(d["issues"])
    from resolv.core.refund_diagnostics import replay_failed_response

    replay = replay_failed_response(d, h[3])
    assert replay["stage"] == stage and replay["exception_type"] == kind
    assert replay["response_hash_matches"] is True
    assert replay["provider_calls"] == 0


def test_diagnostic_payload_omits_reasoning_headers_and_provider_credentials():
    h = harness(
        [
            text_chunk(
                [
                    {"type": "reasoning", "reasoning": "PRIVATE_THOUGHT"},
                    {"type": "text", "text": '{"operation":'},
                ],
                additional_kwargs={
                    "reasoning_content": "PRIVATE_THOUGHT",
                    "api_key": "PRIVATE_KEY",
                },
            )
        ]
    )
    rows = []
    h[0].diagnostic_hook = rows.append
    with pytest.raises(RefundFlowError):
        call(h)
    assert rows, "failed public response must be retained by explicit diagnostics hook"
    raw = json.dumps(rows)
    assert "PRIVATE_THOUGHT" not in raw and "PRIVATE_KEY" not in raw
    assert rows[0]["response"]["content"] == [{"type": "text", "text": '{"operation":'}]


def test_diagnostic_sink_failure_is_visible_after_original_failed_audit():
    h = harness([text_chunk('{"operation":')])

    def broken(_):
        raise OSError("private diagnostic path")

    h[0].diagnostic_hook = broken
    with pytest.raises(RefundFlowError, match="refund_model_diagnostic_store_failed"):
        call(h)
    assert h[2][-1].error_code == "refund_model_output_invalid"


def test_default_client_has_no_implicit_diagnostic_storage():
    h = harness([text_chunk('{"operation":')])
    assert hasattr(h[0], "diagnostic_hook") and h[0].diagnostic_hook is None
    with pytest.raises(RefundFlowError, match="refund_model_output_invalid"):
        call(h)


def test_success_does_not_create_a_failure_payload():
    h = harness(tool_chunks())
    rows = []
    h[0].diagnostic_hook = rows.append
    call(h)
    assert rows == []


def test_explicit_thinking_tags_are_not_persisted_in_public_response_snapshot():
    h = harness([text_chunk('<think>PRIVATE_THOUGHT</think>{"operation":')])
    rows = []
    h[0].diagnostic_hook = rows.append
    with pytest.raises(RefundFlowError):
        call(h)
    assert "PRIVATE_THOUGHT" not in json.dumps(rows)
    assert rows[0]["stage"] == "json"


def test_root_semantic_rule_is_identified_without_serializing_validation_input():
    from tests.test_refund_decision import decision

    value = decision().model_dump(mode="json")
    value["missing_information"] = ["PRIVATE_USER_CONTENT"]
    h = harness([text_chunk(json.dumps(value))])
    rows = []
    h[0].diagnostic_hook = rows.append
    with pytest.raises(RefundFlowError):
        call(h)
    assert rows[0]["issues"] == [
        {"loc": [], "type": "value_error", "rule": "unresolved_eligibility_conditions"}
    ]
    assert "PRIVATE_USER_CONTENT" not in json.dumps(rows[0]["issues"])


def test_provider_request_failure_records_safe_status_and_parameter_without_private_message():
    class ProviderError(Exception):
        status_code = 400
        body: ClassVar[dict] = {
            "error": {
                "code": "invalid_json_schema",
                "type": "invalid_request_error",
                "param": "response_format",
                "message": "PRIVATE_PROVIDER_MESSAGE",
            }
        }

    h = harness([ProviderError("PRIVATE_PROVIDER_MESSAGE")])
    rows = []
    h[0].diagnostic_hook = rows.append
    with pytest.raises(RefundFlowError, match="refund_model_unavailable"):
        call(h)
    assert rows, "request-level errors must identify provider schema rejection"
    assert rows[0]["stage"] == "transport" and rows[0]["http_status"] == 400
    assert rows[0]["provider_error"] == {
        "code": "invalid_json_schema",
        "type": "invalid_request_error",
        "param": "response_format",
    }
    assert "PRIVATE_PROVIDER_MESSAGE" not in json.dumps(rows)
    assert rows[0]["replayable"] is False
    assert h[2][-1].outcome == "unknown"
