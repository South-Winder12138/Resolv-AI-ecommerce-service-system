"""A terminal native DTO is a return channel, never a business-tool capability."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk, messages_from_dict

from resolv.core.refund_config import default_refund_profile
from resolv.core.refund_model import refund_input_estimate, refund_request_parameters
from resolv.core.refund_response import ResponseDecodeError, decode_refund_response
from resolv.schemas.refund_flow import RefundRunRecord
from resolv.services.context import TokenCounter
from tests.test_refund_decision import decision
from tests.test_refund_model import tool_chunks
from tests.test_refund_round_policy import prefix

RETURN = "return_refund_decision"


def archived():
    p = Path(__file__).resolve().parents[1] / (
        "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-context-identity-F1.json"
    )
    return RefundRunRecord.model_validate_json(
        json.dumps(json.loads(p.read_text())["cases"][0]["refund_record"])
    )


def native_profile():
    p = archived().profile
    return default_refund_profile(
        understanding_config=p.query_model,
        business_config=p.decision_model,
        summary_profile=p.summary_profile.value,
        adapter_descriptor=p.adapter,
    ).model_copy(update={"selection": p.selection})


def decode(chunks, *, required=False, final=False):
    p = native_profile().decision_model
    return decode_refund_response(
        chunks[0],
        p,
        prepare=False,
        require_tool=required,
        final_round=final,
        counter=TokenCounter(p.tokenizer_profile),
        estimate=60000,
    )


def test_fresh_native_profile_separates_policy_rules_from_server_prerequisites():
    p = native_profile()
    assert p.config_version == 7
    assert p.decision_model.prompt_version == "refund-decision-v6-native-return"
    assert "rules只放实际政策规则" in p.decision_model.system_prompt
    assert "final_round=true时不得调用工具" not in p.decision_model.system_prompt
    assert (
        "系统前提"
        in p.decision_model.output_schema.value["properties"]["critical_fields"][
            "description"
        ]
    )
    assert p.max_agent_steps == 6 and p.decision_model.input_limit == 60000
    assert p.decision_model.max_retries == 0


@pytest.mark.parametrize(
    "observed,expected",
    [(0, "query_refund_order"), (1, "query_refund_status"), (2, None)],
)
def test_mandatory_native_read_is_named_and_final_exposes_only_return(
    observed, expected
):
    from resolv.core.refund_native import decision_tools

    p, r = native_profile(), archived()
    observations = r.observations[:observed]
    name = p.required_read_tool(observations)
    assert name == expected
    params = refund_request_parameters(
        p.decision_model, require_tool=bool(name), required_tool=name
    )
    assert "response_format" not in params
    if name:
        assert params["tool_choice"] == {"type": "function", "function": {"name": name}}
        assert RETURN not in {
            t["function"]["name"]
            for t in decision_tools(p.decision_model, require_tool=True)
        }
    final = refund_request_parameters(p.decision_model, final_round=True)
    assert final["tool_choice"] == {"type": "function", "function": {"name": RETURN}}
    assert [
        t["function"]["name"]
        for t in decision_tools(p.decision_model, final_round=True)
    ] == [RETURN]


def test_native_terminal_arguments_decode_without_executable_tool_calls():
    value = decision()
    for final in (False, True):
        (calls, returned), _, finish = decode(
            tool_chunks(name=RETURN, args=value.model_dump_json()), final=final
        )
        assert calls == () and returned == value and finish == "tool_calls"


@pytest.mark.parametrize(
    "kind",
    [
        "plain",
        "early",
        "business_final",
        "mixed",
        "duplicate",
        "nonfinite",
        "empty_citation",
    ],
)
def test_native_return_refuses_invalid_or_ambiguous_results(kind):
    answer = decision().model_dump(mode="json")
    chunk = tool_chunks(name=RETURN, args=json.dumps(answer))[0]
    required = final = False
    if kind == "plain":
        chunk = AIMessageChunk(
            content=json.dumps(answer),
            response_metadata={"finish_reason": "stop"},
            usage_metadata=chunk.usage_metadata,
        )
    elif kind == "early":
        required = True
    elif kind == "business_final":
        chunk, final = tool_chunks()[0], True
    elif kind in {"mixed", "duplicate"}:
        second = {
            "name": "query_refund_payment" if kind == "mixed" else RETURN,
            "args": "{}" if kind == "mixed" else json.dumps(answer),
            "id": "tc-2",
            "index": 1,
        }
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[*chunk.tool_call_chunks, second],
            response_metadata=chunk.response_metadata,
            usage_metadata=chunk.usage_metadata,
        )
    elif kind == "nonfinite":
        chunk = tool_chunks(
            name=RETURN, args=json.dumps(answer)[:-1] + ', "extra": NaN}'
        )[0]
    elif kind == "empty_citation":
        answer["rules"][0]["policy_citations"] = []
        chunk = tool_chunks(name=RETURN, args=json.dumps(answer))[0]
    with pytest.raises(ResponseDecodeError) as exc:
        decode([chunk], required=required, final=final)
    if kind == "empty_citation":
        assert exc.value.stage == "contract"
        assert exc.value.cause.errors()[0]["loc"] == ("rules", 0, "policy_citations")


def test_native_context_memory_and_request_count_identical_phase_parameters():
    from types import SimpleNamespace

    from resolv.services.refund_flow import _MemoryBuilder
    from tests.test_refund_context_identity import saved

    r, p = saved(), native_profile()
    for step in (1, 2, 4, 6):
        c = prefix(r, p, step)
        responses = r.model_requests[: step - 1]
        seen = {call.id for response in responses for call in response.tool_calls}
        observations = tuple(o for o in r.observations if o.tool_call_id in seen)
        kwargs = {
            "require_tool": p.context_requires_tool(step, observations),
            "final_round": p.final_decision_round(step),
            "required_tool": None
            if p.final_decision_round(step)
            else p.required_read_tool(observations),
        }
        assert c.input_estimate == refund_input_estimate(
            messages_from_dict(c.messages.value),
            p.decision_model,
            TokenCounter(p.decision_model.tokenizer_profile),
            **kwargs,
        )
        observed = r.model_copy(
            update={
                "profile": p,
                "model_requests": responses,
                "observations": observations,
            }
        )
        built = _MemoryBuilder(
            observed, TokenCounter(p.decision_model.tokenizer_profile)
        ).build(SimpleNamespace(summary=None, history=[]))
        assert built.input_tokens == c.input_estimate


def test_archived_v6_inputs_and_text_protocol_are_frozen():
    r = archived()
    for step in range(1, 5):
        c = prefix(r, r.profile, step)
        a = next(
            a
            for a in reversed(r.audits)
            if a.model_step == step and a.purpose == "refund_decision"
        )
        assert (c.input_hash, c.input_estimate) == (a.input_hash, a.input_estimate)
    assert r.profile.config_version == 6
    assert (
        refund_request_parameters(r.profile.decision_model, final_round=True)[
            "tool_choice"
        ]
        == "none"
    )


def test_native_client_rejects_provider_disobeying_the_named_mandatory_read():
    from langchain_core.messages import SystemMessage

    from resolv.schemas.refund import RefundFlowError
    from resolv.services.hybrid_fingerprints import canonical_hash
    from tests.test_refund_model import harness

    client, sdk, audits, _, _ = harness(tool_chunks(name="query_refund_order"))
    p = native_profile().decision_model.model_copy(
        update={
            "model_name": "fixture",
            "api_address_fingerprint": canonical_hash("https://offline.invalid/v1"),
        }
    )
    with pytest.raises(RefundFlowError, match="refund_model_output_invalid"):
        client.decide(
            [SystemMessage(content=p.system_prompt)],
            profile=p,
            model_step=2,
            attempt=1,
            require_tool=True,
            required_tool="query_refund_status",
            audit_hook=audits.append,
        )
    assert sdk.calls == 1 and audits[-1].outcome == "failed"


@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
def test_installed_sdk_native_return_has_no_format_or_business_permission_leak(
    api_mode,
):
    from types import SimpleNamespace

    import httpx
    from langchain_core.messages import SystemMessage
    from langchain_openai import ChatOpenAI

    from resolv.core.refund_model import RefundModelClient
    from resolv.services.hybrid_fingerprints import canonical_hash

    p = native_profile().decision_model.model_copy(
        update={
            "model_api_mode": api_mode,
            "api_address_fingerprint": canonical_hash("https://offline.invalid/v1"),
        }
    )
    payloads = []

    def handle(request):
        body = json.loads(request.content)
        payloads.append(body)
        name = "query_refund_order" if len(payloads) in {1, 3} else RETURN
        args = "{}" if name != RETURN else decision().model_dump_json()
        if api_mode == "chat_completions":
            base = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": p.model_name,
            }
            chunks = [
                base
                | {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_native",
                                        "type": "function",
                                        "function": {"name": name, "arguments": args},
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                base
                | {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ]
                },
                base
                | {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                },
            ]
            stream = (
                "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n"
            )
        else:
            item = {
                "id": "fc_native",
                "type": "function_call",
                "call_id": "call_native",
                "name": name,
                "arguments": args,
                "status": "completed",
            }
            response = {
                "id": "resp_mock",
                "object": "response",
                "created_at": 0,
                "model": p.model_name,
                "status": "completed",
                "output": [item],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "total_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
            events = [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": item | {"arguments": "", "status": "in_progress"},
                    "sequence_number": 0,
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "item_id": item["id"],
                    "delta": args,
                    "sequence_number": 1,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": item,
                    "sequence_number": 2,
                },
                {
                    "type": "response.completed",
                    "response": response,
                    "sequence_number": 3,
                },
            ]
            stream = "".join(
                "event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n"
                for e in events
            )
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=stream
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        async_http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        c = RefundModelClient(
            SimpleNamespace(
                openai_model=p.model_name,
                openai_base_url="https://offline.invalid/v1",
                openai_api_key="offline",
            ),
            client_factory=lambda **kw: ChatOpenAI(
                **kw, http_client=http, http_async_client=async_http
            ),
        )
        try:
            results = []
            for required, final in ((True, False), (False, True), (True, False)):
                results.append(
                    c.decide(
                        [SystemMessage(content=p.system_prompt)],
                        profile=p,
                        model_step=6 if final else 1,
                        attempt=1,
                        require_tool=required,
                        final_round=final,
                        required_tool="query_refund_order" if required else None,
                        audit_hook=lambda _: None,
                    )
                )
        finally:
            c.close()
    assert results[1].decision == decision() and not results[1].tool_calls
    assert len(payloads) == 3
    for i, body in enumerate(payloads):
        assert "response_format" not in body and not body.get("text", {}).get("format")
        tools = [
            t.get("name", t.get("function", {}).get("name")) for t in body["tools"]
        ]
        choice = body["tool_choice"]
        assert choice.get("name", choice.get("function", {}).get("name")) == (
            RETURN if i == 1 else "query_refund_order"
        )
        assert (
            (tools == [RETURN]) if i == 1 else (RETURN not in tools and len(tools) == 4)
        )
