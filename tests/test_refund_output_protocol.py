"""Fresh structured decisions reserve schema cost; old runs keep frozen requests."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import messages_from_dict

from resolv.core.refund_model import refund_input_estimate, refund_request_parameters
from resolv.schemas.refund_flow import RefundRunRecord
from resolv.services.context import TokenCounter
from tests.test_refund_round_policy import archived, fresh, prefix


@pytest.mark.parametrize(
    "required,final", [(True, False), (False, False), (False, True)]
)
def test_new_model_contract_constrains_content_only_after_mandatory_reads(
    required, final
):
    profile = fresh(archived().profile)
    p = refund_request_parameters(
        profile.decision_model, require_tool=required, final_round=final
    )
    if required:
        assert "response_format" not in p and p["tool_choice"] == "required"
    else:
        assert p.get("response_format", {}).get("type") == "json_schema"
        schema = p["response_format"]["json_schema"]
        assert schema["strict"] is True and schema["name"] == "refund_decision"
        assert schema["schema"]["additionalProperties"] is False
        assert "order_binding_hash" in schema["schema"]["required"]
    if final:
        assert p["tool_choice"] == "none"


def test_fresh_profile_version_and_context_account_for_schema_before_sixth_round():
    saved = archived()
    profile = fresh(saved.profile)
    assert profile.config_version == 6
    assert (
        profile.decision_model.prompt_version == "refund-decision-v5-context-identity"
    )
    context = prefix(saved, profile, 3)
    expected = refund_input_estimate(
        messages_from_dict(context.messages.value),
        profile.decision_model,
        TokenCounter(profile.decision_model.tokenizer_profile),
        require_tool=False,
    )
    assert context.input_estimate == expected


def test_fresh_wire_schema_requires_every_field_without_weakening_business_rules():
    from pydantic import ValidationError

    from resolv.schemas.refund_flow import RefundDecisionOutput
    from tests.test_refund_decision import decision

    profile = fresh(archived().profile)
    schema = refund_request_parameters(profile.decision_model)["response_format"][
        "json_schema"
    ]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    assert all("default" not in v for v in schema["properties"].values())
    assert profile.decision_model.output_schema.value["required"] == schema["required"]
    assert "所有字段必须显式输出" in profile.decision_model.system_prompt
    data = decision().model_dump(mode="json")
    RefundDecisionOutput.model_validate_json(json.dumps(data))
    for field in (
        "observation_refs",
        "policy_citations",
        "rules",
        "critical_fields",
        "recheck_conditions",
    ):
        incomplete = {k: v for k, v in data.items() if k != field}
        with pytest.raises(ValidationError) as exc:
            RefundDecisionOutput.model_validate_json(json.dumps(incomplete))
        assert any(
            e["loc"] == (field,) and e["type"] == "missing" for e in exc.value.errors()
        )
    data.update(observation_refs=[], policy_citations=[], rules=[])
    with pytest.raises(ValidationError, match="requires sourced policy and facts"):
        RefundDecisionOutput.model_validate_json(json.dumps(data))


def test_actual_v4_failure_and_six_inputs_replay_without_any_provider_call():
    from langchain_core.messages import AIMessageChunk

    from resolv.core.refund_response import ResponseDecodeError, decode_refund_response

    path = (
        Path(__file__).resolve().parents[1]
        / "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-output-repair-F1.json"
    )
    record = RefundRunRecord.model_validate_json(
        json.dumps(json.loads(path.read_text())["cases"][0]["refund_record"])
    )
    frozen = record.model_dump_json()
    for step in range(1, 7):
        context = prefix(record, record.profile, step)
        audit = next(
            a
            for a in reversed(record.audits)
            if a.model_step == step and a.purpose == "refund_decision"
        )
        assert (context.input_hash, context.input_estimate) == (
            audit.input_hash,
            audit.input_estimate,
        )
    assert record.model_dump_json() == frozen and record.profile.config_version == 4
    # Minimal equivalent of the captured six-field answer, without copying private payloads.
    answer = {
        "operation": "apply",
        "verdict": "eligible",
        "order_binding_hash": "a" * 64,
        "evidence_hash": "b" * 64,
        "reason": "申请审核",
        "next_action": "show_offer",
    }
    reply = AIMessageChunk(
        content=json.dumps(answer),
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 1, "output_tokens": 100, "total_tokens": 101},
    )
    for model in (record.profile.decision_model, fresh(record.profile).decision_model):
        with pytest.raises(ResponseDecodeError) as exc:
            decode_refund_response(
                reply,
                model,
                prepare=False,
                require_tool=False,
                final_round=True,
                counter=TokenCounter(model.tokenizer_profile),
                estimate=30000,
            )
        assert exc.value.stage == "contract"
        errors = exc.value.cause.errors()
        if model.prompt_version in {
            "refund-decision-v4-complete-json-schema",
            "refund-decision-v5-context-identity",
        }:
            assert any(
                e["loc"] == ("rules",) and e["type"] == "missing" for e in errors
            )
        else:
            assert "requires sourced policy and facts" in errors[0]["msg"]


def test_v3_five_actual_request_hashes_and_parameter_modes_are_preserved():
    p = (
        Path(__file__).resolve().parents[1]
        / "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-round-control-F1.json"
    )
    record = RefundRunRecord.model_validate_json(
        json.dumps(json.loads(p.read_text())["cases"][0]["refund_record"])
    )
    raw = record.profile.model_dump_json()
    for step in range(1, 6):
        context = prefix(record, record.profile, step)
        audit = next(
            a
            for a in reversed(record.audits)
            if a.model_step == step and a.purpose == "refund_decision"
        )
        assert context.input_hash == audit.input_hash
        assert "response_format" not in refund_request_parameters(
            record.profile.decision_model, require_tool=step <= 2
        )
    assert record.profile.model_dump_json() == raw


@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
def test_installed_sdk_sends_structured_format_and_decodes_native_final_stream(
    api_mode,
):
    from types import SimpleNamespace

    import httpx
    from langchain_core.messages import SystemMessage
    from langchain_openai import ChatOpenAI

    from resolv.core.refund_model import RefundModelClient
    from resolv.services.hybrid_fingerprints import canonical_hash
    from tests.test_refund_decision import decision

    p = fresh(archived().profile).decision_model.model_copy(
        update={
            "model_api_mode": api_mode,
            "api_address_fingerprint": canonical_hash("https://offline.invalid/v1"),
        }
    )
    answer = decision().model_dump_json()
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
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
                            "delta": {"role": "assistant", "content": answer},
                            "finish_reason": None,
                        }
                    ]
                },
                base
                | {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
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
            body = (
                "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n"
            )
        else:
            message = {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            }
            response = {
                "id": "resp_mock",
                "object": "response",
                "created_at": 0,
                "model": p.model_name,
                "status": "completed",
                "output": [message],
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
                    "type": "response.output_text.delta",
                    "delta": answer,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_mock",
                    "sequence_number": 0,
                },
                {
                    "type": "response.completed",
                    "response": response,
                    "sequence_number": 1,
                },
            ]
            body = "".join(
                "event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n"
                for e in events
            )
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=body
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = RefundModelClient(
            SimpleNamespace(
                openai_model=p.model_name,
                openai_base_url="https://offline.invalid/v1",
                openai_api_key="offline",
            ),
            client_factory=lambda **kw: ChatOpenAI(**kw, http_client=http),
        )
        audits = []
        try:
            response = client.decide(
                [SystemMessage(content=p.system_prompt)],
                profile=p,
                model_step=6,
                attempt=1,
                require_tool=False,
                final_round=True,
                audit_hook=audits.append,
            )
        finally:
            client.close()
    assert response.decision == decision() and len(payloads) == 1
    payload = payloads[0]
    fmt = (
        payload["response_format"]["json_schema"]
        if api_mode == "chat_completions"
        else payload["text"]["format"]
    )
    assert fmt["name"] == "refund_decision" and fmt["strict"] is True
    assert fmt["schema"]["additionalProperties"] is False
    assert payload["tool_choice"] == "none" and len(payload["tools"]) == 4
    assert audits[-1].outcome == "succeeded"
