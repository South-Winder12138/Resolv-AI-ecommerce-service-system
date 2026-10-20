"""Final-round convergence is bounded; frozen v1/v2 histories remain replayable."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import SystemMessage, ToolMessage, messages_from_dict
from sqlalchemy import func, select

from resolv.core.refund_config import default_refund_profile
from resolv.models.conversation import ConversationRun
from resolv.models.refund_application import RefundApplication
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_flow import RefundRunRecord
from resolv.services.context import TokenCounter
from resolv.services.refund_context import build_refund_context
from tests.refund_flow_fixtures import drive, flow_case
from tests.test_refund_model import harness, tool_chunks

ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-C3-retry-1.json"
)


def archived():
    return RefundRunRecord.model_validate_json(
        json.dumps(json.loads(ARCHIVE.read_text())["cases"][0]["refund_record"])
    )


def fresh(profile):
    # These archived round-policy tests describe frozen v6. Fresh v7 has its
    # own native-return/SDK/service tests; do not silently relabel v6 evidence.
    from resolv.core.refund_prompts import (
        CONTEXT_IDENTITY_PROMPT_VERSION,
        decision_system_prompt,
    )
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund_flow import RefundDecisionOutput
    from resolv.services.hybrid_fingerprints import canonical_hash
    from resolv.services.refund_tools import refund_tool_schemas

    current = default_refund_profile(
        understanding_config=profile.query_model,
        business_config=profile.decision_model,
        summary_profile=profile.summary_profile.value,
        adapter_descriptor=profile.adapter,
    )
    model = current.decision_model.model_copy(
        update={
            "system_prompt": decision_system_prompt(
                round_control=True, complete_output=True, context_identity=True
            ),
            "prompt_version": CONTEXT_IDENTITY_PROMPT_VERSION,
            "output_schema": FrozenJSON.from_value(
                RefundDecisionOutput.model_json_schema()
            ),
            "tool_schema_fingerprint": canonical_hash(refund_tool_schemas()),
        }
    )
    return current.model_copy(
        update={
            "selection": profile.selection,
            "config_version": 6,
            "decision_model": model,
        }
    )


def prefix(record, profile, step):
    responses = record.model_requests[: step - 1]
    calls = {c.id for r in responses for c in r.tool_calls}
    observations = tuple(o for o in record.observations if o.tool_call_id in calls)
    run = SimpleNamespace(
        primary_json=record.primary_json, profile=profile, model_requests=responses
    )
    return build_refund_context(
        run,
        record.request,
        record.binding,
        record.evidence,
        observations,
        history=[],
        summary=None,
        counter=TokenCounter(profile.decision_model.tokenizer_profile),
        profile=profile,
    )


def test_new_round_policy_is_frozen_separately_from_60000_budget():
    saved = archived()
    profile = fresh(saved.profile)
    assert profile.config_version == 6
    assert (
        profile.decision_model.prompt_version == "refund-decision-v5-context-identity"
    )
    assert profile.decision_model.input_limit == 60000
    assert saved.profile.config_version == 2


@pytest.mark.parametrize("step", range(1, 7))
def test_round_hint_reaches_model_without_losing_policy_or_native_results(step):
    saved = archived()
    context = prefix(saved, fresh(saved.profile), step)
    messages = messages_from_dict(context.messages.value)
    hint = messages[-1]
    assert isinstance(hint, SystemMessage)
    control = json.loads(hint.content.split("\n", 1)[0])["refund_round_control"]
    assert control == {
        "current_round": step,
        "max_rounds": 6,
        "remaining_rounds": 6 - step,
        "final_round": step == 6,
        "tool_calls_allowed": step < 6,
    }
    assert messages[2].content == saved.evidence.serialized
    expected = {c.id for r in saved.model_requests[: step - 1] for c in r.tool_calls}
    assert {m.tool_call_id for m in messages if isinstance(m, ToolMessage)} == expected
    if step == 6:
        assert "不得调用工具" in hint.content and "undetermined" in hint.content
        assert "human_draft" in hint.content


def test_historical_v2_six_round_inputs_and_estimates_are_byte_exact():
    saved = archived()
    frozen = saved.model_dump_json()
    for step, response in enumerate(saved.model_requests, 1):
        context = prefix(saved, saved.profile, step)
        assert context.input_hash == response.input_hash
        # Legacy context reserves required-tool parameters even on later rounds.
        assert context.input_estimate >= response.audit.input_estimate
    restored = RefundRunRecord.model_validate_json(frozen)
    assert restored.model_dump_json() == frozen


@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
def test_final_model_call_forbids_tools_and_rejects_provider_violation(api_mode):
    client, sdk, audits, profile, messages = harness(tool_chunks())
    profile = profile.model_copy(update={"model_api_mode": api_mode})
    with pytest.raises(RefundFlowError, match="refund_model_output_invalid"):
        client.decide(
            messages,
            profile=profile,
            model_step=6,
            attempt=1,
            require_tool=False,
            final_round=True,
            audit_hook=audits.append,
        )
    assert sdk.bound[-1]["tool_choice"] == "none"
    assert audits[-1].outcome == "failed" and sdk.calls == 1


def test_final_round_and_required_tool_are_rejected_before_provider_work():
    client, sdk, audits, profile, messages = harness(tool_chunks())
    with pytest.raises(RefundFlowError, match="refund_model_profile_mismatch"):
        client.decide(
            messages,
            profile=profile,
            model_step=6,
            attempt=1,
            require_tool=True,
            final_round=True,
            audit_hook=audits.append,
        )
    assert not audits and sdk.calls == 0


def test_real_sdk_wire_has_none_then_default_without_cached_binding_leak():
    import httpx
    from langchain_openai import ChatOpenAI

    from resolv.core.refund_model import RefundModelClient, refund_input_estimate
    from tests.test_refund_decision import decision
    from tests.test_refund_model import Counter
    from tests.test_refund_model import profile as make_profile

    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        final = len(payloads) == 2
        delta = (
            {"role": "assistant", "content": decision().model_dump_json()}
            if final
            else {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "offline-call",
                        "type": "function",
                        "function": {"name": "query_refund_order", "arguments": "{}"},
                    }
                ],
            }
        )
        base = {
            "id": "offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fixture",
        }
        chunks = [
            base | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            base
            | {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop" if final else "tool_calls",
                    }
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
        body = (
            "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=body
        )

    transport = httpx.MockTransport(handle)
    with httpx.Client(transport=transport) as http:
        profile = make_profile()
        client = RefundModelClient(
            SimpleNamespace(
                openai_model="fixture",
                openai_api_key="offline",
                openai_base_url="https://offline.invalid/v1",
            ),
            client_factory=lambda **kw: ChatOpenAI(**kw, http_client=http),
            counter_factory=lambda _: Counter(),
        )
        messages = [SystemMessage(content=profile.system_prompt)]
        audits = []
        try:
            for step, required, final in [
                (1, True, False),
                (6, False, True),
                (2, False, False),
            ]:
                response = client.decide(
                    messages,
                    profile=profile,
                    model_step=step,
                    attempt=1,
                    require_tool=required,
                    final_round=final,
                    audit_hook=audits.append,
                )
                assert response.audit.input_estimate == refund_input_estimate(
                    messages,
                    profile,
                    Counter(),
                    require_tool=required,
                    final_round=final,
                )
                assert (response.decision is not None) == final
        finally:
            client.close()
    assert [p.get("tool_choice") for p in payloads] == ["required", "none", None]
    assert all(len(p["tools"]) == 4 for p in payloads)


def test_r1_native_harness_uses_same_new_final_round_control():
    from scripts.smoke_refund_flow_model import RealTransport

    saved = archived()
    transport = RealTransport.__new__(RealTransport)
    transport.profile = fresh(saved.profile)
    transport.native_step = 5
    transport.messages = []
    transport.native_messages = []
    seen = []

    def decide(messages, **kwargs):
        seen.append((messages, kwargs))
        raise RefundFlowError("offline_stop")

    transport.client = SimpleNamespace(decide=decide)
    with pytest.raises(RefundFlowError, match="offline_stop"):
        transport.native_round(6, lambda _: None)
    messages, kwargs = seen[0]
    assert kwargs.get("final_round") is True and kwargs["require_tool"] is False
    control = json.loads(messages[-1].content.split("\n", 1)[0])["refund_round_control"]
    assert control["current_round"] == 6 and control["tool_calls_allowed"] is False
    assert transport.native_messages == []


def upgrade(h, max_steps=6):
    profile = fresh(h.profile).model_copy(update={"max_agent_steps": max_steps})
    with h.factory.begin() as s:
        s.get(ConversationRun, h.run.run_id).refund_config = profile.model_dump(
            mode="json"
        )


@pytest.mark.parametrize("ending", ["eligible", "undetermined"])
def test_five_tool_rounds_then_validated_final_decision_without_new_reads(
    tmp_path, ending
):
    with flow_case(tmp_path) as h:
        upgrade(h)
        original = h.client.decide
        requests = []

        def decide(messages, *, final_round=False, **kwargs):
            requests.append((kwargs["model_step"], kwargs["require_tool"], final_round))
            h.client.mode = (
                ending
                if final_round
                else "eligible"
                if kwargs["model_step"] == 1
                else "loop"
            )
            return original(messages, **kwargs)

        h.client.decide = decide
        record = drive(h)
        assert record.decision is not None and record.decision.verdict == ending
        assert not record.step_limit and record.error_code is None
        assert len(record.model_requests) == 6 and len(record.observations) == 7
        assert requests == [(1, True, False)] + [
            (step, False, step == 6) for step in range(2, 7)
        ]
        assert record.model_requests[-1].tool_calls == ()
        assert h.flow.main_agent(h.run.run_id, h.execution) == record
        with h.business() as s:
            assert s.scalar(select(func.count()).select_from(RefundApplication)) == 0


def test_service_also_rejects_final_tool_calls_from_injected_client(tmp_path):
    with flow_case(tmp_path, mode="loop") as h:
        upgrade(h)
        original = h.client.decide
        h.client.decide = lambda messages, *, final_round=False, **kw: original(
            messages, **kw
        )
        with pytest.raises(RefundFlowError, match="refund_model_output_invalid"):
            drive(h)
        record = h.flow._get(h.run.run_id)
        assert record.error_code == "refund_model_output_invalid"
        assert len(record.model_requests) == len(record.observations) == 5


def test_shortened_ceiling_is_final_and_cannot_bypass_mandatory_observations(tmp_path):
    with flow_case(tmp_path) as h:
        upgrade(h, max_steps=1)
        original = h.client.decide
        seen = []

        def decide(messages, *, final_round=False, **kwargs):
            seen.append((kwargs["require_tool"], final_round))
            return original(messages, **kwargs)

        h.client.decide = decide
        with pytest.raises(RefundFlowError, match="refund_model_output_invalid"):
            drive(h)
        assert seen == [(False, True)]
        assert h.flow._get(h.run.run_id).observations == ()


def test_final_json_still_requires_actual_mandatory_observations(tmp_path):
    with flow_case(tmp_path, mode="loop") as h:
        upgrade(h)
        original = h.client.decide

        def decide(messages, *, final_round=False, **kwargs):
            h.client.mode = "undetermined" if final_round else "loop"
            return original(messages, **kwargs)

        h.client.decide = decide
        with pytest.raises(
            RefundFlowError, match="refund_required_observation_missing"
        ):
            drive(h)
        assert h.flow._get(h.run.run_id).decision is None


@pytest.mark.parametrize(
    "point", ["refund_after_model_request_saved", "refund_after_decision_saved"]
)
def test_saved_final_response_resumes_without_another_model_or_tool_call(
    tmp_path, point
):
    with flow_case(tmp_path) as h:
        upgrade(h)
        original = h.client.decide

        def decide(messages, *, final_round=False, **kwargs):
            h.client.mode = (
                "eligible" if final_round or kwargs["model_step"] == 1 else "loop"
            )
            return original(messages, **kwargs)

        def crash(name):
            if name == point and h.counts["decide"] == 6:
                h.service.fault_hook = None
                raise RuntimeError("owned final-round crash")

        h.client.decide, h.service.fault_hook = decide, crash
        with pytest.raises(RuntimeError, match="owned final-round crash"):
            drive(h)
        before = h.counts.copy()
        saved = h.flow._get(h.run.run_id)
        record = drive(h)
        assert record.profile.model_dump_json() == saved.profile.model_dump_json()
        assert record.decision.verdict == "eligible"
        assert h.counts == before and len(record.model_requests) == 6
