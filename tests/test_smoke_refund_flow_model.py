import importlib

import pytest


def module():
    assert importlib.util.find_spec("scripts.smoke_refund_flow_model"), (
        "missing explicit R1 transport gate"
    )
    return importlib.import_module("scripts.smoke_refund_flow_model")


def test_real_gate_requires_explicit_ownership_and_report():
    m = module()
    with pytest.raises(SystemExit):
        m.main([])
    with pytest.raises(ValueError):
        m.validate_model_gate({"passed": True, "calls": [], "business_writes": 0})


def test_model_gate_refuses_missing_usage_and_invalid_native_scope():
    m = module()
    report = valid_report()
    m.validate_model_gate(report)
    for field in ("input_tokens", "output_tokens"):
        changed = valid_report()
        changed["calls"][0]["audit"][field] = None
        with pytest.raises(ValueError):
            m.validate_model_gate(changed)
    changed = valid_report()
    changed["advertised_tools"].append("apply_refund")
    with pytest.raises(ValueError):
        m.validate_model_gate(changed)
    changed = valid_report()
    changed["calls"][1]["tool_names"] = ["cancel_order"]
    with pytest.raises(ValueError):
        m.validate_model_gate(changed)


def test_failed_gate_preserves_unknown_usage_without_retry(tmp_path):
    m = module()
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("never expose credentials")

    result = m.run_transport_calls(invoke=fail)
    assert result["passed"] is False and len(calls) == 1
    assert result["calls"][0]["audit"]["input_tokens"] is None
    assert "credentials" not in str(result)


def valid_report():
    purposes = [
        "refund_query",
        "refund_decision",
        "refund_decision",
        "refund_decision",
        "refund_summary",
        "refund_summary",
    ]
    return {
        "passed": True,
        "business_writes": 0,
        "source_fingerprints": {"example": "a" * 64},
        "advertised_tools": [
            "query_refund_order",
            "query_refund_payment",
            "query_refund_delivery",
            "query_refund_status",
        ],
        "boundary_mutation_rejected": True,
        "calls": [
            {
                "purpose": p,
                "model_name": "fixture",
                "protocol": "chat_completions",
                "profile_hash": "c" * 64,
                "scenario": str(i),
                "input_limit": 12000,
                "output_limit": 2000,
                "structure_valid": True,
                "tool_names": ["query_refund_order"] if i in (1, 3) else [],
                "audit": {
                    "outcome": "succeeded",
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "input_estimate": 100,
                    "input_hash": "b" * 64,
                },
            }
            for i, p in enumerate(purposes)
        ],
    }


def test_real_transport_assembly_is_offline_and_uses_real_contracts(monkeypatch):
    from contextlib import ExitStack

    from resolv.core import config

    # Import consumers before temporarily replacing config constructors. A
    # first import under the patch would retain the stub after fixture cleanup.
    from resolv.services import routed_runtime

    settings = config.Settings(
        _env_file=None,
        openai_api_key="offline",
        openai_base_url="https://example.aliyuncs.com/compatible-mode/v1",
        openai_model="qwen3.8-flash",
    )
    context = config.ContextSettings(
        _env_file=None,
        model_context_window_tokens=256000,
        tokenizer_profile="o200k_base",
    )
    summary = config.SummaryModelSettings(
        _env_file=None,
        model_name="gpt-6-sol",
        api_key="offline",
        base_url="https://summary.invalid/v1",
        model_context_window_tokens=256000,
    )
    monkeypatch.setattr(config, "Settings", lambda: settings)
    monkeypatch.setattr(config, "ContextSettings", lambda: context)
    monkeypatch.setattr(config, "SummaryModelSettings", lambda: summary)
    monkeypatch.setattr(config, "UnderstandingSettings", lambda: None)
    from types import SimpleNamespace

    monkeypatch.setattr(
        config,
        "UnderstandingSettings",
        lambda: SimpleNamespace(
            reasoning_effort="low", provider_input_overhead_tokens=6500
        ),
    )
    m = module()
    with ExitStack() as stack:
        transport = m.RealTransport(stack)
        assert transport.profile.query_model.enable_thinking is True
        assert transport.profile.decision_model.enable_thinking is False
        assert transport.binding.order_id == "R1-A" and transport.evidence.windows
        assert transport.client._clients == {}
        from resolv.core.business_model import make_business_config

        effective = routed_runtime.business_context_settings(
            make_business_config(settings), context
        )
        assert effective.app_max_input_tokens == 60000
        assert effective.reserved_output_tokens == 10000
        assert effective.knowledge_context_max_tokens == 1


def test_summary_batch_cannot_hide_a_second_unreserved_provider_call():
    from types import SimpleNamespace

    m = module()
    sent = []
    reserved = []
    started = SimpleNamespace(
        outcome="started",
        model_dump=lambda **kwargs: {
            "outcome": "started",
            "input_tokens": None,
            "output_tokens": None,
        },
    )

    def invoke(**kwargs):
        kwargs["audit_hook"](started)
        sent.append("first")
        kwargs["audit_hook"](started)
        sent.append("second")

    result = m.run_transport_calls(
        invoke=invoke, before_call=lambda *a: reserved.append(a)
    )
    assert sent == ["first"] and len(reserved) == 1 and result["passed"] is False


def test_initialization_failure_has_safe_immutable_report(tmp_path, monkeypatch):
    import json

    m = module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    w = tmp_path / m.WORKSPACE
    w.mkdir(parents=True)
    monkeypatch.setattr(
        m, "committed_sources", lambda _: ("a" * 40, {"fixture": "b" * 64})
    )
    monkeypatch.setattr(m, "snapshot_formal_resources", lambda **kwargs: {})
    monkeypatch.setattr(
        m,
        "initialize_refund_resources",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("private credential")),
    )
    result = m.main(
        [
            "--ownership",
            str(w / "resource-ownership.json"),
            "--initialize",
            "--report-dir",
            str(w / "R1"),
        ]
    )
    assert result == 1
    report = json.loads((w / "R1/report.json").read_text())
    assert report["passed"] is False and "credential" not in str(report)
    with pytest.raises(ValueError):
        m.main(
            [
                "--ownership",
                str(w / "resource-ownership.json"),
                "--initialize",
                "--report-dir",
                str(w / "R1"),
            ]
        )


def test_gate_requires_model_protocol_and_frozen_profile_evidence():
    m = module()
    report = valid_report()
    report["calls"][0]["protocol"] = None
    with pytest.raises(ValueError):
        m.validate_model_gate(report)


def test_C1_native_rounds_each_reserve_and_keep_the_six_round_limit():
    from types import SimpleNamespace

    m = module()
    reserved, observed = [], []

    def invoke(*, purpose, scenario, audit_hook):
        observed.append(scenario)
        audit = {
            "outcome": "succeeded",
            "input_tokens": 80,
            "output_tokens": 20,
            "input_estimate": 100,
            "input_hash": "b" * 64,
        }
        audit_hook(
            SimpleNamespace(
                outcome="started",
                model_dump=lambda **kw: audit | {"outcome": "started"},
            )
        )
        audit_hook(SimpleNamespace(outcome="succeeded", model_dump=lambda **kw: audit))
        tools = scenario in {"native_round_1", "native_round_2", "hostile_instruction"}
        call = valid_report()["calls"][0] | {
            "tool_names": ["query_refund_order"] if tools else [],
            "output": {
                "tool_calls": [{"name": "query_refund_order"}] if tools else [],
                "decision": {} if scenario == "native_round_3" else None,
            },
        }
        return call

    invoke.boundary_mutation_rejected = True
    report = m.run_transport_calls(
        invoke=invoke,
        continuation=True,
        before_call=lambda p, s: reserved.append((p, s)),
    )
    report["source_fingerprints"] = {"fixture": "a" * 64}
    assert report["passed"] and len(reserved) == 7
    assert observed[1:4] == ["native_round_1", "native_round_2", "native_round_3"]
    m.validate_model_gate(report)


def test_C1_missing_final_stops_without_hostile_or_summary_calls():
    m = module()
    seen = []

    def invoke(*, purpose, scenario, audit_hook):
        seen.append(scenario)
        return {
            "structure_valid": True,
            "tool_names": ["query_refund_order"],
            "output": {
                "tool_calls": [{"name": "query_refund_order"}],
                "decision": None,
            },
        }

    report = m.run_transport_calls(invoke=invoke, continuation=True)
    assert not report["passed"]
    assert seen == ["query", *[f"native_round_{i}" for i in range(1, 7)]]


def test_failed_non_native_phase_does_not_report_a_stale_native_output():
    class Failed:
        def __init__(self):
            self.last_native_output = {"decision": "earlier native response"}

        def __call__(self, **kwargs):
            raise TimeoutError("offline")

    result = module().run_transport_calls(invoke=Failed(), continuation=True)
    assert not result["passed"] and "output" not in result["calls"][0]


def test_preflight_failed_native_round_clears_the_previous_round_output():
    from types import SimpleNamespace

    from resolv.schemas.refund import RefundFlowError

    t = module().RealTransport.__new__(module().RealTransport)
    t.profile = SimpleNamespace(
        decision_model=None, config_version=1, final_decision_round=lambda step: False
    )
    t.messages, t.native_messages, t.native_step = [], [], 5
    t.last_native_output = {"model_step": 5}

    def failed(*args, **kwargs):
        raise RefundFlowError("refund_model_input_budget_exceeded")

    t.client = SimpleNamespace(decide=failed)
    with pytest.raises(RefundFlowError, match="refund_model_input_budget_exceeded"):
        t.native_round(6, lambda a: None)
    assert t.native_step == 5 and t.last_native_output is None


@pytest.mark.parametrize("loop", [False, True])
def test_C1_actual_native_history_only_contains_observed_tool_results(tmp_path, loop):
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage, ToolMessage

    from resolv.schemas.knowledge_branch import FrozenJSON, canonical_json
    from resolv.schemas.refund import RefundFlowError
    from resolv.schemas.refund_flow import RefundModelResponse, RefundToolCall
    from resolv.services.hybrid_fingerprints import canonical_hash
    from tests.refund_fixtures import NOW
    from tests.refund_flow_fixtures import drive, flow_case

    with flow_case(tmp_path) as h:
        record = drive(h)
        t = module().RealTransport.__new__(module().RealTransport)
        t.profile, t.binding, t.request, t.evidence, t.now = (
            record.profile,
            record.binding,
            record.request,
            record.evidence,
            NOW,
        )
        t.observations = []
        t.messages = [
            HumanMessage(
                content=canonical_json(
                    {
                        "binding": record.binding.model_dump(mode="json"),
                        "request": record.request.model_dump(mode="json"),
                    }
                )
            ),
            HumanMessage(content=record.evidence.serialized),
        ]
        histories = []

        def decide(messages, *, profile, model_step, attempt, require_tool, audit_hook):
            histories.append(
                [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
            )
            if not loop and model_step == 3:
                return h.client.decide(
                    messages,
                    profile=profile,
                    model_step=model_step,
                    attempt=attempt,
                    require_tool=require_tool,
                    audit_hook=audit_hook,
                )
            names = (
                ["query_refund_order"]
                if loop or model_step == 1
                else ["query_refund_status", "query_refund_payment"]
            )
            calls = tuple(
                RefundToolCall(
                    id=f"c{model_step}-{i}", name=n, args=FrozenJSON.from_value({})
                )
                for i, n in enumerate(names)
            )
            a = h.client.audit(messages, profile, audit_hook, model_step)
            return RefundModelResponse(
                model_step=model_step,
                tool_calls=calls,
                input_hash=a.input_hash,
                output_hash=canonical_hash(
                    {
                        "tool_calls": [c.model_dump(mode="json") for c in calls],
                        "decision": None,
                    }
                ),
                audit=a,
            )

        t.client = SimpleNamespace(decide=decide)
        if loop:
            for i in range(1, 6):
                t.native_round(i, lambda a: None)
            with pytest.raises(RefundFlowError, match="r1_native_round_limit"):
                t.native_round(6, lambda a: None)
            assert len(t.observations) == 5
        else:
            t.native_round(1, lambda a: None)
            t.native_round(2, lambda a: None)
            result = t.native_round(3, lambda a: None)
            assert result.decision is not None and len(t.observations) == 3
            assert histories == [[], ["c1-0"], ["c1-0", "c2-0", "c2-1"]]
