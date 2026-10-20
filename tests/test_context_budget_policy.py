"""New token allocation is explicit; historical frozen settings keep their values."""

import importlib
from hashlib import sha256

import pytest

from resolv.core.config import ContextSettings, SummaryModelSettings


def policy():
    assert importlib.util.find_spec("resolv.core.context_budget"), (
        "missing 60000-token budget policy"
    )
    return importlib.import_module("resolv.core.context_budget")


def test_new_context_and_summary_budgets_override_old_env_without_writing_it(tmp_path):
    m = policy()
    env = tmp_path / "old.env"
    env.write_text(
        "CONTEXT_MODEL_CONTEXT_WINDOW_TOKENS=256000\nCONTEXT_TOKENIZER_PROFILE=o200k_base\nCONTEXT_APP_MAX_INPUT_TOKENS=12000\nCONTEXT_PROVIDER_INPUT_OVERHEAD_TOKENS=5000\nSUMMARY_MODEL_NAME=offline-summary\nSUMMARY_API_KEY=offline\nSUMMARY_BASE_URL=https://invalid.example/v1\nSUMMARY_MODEL_CONTEXT_WINDOW_TOKENS=256000\nSUMMARY_INPUT_MAX_TOKENS=12000\nSUMMARY_PROVIDER_INPUT_OVERHEAD_TOKENS=5000\n"
    )
    digest = sha256(env.read_bytes()).hexdigest()
    old = ContextSettings(_env_file=env)
    old_summary = SummaryModelSettings(_env_file=env)
    context = m.apply_context_budget(old)
    summary = m.apply_summary_budget(old_summary)
    assert (
        context.input_limit,
        context.history_soft_budget_tokens,
        context.recent_history_target_tokens,
        context.summary_max_tokens,
        context.knowledge_context_max_tokens,
        context.reserved_output_tokens,
        context.safety_margin_tokens,
    ) == (60000, 30000, 10000, 4000, 15000, 10000, 5000)
    assert (
        summary.input_limit,
        summary.summary_max_tokens,
        summary.reserved_output_tokens,
        summary.safety_margin_tokens,
    ) == (60000, 4000, 10000, 5000)
    assert (
        context.provider_input_overhead_tokens
        == summary.provider_input_overhead_tokens
        == 5000
    )
    assert (
        context.model_context_window_tokens
        == summary.model_context_window_tokens
        == 256000
    )
    assert context.summary_trigger_ratio == old.summary_trigger_ratio == 0.8
    assert old.input_limit == old_summary.input_limit == 12000
    assert sha256(env.read_bytes()).hexdigest() == digest
    assert m.apply_context_budget(context).model_dump() == context.model_dump()
    assert m.apply_summary_budget(summary).model_dump() == summary.model_dump()


def test_budget_policy_respects_physical_model_window_and_revalidates():
    m = policy()
    base = ContextSettings(
        _env_file=None,
        model_context_window_tokens=40000,
        tokenizer_profile="o200k_base",
    )
    assert m.apply_context_budget(base).input_limit == 25000
    base = ContextSettings(
        _env_file=None,
        model_context_window_tokens=14000,
        tokenizer_profile="o200k_base",
    )
    with pytest.raises(ValueError):
        m.apply_context_budget(base)


def test_new_refund_profile_scales_evidence_and_preserves_old_profile_roundtrip():
    m = policy()
    from resolv.core.refund_config import default_refund_profile
    from resolv.schemas.knowledge_branch import FrozenJSON
    from resolv.schemas.refund import AdapterDescriptor
    from resolv.schemas.refund_flow import RefundProfile
    from tests.test_refund_model import profile

    adapter = AdapterDescriptor(
        namespace="offline", implementation_version="1", authorization_version="1"
    )
    legacy = RefundProfile(
        query_model=profile("refund_query"),
        decision_model=profile(),
        summary_profile=FrozenJSON.from_value({}),
        adapter=adapter,
    )
    frozen = legacy.model_dump_json()
    fingerprint = legacy.fingerprint
    current = default_refund_profile(
        understanding_config=legacy.query_model,
        business_config=legacy.decision_model,
        summary_profile={},
        adapter_descriptor=adapter,
    )
    assert current.config_version == 7 and current.evidence_max_tokens == 15000
    for model in (current.query_model, current.decision_model):
        assert model.input_limit == 60000
        assert (
            model.reserved_output_tokens == 10000 and model.safety_margin_tokens == 5000
        )
        assert (
            model.recent_history_target_tokens == 10000
            and model.provider_input_overhead_tokens == 6500
        )
    assert current.summary_profile.value["input_max_tokens"] == 60000
    assert current.summary_profile.value["summary_max_tokens"] == 4000
    restored = RefundProfile.model_validate_json(frozen)
    assert restored.fingerprint == fingerprint and restored.model_dump_json() == frozen
    assert restored.config_version == 1 and restored.evidence_max_tokens == 3000
    assert restored.decision_model.input_limit == 12000
    assert m.model_budget()["app_max_input_tokens"] == 60000


def test_new_business_config_uses_same_budget_without_changing_legacy_contract():
    policy()
    from types import SimpleNamespace

    from resolv.core.business_model import make_business_config
    from resolv.core.config import BusinessExecutionSettings
    from resolv.schemas.branch_execution import BusinessRunConfig

    config = make_business_config(
        SimpleNamespace(
            openai_model="offline", openai_base_url="https://invalid.example/v1"
        ),
        BusinessExecutionSettings(_env_file=None),
    )
    assert config.input_limit == 60000 and config.reserved_output_tokens == 10000
    assert config.provider_input_overhead_tokens == 6500 and config.max_agent_steps == 6
    legacy = BusinessRunConfig(
        model_name="offline",
        api_address_fingerprint="a" * 64,
        system_prompt="offline",
        tool_schema_fingerprint="b" * 64,
    )
    assert legacy.input_limit == 12000 and legacy.reserved_output_tokens == 2000


def test_current_refund_memory_builder_uses_scaled_auxiliary_budgets():
    from types import SimpleNamespace

    from resolv.core.refund_config import default_refund_profile
    from resolv.schemas.refund import AdapterDescriptor
    from resolv.services.refund_flow import _MemoryBuilder
    from tests.test_refund_model import Counter, profile

    p = default_refund_profile(
        understanding_config=profile("refund_query"),
        business_config=profile(),
        summary_profile={},
        adapter_descriptor=AdapterDescriptor(
            namespace="offline", implementation_version="1", authorization_version="1"
        ),
    )
    memory = _MemoryBuilder(SimpleNamespace(profile=p), Counter())
    assert memory.settings.history_soft_budget_tokens == 30000
    assert memory.settings.summary_max_tokens == 4000
    assert memory.settings.knowledge_context_max_tokens == 15000


def test_saved_F1_context_fits_new_budget_without_dropping_evidence_or_observations():
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from resolv.core.refund_config import default_refund_profile
    from resolv.schemas.refund import RefundFlowError
    from resolv.schemas.refund_flow import RefundRunRecord
    from resolv.services.context import TokenCounter
    from resolv.services.refund_context import build_refund_context

    p = (
        Path(__file__).resolve().parents[1]
        / "docs/superpowers/acceptance/2026-10-16-refund-flow/R2-C2.json"
    )
    saved = json.loads(p.read_text())["cases"][0]["refund_record"]
    legacy = RefundRunRecord.model_validate_json(json.dumps(saved))
    frozen = legacy.model_dump_json()
    counter = TokenCounter(legacy.profile.decision_model.tokenizer_profile)

    def build(run, profile):
        return build_refund_context(
            run,
            legacy.request,
            legacy.binding,
            legacy.evidence,
            legacy.observations,
            history=[],
            summary=None,
            counter=counter,
            profile=profile,
        )

    with pytest.raises(RefundFlowError, match="refund_model_input_budget_exceeded"):
        build(legacy, legacy.profile)
    current = default_refund_profile(
        understanding_config=legacy.profile.query_model,
        business_config=legacy.profile.decision_model,
        summary_profile=legacy.profile.summary_profile.value,
        adapter_descriptor=legacy.profile.adapter,
    )
    run = SimpleNamespace(
        primary_json=legacy.primary_json,
        model_requests=legacy.model_requests,
        profile=current,
    )
    value = build(run, current)
    assert 12000 < value.input_estimate < 60000
    assert value.messages.value[2]["data"]["content"] == legacy.evidence.serialized
    assert {
        m["data"]["tool_call_id"] for m in value.messages.value if m["type"] == "tool"
    } == {o.tool_call_id for o in legacy.observations}
    assert (
        legacy.model_dump_json() == frozen
        and legacy.profile.decision_model.input_limit == 12000
    )
