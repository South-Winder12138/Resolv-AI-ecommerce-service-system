import importlib
from contextlib import contextmanager

import pytest

from resolv.schemas.understanding import UnderstandingError
from tests.branch_fixtures import branch_factory

__all__ = ["branch_factory"]


def test_required_understanding_ignores_optional_switch(monkeypatch):
    import resolv.services.understanding_runtime as module

    sentinel = object()
    calls = []

    @contextmanager
    def resources(options):
        calls.append(options.enabled)
        yield sentinel

    monkeypatch.setenv("UNDERSTANDING_ENABLED", "false")
    monkeypatch.setattr(module, "resources", resources)
    try:
        with module.understanding_runtime(required=True) as service:
            assert service is sentinel
    except TypeError:
        pytest.fail("required understanding runtime missing")
    assert calls == [False]


def test_required_understanding_initialization_error_is_not_optional(monkeypatch):
    import resolv.services.understanding_runtime as module

    @contextmanager
    def failed(options):
        raise RuntimeError("credential secret")
        yield

    monkeypatch.setattr(module, "resources", failed)
    try:
        with (
            pytest.raises(UnderstandingError, match="understanding_unavailable"),
            module.understanding_runtime(required=True),
        ):
            pytest.fail("required runtime bypassed")
    except TypeError:
        pytest.fail("required understanding runtime missing")


def test_routed_runtime_requires_understanding_and_never_loads_knowledge(monkeypatch):
    try:
        module = importlib.import_module("resolv.services.routed_runtime")
    except ModuleNotFoundError:
        pytest.fail("routed runtime missing")

    @contextmanager
    def unavailable(*, required=False):
        assert required is True
        raise UnderstandingError("understanding_unavailable", True)
        yield

    monkeypatch.setattr(module, "understanding_runtime", unavailable)
    from resolv.schemas.conversation import ConversationError

    with (
        pytest.raises(ConversationError, match="understanding_unavailable"),
        module.routed_conversation_runtime(),
    ):
        pytest.fail("routed runtime bypassed understanding")


def test_required_understanding_close_failure_is_redacted(monkeypatch):
    import resolv.services.understanding_runtime as module

    @contextmanager
    def resources(options):
        try:
            yield object()
        finally:
            raise RuntimeError("credential secret")

    monkeypatch.setattr(module, "resources", resources)
    try:
        with (
            pytest.raises(
                UnderstandingError, match="understanding_unavailable"
            ) as caught,
            module.understanding_runtime(required=True),
        ):
            pass
        assert "secret" not in str(caught.value)
    except TypeError:
        pytest.fail("required understanding runtime missing")


def test_recovery_refuses_changed_provider_identity_before_write(
    branch_factory, tool_database
):
    from langchain_core.messages import AIMessage
    from sqlalchemy import select

    from resolv.models.tool_execution import ToolExecutionReceipt
    from resolv.schemas.branch_execution import BusinessRunConfig
    from tests.test_routing_graph import decision, harness, tool

    service, run, _ = harness(
        branch_factory,
        tool_database,
        "取消订单 MH002",
        "order",
        [decision(tool("cancel_order", "MH002")), AIMessage(content="结果")],
    )
    events = list(service.run(run.run_id))
    approval = next(e.data for e in events if e.type == "approval_required")
    service.decide(approval["approval_id"], run.conversation_id, True)
    saved = BusinessRunConfig.model_validate(
        service.get_run(run.run_id).business_config
    )
    service.business_run_config = saved.model_copy(
        update={"api_address_fingerprint": "0" * 64}
    )
    events = list(service.resume(run.conversation_id, run.run_id))
    assert any(
        e.type == "error" and e.data["code"] == "run_state_incompatible" for e in events
    ), events
    with tool_database() as session:
        assert session.scalar(select(ToolExecutionReceipt)) is None


def test_routed_business_context_rehydrates_without_invalid_knowledge_budget():
    module = importlib.import_module("resolv.services.routed_runtime")
    from resolv.core.config import ContextSettings
    from tests.test_business_model import profile

    assert hasattr(module, "business_context_settings"), (
        "validated routed context construction missing"
    )
    settings = module.business_context_settings(profile())
    assert ContextSettings(_env_file=None, **settings.model_dump()) == settings
    assert settings.input_limit == 12000 and settings.tokenizer_profile == "o200k_base"
