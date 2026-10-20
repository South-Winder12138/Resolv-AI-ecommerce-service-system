import importlib
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient


def module():
    try:
        return importlib.import_module("resolv.services.understanding_runtime")
    except ModuleNotFoundError:
        pytest.fail("optional understanding runtime missing")


def test_disabled_runtime_never_constructs_external_resources(monkeypatch):
    mod = module()
    monkeypatch.setenv("UNDERSTANDING_ENABLED", "false")
    monkeypatch.setattr(
        mod, "resources", lambda *_: pytest.fail("external resource initialized")
    )
    with mod.understanding_runtime() as value:
        assert value is None


def test_invalid_optional_settings_do_not_break_chat_startup(monkeypatch):
    mod = module()
    monkeypatch.setenv("UNDERSTANDING_ENABLED", "true")
    monkeypatch.setenv("UNDERSTANDING_TIMEOUT_SECONDS", "0")
    with mod.understanding_runtime() as service:
        assert service is None


def test_enabled_runtime_closes_resources_once(monkeypatch, understanding_service):
    mod = module()
    calls = []

    @contextmanager
    def resources(_):
        calls.append("open")
        try:
            yield understanding_service
        finally:
            calls.append("close")

    monkeypatch.setattr(mod, "resources", resources)
    monkeypatch.setenv("UNDERSTANDING_ENABLED", "true")
    with mod.understanding_runtime() as service:
        assert service is understanding_service
    assert calls == ["open", "close"]


def test_missing_store_is_independently_unavailable(monkeypatch):
    mod = module()

    @contextmanager
    def resources(_):
        raise RuntimeError("private connection secret")
        yield

    monkeypatch.setattr(mod, "resources", resources)
    monkeypatch.setenv("UNDERSTANDING_ENABLED", "true")
    with mod.understanding_runtime() as service:
        assert service is None


def test_main_mounts_the_required_understanding_service_owned_by_routing(
    api_runtime, monkeypatch, understanding_service
):
    module()
    from resolv import main

    calls = []

    @contextmanager
    def routed():
        calls.append("open")
        api_runtime.understanding = understanding_service
        yield api_runtime
        calls.append("close")

    def forbidden():
        pytest.fail("optional understanding bypass constructed a second runtime")

    monkeypatch.setattr(main, "routed_conversation_runtime", routed, raising=False)
    monkeypatch.setattr(main, "understanding_runtime", forbidden, raising=False)
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert main.app.state.conversation_runtime is api_runtime
        assert main.app.state.understanding_runtime is understanding_service
    assert calls == ["open", "close"]
