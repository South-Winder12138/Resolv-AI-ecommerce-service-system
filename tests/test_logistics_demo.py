from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from resolv.schemas.logistics import LogisticsFailure
from tests.conversation_fixtures import MemoryCheckpoints
from tests.test_logistics_contracts import api
from tests.test_logistics_execution import Client
from tests.test_logistics_gate_support import prepared

__all__ = ["prepared"]


class DemoClient(Client):
    def query(self, tracking_no, *, request_id):
        value = super().query(tracking_no, request_id=request_id)
        return value.model_copy(update={"dataset_hash": self.profile.dataset_hash})


def test_start_requires_prepared_ownership_and_schema(tmp_path):
    module = api("resolv.services.logistics_demo")
    with pytest.raises(ValueError):
        module.create_logistics_demo_app(tmp_path / "missing.json")
    assert list(tmp_path.iterdir()) == []


def test_real_demo_requires_injected_budget_guard(tmp_path):
    module = api("resolv.services.logistics_demo")
    with pytest.raises(ValueError, match="budget guard"):
        module.create_logistics_demo_app(tmp_path / "missing.json", real_models=True)


def test_default_demo_has_zero_real_models(prepared, monkeypatch):
    h = prepared
    module = api("resolv.services.logistics_demo")
    original = h.module.owned_logistics_resources
    remote = DemoClient()
    remote.profile = h.module.owned_profile(h.manifest)

    @contextmanager
    def resources(path):
        with original(path) as owned:
            owned.checkpoint_factory = MemoryCheckpoints()
            yield owned

    monkeypatch.setattr(module, "owned_logistics_resources", resources)
    monkeypatch.setattr(module, "MCPLogisticsClient", lambda profile: remote)
    # A provider constructor would fail if default startup ever called it.
    def forbidden(*args, **kwargs):
        pytest.fail("fake demo constructed a real model")

    monkeypatch.setattr(module, "get_business_model", forbidden)
    monkeypatch.setattr(module, "get_understanding_model", forbidden)
    app = module.create_logistics_demo_app(h.manifest)
    with TestClient(app) as http:
        assert http.get("/health").json()["model_mode"] == "fake"
        service = app.state.conversation_runtime
        c = service.create()
        run, _ = service.accept(c.conversation_id, "fake", "查询订单 MOCK001 的物流", "agent")
        events = list(service.run(run.run_id))
        assert service.get_run(run.run_id).status == "completed", events
        assert len(remote.calls) == 1 and remote.calls[0][0] == "MOCKSF001"
        assert service.detail(c.conversation_id).messages[-1].text.startswith("【模拟物流】")
        assert service.get_run(run.run_id).result_json["schema_version"] == 3
    assert h.module.validate_logistics_ownership(h.manifest)["state"] == "ready"


def test_fake_demo_preserves_technical_failure(prepared, monkeypatch):
    h = prepared
    module = api("resolv.services.logistics_demo")
    original = h.module.owned_logistics_resources
    remote = DemoClient()
    remote.profile = h.module.owned_profile(h.manifest)
    remote.error = LogisticsFailure("logistics_timeout", True)

    @contextmanager
    def resources(path):
        with original(path) as owned:
            owned.checkpoint_factory = MemoryCheckpoints()
            yield owned

    monkeypatch.setattr(module, "owned_logistics_resources", resources)
    monkeypatch.setattr(module, "MCPLogisticsClient", lambda profile: remote)
    app = module.create_logistics_demo_app(h.manifest)
    with TestClient(app):
        service = app.state.conversation_runtime
        c = service.create()
        run, _ = service.accept(c.conversation_id, "timeout", "查询订单 MOCK003 的物流", "agent")
        events = list(service.run(run.run_id))
        assert any(e.type == "error" and e.data["code"] == "logistics_timeout" for e in events), events
        assert len(service.detail(c.conversation_id).messages) == 1 and len(remote.calls) == 1


def test_zero_model_budget_blocks_before_provider_send(prepared, monkeypatch):
    from resolv.core.config import ContextSettings, Settings
    from resolv.schemas.understanding import UnderstandingError

    h = prepared
    module = api("resolv.services.logistics_demo")
    original = h.module.owned_logistics_resources

    @contextmanager
    def resources(path):
        with original(path) as owned:
            owned.checkpoint_factory = MemoryCheckpoints()
            yield owned

    class ForbiddenProvider:
        def bind(self, **kwargs):
            pytest.fail("zero-budget provider was reached")

    class Denied:
        def invoke(self, *args, **kwargs):
            raise UnderstandingError("model_budget_exceeded", False)

    monkeypatch.setattr(module, "owned_logistics_resources", resources)
    monkeypatch.setattr(module, "Settings", lambda **kwargs: Settings(_env_file=None, openai_api_key="offline", openai_base_url="https://offline.invalid/v1"))
    monkeypatch.setattr(module, "ContextSettings", lambda **kwargs: ContextSettings(_env_file=None, model_context_window_tokens=256000, tokenizer_profile="o200k_base"))
    monkeypatch.setattr(module, "get_understanding_model", lambda *args: ForbiddenProvider())
    monkeypatch.setattr(module, "get_business_model", lambda *args: ForbiddenProvider())
    monkeypatch.setattr(module, "close_model", lambda *args: None)
    app = module.create_logistics_demo_app(h.manifest, real_models=True, model_guard=lambda purpose, client: Denied())
    with TestClient(app):
        service = app.state.conversation_runtime
        c = service.create()
        run, _ = service.accept(c.conversation_id, "zero", "查询订单 MOCK001 的物流", "agent")
        events = list(service.run(run.run_id))
        assert any(e.type == "error" and e.data["code"] == "model_budget_exceeded" for e in events), events
