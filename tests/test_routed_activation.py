from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.branch_fixtures import branch_factory

__all__ = ["branch_factory"]


def test_activation_requires_empty_legacy_preflight(branch_factory):
    from resolv.schemas.conversation import ConversationError
    from resolv.services.conversation import ConversationService
    from resolv.services.conversation_routing import assert_no_legacy_inflight
    from tests.conversation_fixtures import MemoryCheckpoints

    service = ConversationService(branch_factory, MemoryCheckpoints())
    c = service.create()
    service.accept(c.conversation_id, "old", "旧问题", "agent")
    with (
        branch_factory() as session,
        pytest.raises(ConversationError, match="legacy_run_requires_resolution"),
    ):
        assert_no_legacy_inflight(session, owner_id="local")


def test_main_requires_routed_runtime_even_when_optional_understanding_is_disabled(
    monkeypatch,
):
    import resolv.main as module

    calls = []
    understanding = object()
    service = SimpleNamespace(understanding=understanding)

    @contextmanager
    def routed():
        calls.append("open")
        try:
            yield service
        finally:
            calls.append("close")

    def forbidden(*args, **kwargs):
        pytest.fail("activation must not construct legacy or knowledge runtimes")

    monkeypatch.setenv("UNDERSTANDING_ENABLED", "false")
    monkeypatch.setenv("KNOWLEDGE_CHAT_ENABLED", "true")
    monkeypatch.setattr(module, "routed_conversation_runtime", routed, raising=False)
    for name in (
        "conversation_runtime",
        "knowledge_chat_runtime",
        "understanding_runtime",
    ):
        monkeypatch.setattr(module, name, forbidden, raising=False)
    with TestClient(module.app) as client:
        assert client.get("/health").status_code == 200
        assert module.app.state.conversation_runtime is service
        assert module.app.state.understanding_runtime is understanding
        assert module.app.state.knowledge_chat_runtime is None
    assert calls == ["open", "close"]
    assert module.app.state.conversation_runtime is None
    assert module.app.state.understanding_runtime is None


def test_main_does_not_fall_back_when_required_routing_is_unavailable(monkeypatch):
    import resolv.main as module
    from resolv.schemas.conversation import ConversationError

    @contextmanager
    def unavailable():
        raise ConversationError("legacy_run_requires_resolution", False)
        yield  # pragma: no cover

    monkeypatch.setenv("KNOWLEDGE_CHAT_ENABLED", "false")
    monkeypatch.setattr(
        module, "routed_conversation_runtime", unavailable, raising=False
    )
    with (
        pytest.raises(ConversationError, match="legacy_run_requires_resolution"),
        TestClient(module.app),
    ):
        pass
