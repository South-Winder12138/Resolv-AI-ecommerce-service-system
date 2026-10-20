import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from resolv.main import app
from resolv.schemas.knowledge import KnowledgeSearchHit
from resolv.services.knowledge_chat import KnowledgeChatRuntime
from tests.test_api_agent import cancel_call, fake_model
from tests.test_api_conversations import ask, create


def hit():
    return KnowledgeSearchHit(
        chunk_id="a" * 64,
        document_id="b" * 64,
        chunk_index=0,
        source="knowledge/refund.md",
        category="售后政策",
        questions="软件商品退款",
        answer="软件商品一旦激活，不支持无理由退款。",
        section_path="售后政策 > 软件商品退款",
        content_type="refund_policy",
        is_key_clause=1,
        text="售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
        score=0.8,
    )


@pytest.mark.parametrize("path", ["/agent/stream", "/chat/stream", "/agent/chat"])
def test_http_retrieves_before_llm_and_keeps_context_out_of_history(
    client, api_runtime, path
):
    class Retriever:
        def retrieve(self, query, top_k):
            assert query == "激活的软件还能退款吗？" and top_k == 5
            return [hit()]

    api_runtime.knowledge = KnowledgeChatRuntime(Retriever(), 5)
    model = fake_model(api_runtime, [AIMessage(content="激活后不支持无理由退款。")])
    session = create(client)
    response = ask(client, session, question="激活的软件还能退款吗？", path=path)
    assert response.status_code == 200
    assert any("软件商品一旦激活" in str(message.content) for message in model.calls[0])
    if path.endswith("stream"):
        assert (
            "event: knowledge_sources" in response.text
            and "knowledge/refund.md" in response.text
        )
    assert [m.text for m in api_runtime.detail(session, None, 50).messages] == [
        "激活的软件还能退款吗？",
        "激活后不支持无理由退款。",
    ]


@pytest.mark.parametrize("path", ["/agent/stream", "/chat/stream", "/agent/chat"])
def test_retrieval_failure_does_not_call_llm_or_commit_final_turn(
    client, api_runtime, path
):
    class Retriever:
        def retrieve(self, query, top_k):
            raise RuntimeError("private password")

    api_runtime.knowledge = KnowledgeChatRuntime(Retriever(), 5)
    session = create(client)
    response = ask(client, session, question="退款", path=path)
    if path.endswith("stream"):
        assert "event: error" in response.text and "event: done" not in response.text
    else:
        assert response.status_code == 503
    assert "private" not in response.text and not api_runtime.model.calls
    assert len(api_runtime.detail(session, None, 50).messages) == 1


def test_reference_snapshot_survives_write_approval(client, api_runtime):
    queries = []

    class Retriever:
        def retrieve(self, query, top_k):
            queries.append(query)
            return [hit()]

    api_runtime.knowledge = KnowledgeChatRuntime(Retriever(), 5)
    model = fake_model(api_runtime, [cancel_call(), AIMessage(content="已取消订单。")])
    session = create(client)
    first = ask(client, session, question="取消 MH002", path="/agent/chat").json()
    second = client.post(
        "/agent/approvals/" + first["approval_id"],
        json={"session_id": session, "approved": True},
    )
    assert second.json()["status"] == "completed" and queries == ["取消 MH002"]
    assert all(
        any("软件商品一旦激活" in str(m.content) for m in messages)
        for messages in model.calls
    )


def test_empty_hits_are_explicit_context():
    from resolv.services.knowledge_chat import KnowledgeChatRuntime

    class Retriever:
        def retrieve(self, query, top_k):
            return []

    history, hits = KnowledgeChatRuntime(Retriever(), 5).prepare("退款", [])
    assert hits == []
    assert json.loads(history[-1].content.split("\n", 1)[1]) == []
    assert "未召回" in history[-2].content


def test_main_never_constructs_knowledge_runtime_even_when_enabled(monkeypatch, api_runtime):
    import resolv.main as main_module

    def runtime(*args, **kwargs):
        pytest.fail("unconnected knowledge branch constructed a runtime")

    monkeypatch.setattr(main_module, "knowledge_chat_runtime", runtime, raising=False)
    monkeypatch.setenv("KNOWLEDGE_CHAT_ENABLED", "true")
    monkeypatch.setenv("KNOWLEDGE_CHAT_MODEL_REVISION", "a" * 40)
    with TestClient(app):
        assert app.state.conversation_runtime is api_runtime
        assert app.state.knowledge_chat_runtime is None
    assert app.state.knowledge_chat_runtime is None


def test_shared_runtime_serializes_inference():
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    from resolv.services.knowledge_chat import KnowledgeChatRuntime

    active = maximum = 0
    guard = Lock()

    class Retriever:
        def retrieve(self, query, top_k):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with guard:
                active -= 1
            return []

    runtime = KnowledgeChatRuntime(Retriever(), 5)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: runtime.prepare("退款", []), range(8)))
    assert len(results) == 8 and maximum == 1


def test_partial_startup_closes_owned_resources(monkeypatch):
    import resolv.services.knowledge_chat as module
    from resolv.core.config import KnowledgeChatSettings

    events = []

    @contextmanager
    def engine(settings):
        events.append("engine_open")
        try:
            yield object()
        finally:
            events.append("engine_close")

    @contextmanager
    def client(settings):
        events.append("client_open")
        try:
            yield object()
        finally:
            events.append("client_close")

    def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(module, "KnowledgeDatabaseSettings", lambda: object())
    monkeypatch.setattr(module, "MilvusSettings", lambda: object())
    monkeypatch.setattr(module, "knowledge_engine", engine)
    monkeypatch.setattr(module, "milvus_client", client)
    monkeypatch.setattr(module, "pinned_knowledge_embedding", fail)
    with (
        pytest.raises(RuntimeError, match="model unavailable"),
        module.knowledge_chat_runtime(
            KnowledgeChatSettings(_env_file=None, enabled=True, model_revision="a" * 40)
        ),
    ):
        pytest.fail("startup must fail")
    assert events == ["engine_open", "client_open", "client_close", "engine_close"]


def test_enabled_runtime_requires_pinned_revision_before_resources(monkeypatch):
    import resolv.services.knowledge_chat as module
    from resolv.core.config import KnowledgeChatSettings
    from resolv.services.knowledge_chat import knowledge_chat_runtime

    monkeypatch.setattr(
        module,
        "knowledge_engine",
        lambda *_: pytest.fail("Must validate before resources"),
    )
    with (
        pytest.raises(ValueError, match="MODEL_REVISION"),
        knowledge_chat_runtime(
            KnowledgeChatSettings(_env_file=None, enabled=True, model_revision=None)
        ),
    ):
        pytest.fail("revision missing")
