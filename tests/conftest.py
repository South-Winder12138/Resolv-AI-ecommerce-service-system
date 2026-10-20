import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

import resolv.tools.logistics as logistics_module
import resolv.tools.order as order_module
import resolv.tools.refund as refund_module
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from tests.conversation_fixtures import conversation_factory

__all__ = ["conversation_factory"]

pytest_plugins = ["tests.understanding_fixtures"]


def pytest_configure(config):
    # PyMilvus calls load_dotenv() during collection. Never let that library
    # promote a developer's local credentials/budgets into ordinary tests.
    # Explicit Settings env-file tests still use dotenv_values as normal.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("MH_HYBRID_SNAPSHOT_SOURCE"):
        # The committed-export sitecustomize already blocks and audits network
        # before collection and is inherited unchanged by Python children.
        return
    # Install before collection. Python children inherit the same no-network
    # boundary via sitecustomize, without changing the developer's environment.
    guard = '''
import socket
def denied(*args, **kwargs):
    raise RuntimeError("ordinary pytest network blocked")
for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "getnameinfo", "gethostbyaddr"):
    setattr(socket, name, denied)
for name in ("connect", "connect_ex"):
    original = getattr(socket.socket, name)
    def guarded(self, *args, _original=original, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            return denied()
        return _original(self, *args, **kwargs)
    setattr(socket.socket, name, guarded)
'''
    config._refund_network_guard = tempfile.TemporaryDirectory(prefix="resolv-offline-")
    directory = config._refund_network_guard.name
    Path(directory, "sitecustomize.py").write_text(guard)
    os.environ["PYTHONPATH"] = directory + os.pathsep + os.environ.get("PYTHONPATH", "")
    exec(compile(guard, "<pytest network boundary>", "exec"), {})  # noqa: S102 -- fixed local test guard, no input


@pytest.fixture(autouse=True)
def disable_external_knowledge_chat(monkeypatch, tmp_path):
    # Local developer .env may enable RAG; ordinary tests must remain offline.
    monkeypatch.setenv("KNOWLEDGE_CHAT_ENABLED", "false")
    monkeypatch.setenv("UNDERSTANDING_ENABLED", "false")
    from resolv.services import knowledge_branch_runtime
    monkeypatch.setattr(knowledge_branch_runtime, "DEFAULT_RELEASE_SELECTOR", tmp_path / "absent-knowledge-selector.json")


@pytest.fixture
def tool_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> sessionmaker[Session]:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'tools.db'}")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        seed_data(session)
    monkeypatch.setattr(order_module, "get_session_factory", lambda _url: factory)
    monkeypatch.setattr(logistics_module, "get_session_factory", lambda _url: factory)
    monkeypatch.setattr(refund_module, "get_session_factory", lambda _url: factory)
    import resolv.tools.cancel as cancel_module

    monkeypatch.setattr(cancel_module, "get_session_factory", lambda _url: factory)
    yield factory
    engine.dispose()


@pytest.fixture
def api_runtime(conversation_factory, tool_database, monkeypatch):
    from contextlib import contextmanager

    import resolv.main as main_module
    from resolv.core.config import ContextSettings
    from resolv.services.context import ContextBuilder
    from resolv.services.conversation import ConversationService
    from resolv.services.tool_execution import WriteExecutionService
    from tests.conversation_fixtures import MemoryCheckpoints
    from tests.test_context import Counter
    from tests.test_conversation_graph import Model

    service = ConversationService(
        conversation_factory,
        MemoryCheckpoints(),
        context_builder=ContextBuilder(
            ContextSettings(
                _env_file=None,
                model_context_window_tokens=16000,
                tokenizer_profile="cl100k_base",
            ),
            Counter(),
        ),
        model=Model(),
        write_service=WriteExecutionService(conversation_factory, tool_database),
    )
    # During API migration, even the old route must remain offline in RED runs.
    from langchain_core.messages import HumanMessage

    import resolv.api.chat as chat_api
    from resolv import agent_loop

    monkeypatch.setattr(agent_loop, "get_chat_model", lambda: service.model)
    monkeypatch.setattr(
        chat_api,
        "stream_chat",
        lambda question, history=(): service.model.stream(
            [*history, HumanMessage(content=question)]
        ),
        raising=False,
    )

    @contextmanager
    def owner(*, knowledge=None):
        if knowledge is not None:
            service.knowledge = knowledge
        yield service

    monkeypatch.setattr(main_module, "conversation_runtime", owner, raising=False)
    monkeypatch.setattr(main_module, "routed_conversation_runtime", owner, raising=False)
    monkeypatch.setattr(
        main_module.app.state, "conversation_runtime", service, raising=False
    )
    return service


@pytest.fixture
def client(api_runtime):
    from fastapi.testclient import TestClient

    from resolv.main import app

    with TestClient(app) as resource:
        yield resource
