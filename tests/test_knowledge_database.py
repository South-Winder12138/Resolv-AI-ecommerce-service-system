import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from resolv.core import config
from resolv.db import engine as engine_module

URI = "mysql+pymysql://knowledge:fake%40password@localhost:3307/resolv_knowledge?charset=utf8mb4"


def boundary(monkeypatch, *, creation_error=None, disposal_error=None):
    module = importlib.import_module("resolv.db.knowledge")
    events = []

    class FakeEngine:
        dialect = SimpleNamespace(name="mysql")

        def dispose(self):
            events.append("dispose")
            if disposal_error:
                raise disposal_error

    engine = FakeEngine()

    def fake_driver_factory(url, **kwargs):
        events.append((url, kwargs))
        if creation_error:
            raise creation_error
        return engine

    monkeypatch.setattr(engine_module, "create_engine", fake_driver_factory)
    settings = config.KnowledgeDatabaseSettings(
        knowledge_database_url=URI, _env_file=None
    )
    return module, settings, engine, events


def test_knowledge_engine_uses_common_factory_with_safe_bounded_options(monkeypatch):
    module, settings, engine, events = boundary(monkeypatch)
    assert module.create_knowledge_engine(settings) is engine
    assert events == [(URI, {
        "pool_pre_ping": True,
        "hide_parameters": True,
        "connect_args": {"connect_timeout": 10},
    })]


def test_sqlite_foreign_keys_still_work_with_optional_engine_settings(tmp_path):
    engine = engine_module.create_db_engine(
        f"sqlite:///{tmp_path / 'business.db'}", pool_pre_ping=True,
        hide_parameters=True,
    )
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("PRAGMA foreign_keys")) == 1
    finally:
        engine.dispose()


def test_engine_creation_failure_is_safe_and_keeps_cause(monkeypatch):
    error = RuntimeError("private credentials")
    module, settings, _, events = boundary(monkeypatch, creation_error=error)
    with pytest.raises(module.KnowledgeStoreError) as captured:
        module.create_knowledge_engine(settings)
    assert captured.value.__cause__ is error
    assert "private" not in str(captured.value)
    assert "dispose" not in events


def test_engine_context_disposes_after_success(monkeypatch):
    module, settings, engine, events = boundary(monkeypatch)
    with module.knowledge_engine(settings) as owned:
        assert owned is engine
        events.append("body")
    assert events[-2:] == ["body", "dispose"]


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_body_failure_preserved_and_engine_disposed(monkeypatch, interrupt, cleanup_fails):
    cleanup = RuntimeError("private cleanup") if cleanup_fails else None
    module, settings, _, events = boundary(monkeypatch, disposal_error=cleanup)
    error = KeyboardInterrupt() if interrupt else RuntimeError("body")
    with pytest.raises(type(error)) as captured, module.knowledge_engine(settings):
        raise error
    assert captured.value is error
    assert events[-1] == "dispose"
    if cleanup_fails:
        assert len(error.__notes__) == 1
        assert "private" not in error.__notes__[0]


def test_disposal_failure_is_not_success(monkeypatch):
    error = RuntimeError("private cleanup")
    module, settings, _, events = boundary(monkeypatch, disposal_error=error)
    with pytest.raises(module.KnowledgeStoreError) as captured, module.knowledge_engine(settings):
        pass
    assert captured.value.__cause__ is error
    assert "private" not in str(captured.value)
    assert events[-1] == "dispose"


def test_fresh_import_creates_no_settings_engine_or_model():
    result = subprocess.run([sys.executable, "-c", '''
import sys
import sqlalchemy
import pydantic_settings
def forbidden(*args, **kwargs):
    raise AssertionError("import initialized external resource")
sqlalchemy.create_engine = forbidden
pydantic_settings.BaseSettings.__init__ = forbidden
import resolv.db.knowledge
assert "FlagEmbedding" not in sys.modules
assert "torch" not in sys.modules
'''], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
