import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from resolv.core.config import KnowledgeDatabaseSettings
from resolv.db import engine as engine_module

URI = "mysql+pymysql://knowledge:private_fake_password@localhost:3307/resolv_knowledge?charset=utf8mb4"


@pytest.fixture
def mysql_smoke(monkeypatch):
    module = importlib.import_module("scripts.smoke_mysql")
    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", URI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(module, "KnowledgeDatabaseSettings", lambda: KnowledgeDatabaseSettings(_env_file=None))
    events = []
    state = SimpleNamespace(stage=None, result=1, close_error=None, dispose_error=None,
                            failure=SQLAlchemyError("private credentials from driver"))

    class FakeConnection:
        def scalar(self, statement):
            events.append(("query", str(statement)))
            if state.stage == "query":
                raise state.failure
            return state.result

        def close(self):
            events.append("connection_close")
            if state.close_error:
                raise state.close_error

    class FakeEngine:
        dialect = SimpleNamespace(name="mysql")

        def connect(self):
            events.append("connect")
            if state.stage == "connect":
                raise state.failure
            return FakeConnection()

        def dispose(self):
            events.append("dispose")
            if state.dispose_error:
                raise state.dispose_error

    def factory(url, **kwargs):
        events.append("create")
        if state.stage == "create":
            raise state.failure
        return FakeEngine()

    monkeypatch.setattr(engine_module, "create_engine", factory)
    return module, events, state


def test_smoke_only_selects_one_and_closes_before_success(mysql_smoke, capsys):
    module, events, _ = mysql_smoke
    module.main()
    assert events == ["create", "connect", ("query", "SELECT 1"), "connection_close", "dispose"]
    output = capsys.readouterr().out
    assert "Knowledge MySQL SELECT 1: 1" in output
    assert "MySQL smoke passed; resources closed." in output
    assert "private" not in output
    assert "mysql+pymysql" not in output


@pytest.mark.parametrize("value", [None, 0, "1", True])
def test_invalid_scalar_never_claims_success(mysql_smoke, capsys, value):
    module, events, state = mysql_smoke
    state.result = value
    with pytest.raises(SystemExit) as captured:
        module.main()
    assert captured.value.code
    assert events[-2:] == ["connection_close", "dispose"]
    assert "passed" not in capsys.readouterr().out


@pytest.mark.parametrize("missing", [False, True])
def test_config_fails_before_resources_and_hides_url(mysql_smoke, monkeypatch, missing):
    module, events, _ = mysql_smoke
    if missing:
        monkeypatch.delenv("KNOWLEDGE_DATABASE_URL")
    else:
        monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", "sqlite:///private_password.db")
    with pytest.raises(SystemExit) as captured:
        module.main()
    assert "private" not in str(captured.value)
    assert events == []


@pytest.mark.parametrize("stage", ["create", "connect", "query"])
def test_database_failures_are_safe_and_cleanup_owned_resources(mysql_smoke, capsys, stage):
    module, events, state = mysql_smoke
    state.stage = stage
    with pytest.raises(SystemExit) as captured:
        module.main()
    assert captured.value.code
    assert "private" not in str(captured.value)
    assert "passed" not in capsys.readouterr().out
    if stage != "create":
        assert events[-1] == "dispose"
    if stage == "query":
        assert events[-2:] == ["connection_close", "dispose"]


@pytest.mark.parametrize("resource", ["close_error", "dispose_error"])
def test_cleanup_failure_is_not_success(mysql_smoke, capsys, resource):
    module, events, state = mysql_smoke
    setattr(state, resource, RuntimeError("private cleanup"))
    with pytest.raises(SystemExit) as captured:
        module.main()
    assert "private" not in str(captured.value)
    assert "passed" not in capsys.readouterr().out
    assert events[-2:] == ["connection_close", "dispose"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_primary_error_survives_both_cleanup_failures(mysql_smoke, capsys, interrupt):
    module, events, state = mysql_smoke
    state.stage = "query"
    state.failure = KeyboardInterrupt() if interrupt else SQLAlchemyError("private query")
    state.close_error = RuntimeError("private close")
    state.dispose_error = RuntimeError("private dispose")
    with pytest.raises(KeyboardInterrupt if interrupt else SystemExit) as captured:
        module.main()
    primary = captured.value if interrupt else captured.value.__cause__
    assert primary is state.failure
    assert len(primary.__notes__) == 2
    assert all("private" not in note for note in primary.__notes__)
    assert events[-2:] == ["connection_close", "dispose"]
    assert "passed" not in capsys.readouterr().out


def test_smoke_import_is_resource_free():
    result = subprocess.run([sys.executable, "-c", '''
import sys
import sqlalchemy
import pydantic_settings
def forbidden(*args, **kwargs):
    raise AssertionError("import initialized resource")
sqlalchemy.create_engine = forbidden
pydantic_settings.BaseSettings.__init__ = forbidden
import scripts.smoke_mysql
assert "FlagEmbedding" not in sys.modules
assert "torch" not in sys.modules
'''], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("option", [
    "port=bad", "database=other", "host=other", "user=root",
    "password=private_override", "init_command=CREATE%20TABLE%20unexpected(id%20INT)",
    "connect_timeout=999", "ssl_ca=unknown",
])
def test_url_driver_options_fail_before_any_engine_or_connection(
    mysql_smoke, monkeypatch, option
):
    module, events, _ = mysql_smoke
    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", URI + "&" + option)
    with pytest.raises(SystemExit) as captured:
        module.main()
    assert events == []
    assert "private" not in str(captured.value)
