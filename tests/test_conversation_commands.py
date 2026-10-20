import importlib

import pytest


@pytest.mark.parametrize("name", ["resolv_knowledge", "mysql", "resolv_conversations;DROP DATABASE mysql", "resolv_conversations_smoke_", "resolv_conversations_smoke_a-b"])
def test_prepare_rejects_unsafe_database_without_execution(name):
    module = importlib.import_module("scripts.prepare_conversation_database")
    with pytest.raises(SystemExit):
        module.main(["--database", name])


def test_prepare_uses_container_admin_without_host_password(monkeypatch, capsys):
    module = importlib.import_module("scripts.prepare_conversation_database")
    commands = []
    monkeypatch.setenv("MYSQL_ROOT_PASSWORD", "host_private_pw")
    monkeypatch.setattr(module, "_execute", lambda command, **kwargs: commands.append((command, kwargs)))
    module.main(["--database", "resolv_conversations_smoke_test1"])
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command[:6] == ["docker", "compose", "-f", "compose.mysql.yaml", "exec", "-T"]
    assert "host_private_pw" not in str(command) + str(kwargs) + capsys.readouterr().out
    assert "DROP" not in str(command)
    assert "resolv_conversations_smoke_test1" in kwargs["input"]


def test_command_error_does_not_print_credentials(monkeypatch, capsys):
    module = importlib.import_module("scripts.prepare_conversation_database")
    def fail(*args, **kwargs):
        raise RuntimeError("private_pw")
    monkeypatch.setattr(module, "_execute", fail)
    with pytest.raises(SystemExit) as raised:
        module.main(["--database", "resolv_conversations"])
    assert "private_pw" not in str(raised.value) + capsys.readouterr().out


def test_smoke_rejects_production_database(monkeypatch):
    module = importlib.import_module("scripts.smoke_conversation_checkpoints")
    monkeypatch.setenv("TEST_CONVERSATION_URL", "mysql+pymysql://app:private_pw@host/resolv_conversations?charset=utf8mb4")
    with pytest.raises(SystemExit) as raised:
        module.main(["--database-env", "TEST_CONVERSATION_URL", "--setup"])
    assert "private_pw" not in str(raised.value)


def test_smoke_bad_config_and_gate_failure_hide_secret(monkeypatch):
    module = importlib.import_module("scripts.smoke_conversation_checkpoints")
    monkeypatch.setenv("TEST_CONVERSATION_URL", "private_pw invalid")
    with pytest.raises(SystemExit) as raised:
        module.main(["--database-env", "TEST_CONVERSATION_URL"])
    assert "private_pw" not in str(raised.value)


def test_smoke_gate_failure_hides_secret(monkeypatch):
    module = importlib.import_module("scripts.smoke_conversation_checkpoints")
    monkeypatch.setenv("TEST_CONVERSATION_URL", "mysql+pymysql://app:private_pw@host/resolv_conversations_smoke_test?charset=utf8mb4")
    def fail(*args, **kwargs):
        raise RuntimeError("private_pw")
    monkeypatch.setattr(module, "run_gate", fail)
    with pytest.raises(SystemExit) as raised:
        module.main(["--database-env", "TEST_CONVERSATION_URL"])
    assert "private_pw" not in str(raised.value)


def test_gate_graph_reads_root_checkpoint_before_action():
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    module = importlib.import_module("scripts.smoke_conversation_checkpoints")
    graph = module._graph(InMemorySaver())
    config = {"configurable": {"thread_id": "offline-gate"}}
    graph.invoke({"messages": [HumanMessage(content="fake")], "model_steps": 0}, config, durability="sync")
    assert graph.get_state(config).next == ("await_approval",)


def test_recovery_smoke_dispatches_independent_process_phase(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.smoke_conversation_recovery")
    monkeypatch.setenv("TEST_CONVERSATION_URL", "mysql+pymysql://app:private_pw@host/resolv_conversations_smoke_test?charset=utf8mb4")
    calls = []
    monkeypatch.setattr(module, "process_smoke", lambda *args: calls.append(args))
    module.main(["--database-env", "TEST_CONVERSATION_URL", "--business-db", str(tmp_path / "business.db"), "--phase", "pause", "--manifest", str(tmp_path / "run.json")])
    assert calls[0][2:] == ("pause", tmp_path / "run.json", "approve", False)
