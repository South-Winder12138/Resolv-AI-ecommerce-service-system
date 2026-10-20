"""Exercise the staging CLI with real files/SQL and a fake network boundary."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import event, select

from resolv.db.engine import create_db_engine
from resolv.db.knowledge_base import KnowledgeBase
from resolv.models.knowledge import KnowledgeChunkRecord

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
URI = "mysql+pymysql://test:private%40password@127.0.0.1:3307/knowledge?charset=utf8mb4"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage_knowledge.py"


@pytest.fixture
def files(tmp_path):
    path = tmp_path / "refund.md"
    path.write_text("# 售后政策\n## 软件商品退款\n软件商品一旦激活，不支持无理由退款。", encoding="utf-8")
    config = tmp_path / "files.toml"
    config.write_text(
        '[files."refund.md"]\ncontent_type="refund_policy"\n'
        'key_clause_keywords=["一旦激活"]\n', encoding="utf-8",
    )
    return path, config


@pytest.fixture
def database(tmp_path, monkeypatch):
    knowledge_db = importlib.import_module("resolv.db.knowledge")
    uri = f"sqlite:///{tmp_path / 'command.db'}"
    reader = create_db_engine(uri)
    KnowledgeBase.metadata.create_all(reader)
    disposed = []
    def network_boundary(_uri, **_options):
        engine = create_db_engine(uri)
        event.listen(engine, "engine_disposed", lambda _engine: disposed.append(True))
        return engine
    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", URI)
    monkeypatch.setattr(knowledge_db, "create_db_engine", network_boundary)
    yield reader, disposed
    reader.dispose()


def _run(files, root, extra=()):
    module = importlib.import_module("scripts.stage_knowledge")
    path, config = files
    module.main([
        str(path), "--config", str(config), "--source-root", str(root),
        "--model-revision", REVISION, *extra,
    ])


def test_cli_created_then_unchanged_has_observable_json_and_closed_resources(files, database, tmp_path, capsys):
    _run(files, tmp_path)
    first = json.loads(capsys.readouterr().out)
    _run(files, tmp_path)
    second = json.loads(capsys.readouterr().out)
    assert (first["outcome"], second["outcome"]) == ("created", "unchanged")
    assert first["document_id"] == second["document_id"]
    assert (first["chunk_count"], first["pending_count"], first["done_count"]) == (1, 1, 0)
    assert first["source"] == "refund.md" and first["content_type"] == "refund_policy"
    assert first["inference_performed"] is False
    assert first["index_profile"]["model_revision"] == REVISION
    assert first["index_profile"]["dimension"] == 1024
    assert first["index_profile"]["normalize_embeddings"] is True
    assert first["index_profile"]["collection_name"] == "resolv_knowledge_v2"
    assert database[1] == [True, True]


@pytest.mark.parametrize("case", [
    "missing mapping", "bad TOML", "unknown metadata", "empty type", "bad UTF8",
    "missing document", "missing config", "outside root", "revision", "collection",
])
def test_invalid_input_creates_no_engine_or_success_output(files, tmp_path, monkeypatch, capsys, case):
    path, config = files
    extra = []
    if case == "missing mapping":
        config.write_text("[files]\n", encoding="utf-8")
    elif case == "bad TOML":
        config.write_text("[invalid TOML", encoding="utf-8")
    elif case == "unknown metadata":
        config.write_text('[files."refund.md"]\ncontent_type="private_value"\nunknown=1', encoding="utf-8")
    elif case == "empty type":
        config.write_text('[files."refund.md"]\ncontent_type=""', encoding="utf-8")
    elif case == "bad UTF8":
        path.write_bytes(b"\xff\xfe")
    elif case == "missing document":
        path.unlink()
    elif case == "missing config":
        config.unlink()
    elif case == "outside root":
        files = (tmp_path.parent / "outside.md", config)
    elif case == "revision":
        extra = ["--model-revision", "main"]
    else:
        extra = ["--collection", "private/collection"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Bad input must not create resource")
    monkeypatch.setattr(importlib.import_module("resolv.db.knowledge"), "create_db_engine", forbidden)
    with pytest.raises(SystemExit) as caught:
        _run(files, tmp_path, extra)
    assert "private" not in str(caught.value)
    assert capsys.readouterr().out == ""


def test_invalid_database_settings_do_not_print_credentials(files, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KNOWLEDGE_DATABASE_URL", "mysql://private:secret@example/db")
    with pytest.raises(SystemExit) as caught:
        _run(files, tmp_path)
    assert "secret" not in str(caught.value) and "example" not in str(caught.value)
    assert capsys.readouterr().out == ""


def test_database_failure_has_no_success_json_and_engine_disposes(files, database, tmp_path, capsys):
    with database[0].begin() as connection:
        KnowledgeBase.metadata.drop_all(connection)
    with pytest.raises(SystemExit) as caught:
        _run(files, tmp_path)
    assert "private" not in str(caught.value)
    assert database[1] == [True]
    assert capsys.readouterr().out == ""


def test_engine_close_failure_does_not_print_success(files, database, tmp_path, monkeypatch, capsys):
    def fail(engine):
        raise RuntimeError("private disposal fault")
    boundary = importlib.import_module("resolv.db.knowledge")
    original = boundary.create_db_engine
    def create(*args, **kwargs):
        engine = original(*args, **kwargs)
        event.listen(engine, "engine_disposed", fail)
        return engine
    monkeypatch.setattr(boundary, "create_db_engine", create)
    with pytest.raises(SystemExit) as caught:
        _run(files, tmp_path)
    assert "private" not in str(caught.value)
    assert capsys.readouterr().out == ""


def test_changed_file_rejected_and_database_content_preserved(files, database, tmp_path, capsys):
    _run(files, tmp_path)
    capsys.readouterr()
    files[0].write_text("# 售后政策\n## 软件商品退款\n允许退款。", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(files, tmp_path)
    assert capsys.readouterr().out == ""
    with database[0].connect() as connection:
        assert connection.scalar(select(KnowledgeChunkRecord.answer)) == "软件商品一旦激活，不支持无理由退款。"


def test_import_is_inert_and_does_not_load_model_or_require_settings(tmp_path):
    code = """
import runpy, sys
from resolv.db import knowledge
def forbidden(*args, **kwargs):
    raise AssertionError("Import must create no resource/settings")
knowledge.create_knowledge_engine = forbidden
from resolv.core import config
config.KnowledgeDatabaseSettings = forbidden
runpy.run_path(sys.argv[1], run_name="stage_import_guard")
assert not {"torch", "FlagEmbedding", "pymilvus", "langchain_openai"} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_actual_cli_entrypoint_needs_only_knowledge_config_with_no_model_or_network(files, tmp_path):
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "MILVUS_", "BGE_", "EMBEDDING_"))
        and key != "DATABASE_URL"
    }
    environment.update({
        "KNOWLEDGE_DATABASE_URL": URI, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    code = """
import runpy, socket, sys
from pathlib import Path
from sqlalchemy import event
from resolv.db.engine import create_db_engine
from resolv.db.knowledge_base import KnowledgeBase
from resolv.models import knowledge as models
from resolv.db import knowledge as database
script, path, config, root, revision = sys.argv[1:]
uri = "sqlite:///" + str(Path(root) / "actual-command.db")
with create_db_engine(uri).begin() as connection:
    KnowledgeBase.metadata.create_all(connection)
disposed = []
def boundary(_uri, **kwargs):
    engine = create_db_engine(uri)
    event.listen(engine, "engine_disposed", lambda engine: disposed.append(True))
    return engine
def forbidden(*args, **kwargs):
    raise AssertionError("No network during ordinary tests")
database.create_db_engine = boundary
socket.socket.connect = forbidden
for _ in range(2):
    sys.argv = [script, path, "--config", config, "--source-root", root, "--model-revision", revision]
    runpy.run_path(script, run_name="__main__")
assert disposed == [True, True]
assert not {"torch", "FlagEmbedding", "pymilvus", "langchain_openai"} & sys.modules.keys()
assert not (Path(root) / "data").exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), *map(str, files), str(tmp_path), REVISION],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["outcome"] for line in result.stdout.splitlines()] == ["created", "unchanged"]
