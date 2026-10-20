"""Execute the real preview pipeline with local files and no service resources."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preview_knowledge.py"


@pytest.fixture
def files(tmp_path):
    path = tmp_path / "refund.md"
    path.write_text("# 售后政策\n\n## 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。", encoding="utf-8")
    config = tmp_path / "files.toml"
    config.write_text(
        '[files."./refund.md"]\ncontent_type = "refund_policy"\n'
        'key_clause_keywords = ["一旦激活", "不支持"]\n',
        encoding="utf-8",
    )
    return path, config


def _run(files, tmp_path, extra=()):
    module = importlib.import_module("scripts.preview_knowledge")
    path, config = files
    module.main([str(path), "--config", str(config), "--source-root", str(tmp_path), *extra])


def test_preview_outputs_complete_structured_chunk_json(files, tmp_path, capsys):
    _run(files, tmp_path)
    result = json.loads(capsys.readouterr().out)
    assert result["source"] == "refund.md"
    assert result["content_type"] == "refund_policy"
    assert (result["max_chars"], result["overlap"], result["chunk_count"]) == (800, 100, 1)
    chunk = result["chunks"][0]
    assert chunk["category"] == "售后政策"
    assert chunk["questions"] == "软件商品退款"
    assert chunk["answer"] == "软件商品一旦激活，不支持无理由退款。"
    assert chunk["section_path"] == "售后政策 > 软件商品退款"
    assert chunk["text"] == "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。"
    assert chunk["is_key_clause"] == 1
    assert chunk["document_id"] == result["document_id"]
    assert chunk["chunk_index"] == 0


def test_preview_txt_and_custom_windows(files, tmp_path, capsys):
    _, config = files
    txt = tmp_path / "plain.txt"
    txt.write_text("abcdefghijBANklmnopqr", encoding="utf-8")
    config.write_text(
        '[files."plain.txt"]\ncontent_type = "plain"\nkey_clause_keywords = ["BAN"]\n',
        encoding="utf-8",
    )
    _run((txt, config), tmp_path, ["--max-chars", "10", "--overlap", "0"])
    result = json.loads(capsys.readouterr().out)
    assert result["chunk_count"] == 3
    assert [chunk["answer"] for chunk in result["chunks"]] == ["abcdefghij", "BANklmnopq", "r"]
    assert [chunk["is_key_clause"] for chunk in result["chunks"]] == [0, 1, 0]
    assert all(chunk["section_path"] == "" for chunk in result["chunks"])


def test_preview_empty_document_is_success_with_no_chunks(files, tmp_path, capsys):
    files[0].write_text("# Empty\n", encoding="utf-8")
    _run(files, tmp_path)
    result = json.loads(capsys.readouterr().out)
    assert result["chunk_count"] == 0
    assert result["chunks"] == []


@pytest.mark.parametrize("config_text", [
    "[files]\n",
    '[files."refund.md"]\ncontent_type = ""',
    '[files."refund.md"]\ncontent_type = "refund"\nkey_clause_keywords = [""]',
    '[files."refund.md"]\ncontent_type = "refund"\nkey_clause_keywords = "keyword"',
    '[files."../outside.md"]\ncontent_type = "private_value"',
    '[files."refund.md"]\ncontent_type = "private_value"\nunknown = true',
    "[invalid TOML",
])
def test_preview_invalid_configuration_fails_without_success_output(files, tmp_path, capsys, config_text):
    files[1].write_text(config_text, encoding="utf-8")
    with pytest.raises(SystemExit) as captured:
        _run(files, tmp_path)
    assert captured.value.code != 0
    assert "private_value" not in str(captured.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("missing", [0, 1])
def test_preview_missing_document_or_config_fails(files, tmp_path, missing, capsys):
    files[missing].unlink()
    with pytest.raises(SystemExit):
        _run(files, tmp_path)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("extra", [
    ["--max-chars", "0"], ["--overlap", "-1"],
    ["--max-chars", "10", "--overlap", "10"],
])
def test_preview_invalid_chunk_parameters_fail(files, tmp_path, extra, capsys):
    with pytest.raises(SystemExit):
        _run(files, tmp_path, extra)
    assert capsys.readouterr().out == ""


def test_actual_cli_needs_no_service_config_or_external_resource_imports(files, tmp_path):
    path, config = files
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "MYSQL_", "KNOWLEDGE_DATABASE_", "MILVUS_", "BGE_", "EMBEDDING_"))
        and key != "DATABASE_URL"
    }
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    code = """
import runpy, socket, sys
def no_network(*args, **kwargs):
    raise AssertionError("Preview must not connect to a network")
socket.socket.connect = no_network
script, path, config, root = sys.argv[1:]
sys.argv = [script, path, "--config", config, "--source-root", root]
runpy.run_path(script, run_name="__main__")
assert not {"torch", "FlagEmbedding", "pymilvus", "sqlalchemy", "langchain_openai"} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(path), str(config), str(tmp_path)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["chunks"][0]["is_key_clause"] == 1
    assert not (tmp_path / "data").exists()


def test_preview_import_does_not_read_user_files_or_require_config(tmp_path):
    code = """
import pathlib, runpy, sys
original_open = pathlib.Path.open
original_read_text = pathlib.Path.read_text
def guard(method):
    def guarded(path, *args, **kwargs):
        # Pydantic discovers plugins through installed package metadata.
        if path.name == "entry_points.txt" and path.parent.name.endswith(".dist-info"):
            return method(path, *args, **kwargs)
        raise AssertionError("Import must not read a source/config")
    return guarded
pathlib.Path.open = guard(original_open)
pathlib.Path.read_text = guard(original_read_text)
runpy.run_path(sys.argv[1], run_name="preview_import_guard")
assert not {"torch", "FlagEmbedding", "pymilvus", "sqlalchemy"} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["refund.md", "REFUND.MD"])
def test_preview_rejects_case_aliases_before_choosing_type(files, tmp_path, capsys, name):
    original, config = files
    alias = tmp_path / "REFUND.MD"
    if not alias.exists() or not original.samefile(alias):
        pytest.skip("Requires a case-insensitive filesystem")
    config.write_text('''
[files."refund.md"]
content_type = "first"
[files."REFUND.MD"]
content_type = "second"
''', encoding="utf-8")
    with pytest.raises(SystemExit):
        _run((tmp_path / name, config), tmp_path)
    assert capsys.readouterr().out == ""
