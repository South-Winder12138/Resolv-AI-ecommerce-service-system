"""Only explicit task-owned private storage may retain a failed public response."""

import importlib
import json

import pytest
from scripts import refund_gate_support as s


def recorder():
    assert importlib.util.find_spec("scripts.refund_failure_storage"), (
        "missing explicit private diagnostics sink"
    )
    return importlib.import_module(
        "scripts.refund_failure_storage"
    ).OwnedRefundDiagnostics


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    w = tmp_path / s.WORKSPACE
    w.mkdir(parents=True)
    manifest = w / "resource-ownership.json"
    obj = s.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    obj["state"] = "ready"
    s.save_json(manifest, obj)
    target = w / "diagnostic-case"
    target.mkdir()
    return manifest, target


def diagnostic():
    return {
        "invocation_id": "c" * 32,
        "stage": "json",
        "exception_type": "JSONDecodeError",
        "error_code": "refund_model_output_invalid",
        "input_hash": "d" * 64,
        "response_hash": "e" * 64,
        "model_step": 5,
        "issues": [],
        "json_error": {"position": 2},
        "response": {"content": "PRIVATE_PUBLIC_PAYLOAD"},
    }


def test_private_file_is_exclusive_and_index_omits_response(owned):
    manifest, target = owned
    sink = recorder()(manifest, target)
    row = diagnostic()
    sink(row)
    files = list((target / "failed-responses").glob("*.json"))
    assert len(files) == 1 and files[0].stat().st_mode & 0o777 == 0o600
    assert files[0].parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(files[0].read_text()) == row
    assert "PRIVATE_PUBLIC_PAYLOAD" not in json.dumps(sink.entries)
    before = files[0].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        sink(dict(row, stage="changed"))
    assert files[0].read_bytes() == before


def test_foreign_or_symlink_target_and_path_injection_are_refused(owned):
    manifest, target = owned
    cls = recorder()
    with pytest.raises(ValueError):
        cls(manifest, target.parents[2])
    link = target.parent / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        cls(manifest, link)
    sink = cls(manifest, target)
    with pytest.raises(ValueError):
        sink(dict(diagnostic(), invocation_id="../../foreign"))
    assert not list(target.rglob("*.json"))


def test_oversized_payload_is_refused_before_writing(owned):
    manifest, target = owned
    sink = recorder()(manifest, target)
    with pytest.raises(ValueError):
        sink(dict(diagnostic(), response={"content": "x" * 1_048_577}))
    assert not list(target.rglob("*.json"))


def test_dotdot_and_replaced_report_directory_cannot_escape_owned_scope(owned):
    manifest, target = owned
    cls = recorder()
    outside = target.parent.parent / "foreign"
    outside.mkdir()
    with pytest.raises(ValueError):
        cls(manifest, target / ".." / ".." / "foreign")
    sink = cls(manifest, target)
    target.rmdir()
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        sink(diagnostic())
    assert list(outside.iterdir()) == []
