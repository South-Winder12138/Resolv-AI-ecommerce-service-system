"""Guard infrastructure itself stays offline, including process/file boundaries."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def test_bounded_retest_accepts_a_recorded_zero_provider_startup_failure(
    tmp_path, monkeypatch
):
    from scripts import refund_gate_support as support

    monkeypatch.setattr(support, "ROOT", tmp_path)
    w = tmp_path / support.WORKSPACE
    w.mkdir(parents=True)
    p = w / "resource-ownership.json"
    support.save_json(
        p, support.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    )
    support.reserve_refund_case_attempt(
        p, case_id="F1", attempt=1, source_hash="c" * 64
    )
    support.authorize_case_retest(p, case_id="F1", source_hash="d" * 64)
    assert support.validate_manifest(p)["retests"] == {"F1": "d" * 64}
    with pytest.raises(ValueError):
        support.authorize_case_retest(p, case_id="F1", source_hash="e" * 64)


def test_owned_resource_descriptor_exposes_all_three_bound_factories(tmp_path):
    from scripts.refund_gate_support import RefundOwnedResources

    from resolv.db.engine import create_db_engine

    engines = [
        create_db_engine("sqlite:///" + str(tmp_path / (name + ".db")))
        for name in ("knowledge", "conversation", "business")
    ]
    try:
        resources = RefundOwnedResources(tmp_path / "manifest", {}, *engines)
        assert hasattr(resources, "knowledge_factory"), (
            "missing knowledge factory producer contract"
        )
        assert resources.knowledge_factory.kw["bind"] is engines[0]
        assert resources.conversation_factory.kw["bind"] is engines[1]
        assert resources.business_factory.kw["bind"] is engines[2]
    finally:
        for engine in engines:
            engine.dispose()


def test_preservation_normalizes_actual_protobuf_sequences_and_numeric_vectors():
    import numpy as np
    from google.protobuf.descriptor_pb2 import FileDescriptorProto
    from scripts import refund_gate_support as support

    from resolv.services.hybrid_fingerprints import canonical_hash

    assert hasattr(support, "preservation_value"), "missing Milvus metadata codec"
    repeated = FileDescriptorProto(dependency=["text"]).dependency
    raw = {
        "function": {"input_field_names": repeated},
        "dense": np.array([0.25, 0.5], dtype=np.float32),
    }
    normalized = support.preservation_value(raw)
    assert normalized == {
        "function": {"input_field_names": ["text"]},
        "dense": [0.25, 0.5],
    }
    original_hash = canonical_hash(normalized)
    repeated.append("changed")
    assert canonical_hash(support.preservation_value(raw)) != original_hash
    with pytest.raises((TypeError, ValueError)):
        support.preservation_value({"unexpected": object()})


def module():
    assert importlib.util.find_spec("scripts.refund_gate_support"), (
        "missing owned refund gate guards"
    )
    return importlib.import_module("scripts.refund_gate_support")


@pytest.fixture
def owned(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    w = tmp_path / m.WORKSPACE
    w.mkdir(parents=True)
    value = m.new_manifest(w, owner="a" * 32, server_fingerprint="b" * 64)
    manifest = w / "resource-ownership.json"
    m.save_json(manifest, value)
    return m, manifest


def test_owned_resource_and_atomic_budget_guards(owned):
    m, p = owned

    def reserve(_):
        try:
            return m.reserve_refund_call(
                p, purpose="refund_query", case_id="R1", attempt=1, source_hash="c" * 64
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        reservations = list(pool.map(reserve, range(16)))
    assert len([r for r in reservations if r]) == 1
    calls = m.budget_ledger(p)
    assert len(calls) == 1 and calls[0]["outcome"] == "reserved"
    reservation = next(r for r in reservations if r)
    m.finish_refund_call(reservation, outcome="unknown", error_code="transport_timeout")
    assert m.budget_ledger(p)[0]["outcome"] == "unknown"
    with pytest.raises(ValueError):
        m.reserve_refund_call(
            p, purpose="refund_query", case_id="R1", attempt=2, source_hash="c" * 64
        )
    with pytest.raises(ValueError):
        m.finish_refund_call(
            reservation, outcome="succeeded", input_tokens=1, output_tokens=1
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("kind", "foreign"),
        ("conversation_database", "formal"),
        ("business_db", "/tmp/other.db"),
        ("server_fingerprint", "not-a-hash"),
        ("owner", "../other"),
    ],
)
def test_foreign_manifest_rejected_before_external_io(owned, key, value):
    m, p = owned
    data = json.loads(p.read_text())
    data[key] = value
    m.save_json(p, data)
    before = p.read_bytes()
    with pytest.raises(ValueError), m.owned_refund_resources(p):
        pytest.fail("foreign resources opened")
    assert p.read_bytes() == before


def test_wrong_file_identity_and_server_rejected(owned, monkeypatch):
    m, p = owned
    monkeypatch.setattr(
        m,
        "configured_urls",
        lambda: (_ for _ in ()).throw(
            AssertionError("must check file before external IO")
        ),
    )
    with pytest.raises(ValueError, match="business"), m.owned_refund_resources(p):
        pytest.fail("missing file accepted")
    from resolv.db.engine import create_db_engine

    obj = json.loads(p.read_text())
    f = Path(obj["business_db"])
    engine = create_db_engine("sqlite:///" + str(f))
    m.initialize_business_ownership(engine, obj)
    engine.dispose()
    obj["business_identity"] = m.file_identity(f)
    m.save_json(p, obj)
    monkeypatch.setattr(m, "configured_urls", lambda: ("k", "c"))
    monkeypatch.setattr(
        m,
        "open_owned_database",
        lambda *a: (_ for _ in ()).throw(ValueError("server changed")),
    )
    with pytest.raises(ValueError, match="server"), m.owned_refund_resources(p):
        pytest.fail("wrong server accepted")
    obj["business_identity"]["inode"] += 1
    m.save_json(p, obj)
    with pytest.raises(ValueError, match="business"), m.owned_refund_resources(p):
        pytest.fail("replaced file accepted")


def test_scoped_admin_sql_rejects_external_accounts_and_existing_names(owned):
    m, p = owned
    obj = m.validate_manifest(p)
    sql = m.bootstrap_sql(obj, ["resolv@%"])
    assert "CREATE USER" not in sql and "ON *.*" not in sql
    assert "mhrefund\\_c\\_" in sql and "mhrefund\\_k\\_" in sql
    for bad in ["root@localhost;DROP", "new'user@%"]:
        with pytest.raises(ValueError):
            m.bootstrap_sql(obj, [bad])


def test_r2_attempts_and_purpose_caps_cannot_expand(owned):
    m, p = owned
    for case in range(1, 9):
        for purpose in ["resolution", "intents", "refund_query"]:
            m.reserve_refund_call(
                p, purpose=purpose, case_id=f"F{case}", attempt=1, source_hash="c" * 64
            )
        for _ in range(6):
            m.reserve_refund_call(
                p,
                purpose="refund_decision",
                case_id=f"F{case}",
                attempt=1,
                source_hash="c" * 64,
            )
        with pytest.raises(ValueError):
            m.reserve_refund_call(
                p,
                purpose="refund_decision",
                case_id=f"F{case}",
                attempt=1,
                source_hash="c" * 64,
            )
    with pytest.raises(ValueError):
        m.reserve_refund_call(
            p, purpose="resolution", case_id="F9", attempt=1, source_hash="c" * 64
        )
    with pytest.raises(ValueError):
        m.reserve_refund_call(
            p, purpose="resolution", case_id="F1", attempt=2, source_hash="c" * 64
        )
    m.authorize_case_retest(p, case_id="F1", source_hash="d" * 64)
    m.reserve_refund_call(
        p, purpose="resolution", case_id="F1", attempt=2, source_hash="d" * 64
    )
    with pytest.raises(ValueError):
        m.authorize_case_retest(p, case_id="F1", source_hash="e" * 64)


def test_report_directory_must_be_new_and_task_owned(owned):
    m, p = owned
    target = m.new_report_dir(p, p.parent / "R1")
    assert target.is_dir()
    for path in (target, p.parent, p.parent.parent / "elsewhere"):
        with pytest.raises(ValueError):
            m.new_report_dir(p, path)


def test_one_attempt_cannot_mix_source_candidates(owned):
    m, p = owned
    m.reserve_refund_call(
        p, purpose="resolution", case_id="F1", attempt=1, source_hash="c" * 64
    )
    with pytest.raises(ValueError):
        m.reserve_refund_call(
            p, purpose="intents", case_id="F1", attempt=1, source_hash="d" * 64
        )
    assert len(m.budget_ledger(p)) == 1
