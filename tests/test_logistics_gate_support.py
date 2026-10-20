"""Run preparation with a real local DB and an explicit offline admin seam."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from resolv.db.engine import create_db_engine
from tests.test_logistics_contracts import api

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    if importlib.util.find_spec("scripts.logistics_gate_support") is None:
        class MissingImplementation:
            def __getattr__(self, name):
                return api("scripts.logistics_gate_support")

        yield MissingImplementation()
        return
    module = api("scripts.logistics_gate_support")
    workspace = tmp_path / "owned"
    workspace.mkdir(mode=0o700)
    engine = create_db_engine(f"sqlite:///{tmp_path / 'mysql-boundary.db'}")
    monkeypatch.setattr(module, "RESOURCE_ROOT", workspace.resolve())
    monkeypatch.setattr(module, "_allocate_ports", lambda: {"api": 18101, "mcp": 18102, "resolv": 18103})
    monkeypatch.setattr(module, "_administrative_identity", lambda: "e" * 64)

    def prepare_mysql(obj):
        assert obj["conversation_database"] == "resolv_conversations_smoke_logistics_mcp_20261020"
        cfg = Config(str(ROOT / "alembic.conversation.ini"))
        with engine.begin() as connection:
            cfg.attributes["connection"] = connection
            command.upgrade(cfg, "20261020_0006")
            for name in ("checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"):
                connection.exec_driver_sql(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)")

    monkeypatch.setattr(module, "_prepare_mysql", prepare_mysql)
    monkeypatch.setattr(module, "_open_owned_conversation", lambda obj: engine)
    monkeypatch.setattr(module, "_conversation_url", lambda: make_url("mysql+pymysql://offline:offline@127.0.0.1/resolv_conversations_smoke_logistics_mcp_20261020?charset=utf8mb4"))
    value = module.prepare_logistics_demo(workspace, "resolv_conversations_smoke_logistics_mcp_20261020")
    yield SimpleNamespace(module=module, workspace=workspace, manifest=workspace / "resource-ownership.json", identity=value, engine=engine)
    engine.dispose()


def test_prepare_refuses_foreign_existing_resources(tmp_path, monkeypatch):
    module = api("scripts.logistics_gate_support")
    workspace = tmp_path / "owned"
    workspace.mkdir()
    monkeypatch.setattr(module, "RESOURCE_ROOT", workspace.resolve())
    (workspace / "business.db").write_bytes(b"unowned existing data")
    with pytest.raises(ValueError):
        module.prepare_logistics_demo(workspace, "resolv_conversations_smoke_logistics_mcp_20261020")
    assert (workspace / "business.db").read_bytes() == b"unowned existing data"
    assert not (workspace / "resource-ownership.json").exists()


def test_protection_refresh_reenumerates_formal_tables_and_collections(tmp_path, monkeypatch):
    from scripts import logistics_gate_support as module
    from scripts import refund_gate_support

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "implementation-baseline.json").write_text('{"protected":{}}')
    for name in (".env", "pyproject.toml", "uv.lock", "frontend/package-lock.json", "knowledge/active-release.json"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("unchanged")
    monkeypatch.setattr(module, "RESOURCE_ROOT", workspace)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    layouts = []
    business_urls = []

    def snapshot(**kwargs):
        layouts.append(kwargs.get("layout"))
        business_urls.append(kwargs.get("business_database_url"))
        return {"all_current_objects": True}

    monkeypatch.setattr(refund_gate_support, "snapshot_formal_resources", snapshot)
    module.snapshot_logistics_protection(layout={"formal": {"old_objects_only": True}})
    assert layouts == [None]
    assert business_urls == ["sqlite:///" + str(tmp_path / "data/resolv.db")]


def test_explicit_source_rebind_preserves_owned_data_and_unopened_budget(prepared, monkeypatch):
    module = prepared.module
    assert hasattr(module, "rebind_logistics_source"), "missing audited source rebind"
    before = (prepared.workspace / "business.db").read_bytes()
    previous = prepared.identity["source_revision"]
    monkeypatch.setattr(module, "source_revision", lambda: "a" * 40)
    result = module.rebind_logistics_source(prepared.manifest, previous_revision=previous, previous_source_root=prepared.identity["source_root"])
    assert result["source_revision"] == "a" * 40
    assert result["source_rebindings"][-1]["previous_revision"] == previous
    assert (prepared.workspace / "business.db").read_bytes() == before
    assert module._read_json(prepared.workspace / "budget.json")["state"] == "unopened"
    assert module.validate_logistics_ownership(prepared.manifest)["resource_id"] == prepared.identity["resource_id"]


def test_source_rebind_rejects_spent_budget(prepared, monkeypatch):
    module = prepared.module
    assert hasattr(module, "rebind_logistics_source"), "missing audited source rebind"
    module.save_json(prepared.workspace / "budget.json", {"state": "closed", "calls": [], "retrievals": 0})
    previous = prepared.manifest.read_bytes()
    with pytest.raises(ValueError, match="unopened"):
        module.rebind_logistics_source(prepared.manifest, previous_revision=prepared.identity["source_revision"], previous_source_root=prepared.identity["source_root"])
    assert prepared.manifest.read_bytes() == previous


def test_explicit_preparation_owns_only_three_orders_and_mapping_placeholders(prepared):
    import sqlite3

    h = prepared
    assert h.identity["owner_id"] == "resolv-logistics-mcp-v1"
    assert h.module.validate_logistics_ownership(h.manifest)["source_root"] == str(ROOT)
    with sqlite3.connect(h.workspace / "business.db") as c:
        assert c.execute("SELECT order_no,status FROM orders ORDER BY order_no").fetchall() == [("MOCK001", "shipped"), ("MOCK002", "processing"), ("MOCK003", "shipped")]
        assert c.execute("SELECT tracking_no,status,location FROM shipments ORDER BY tracking_no").fetchall() == [("MOCKSF001", "unknown", None), ("MOCKSF003", "unknown", None)]
    for name in ("resource-ownership.json", "profile.json", "budget.json", "business.db"):
        assert (h.workspace / name).stat().st_mode & 0o777 == 0o600
    assert h.workspace.stat().st_mode & 0o777 == 0o700
    before = set(inspect(h.engine).get_table_names())
    with h.module.owned_logistics_resources(h.manifest) as resources:
        assert resources.identity["ports"] == {"api": 18101, "mcp": 18102, "resolv": 18103}
    assert set(inspect(h.engine).get_table_names()) == before
    with pytest.raises(ValueError):
        h.module.prepare_logistics_demo(h.workspace, h.identity["conversation_database"])


def test_profile_cannot_override_loopback_or_dataset(prepared):
    h = prepared
    target = h.workspace / "profile.json"
    original = target.read_bytes()
    for key, value in [("mcp_url", "http://remote.example/mcp"), ("dataset_hash", "f" * 64)]:
        body = json.loads(original)
        body[key] = value
        target.write_text(json.dumps(body))
        with pytest.raises(ValueError):
            h.module.validate_logistics_ownership(h.manifest)
    target.write_bytes(original)
    assert h.module.validate_logistics_ownership(h.manifest)["state"] == "ready"


def test_swapped_manifest_or_symlink_is_rejected(prepared):
    h = prepared
    original = h.manifest.read_bytes()
    body = json.loads(original)
    body["resource_id"] = "f" * 32
    h.manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        h.module.validate_logistics_ownership(h.manifest)
    h.manifest.write_bytes(original)
    link = h.workspace / "alias.json"
    link.symlink_to(h.manifest)
    with pytest.raises(ValueError):
        h.module.validate_logistics_ownership(link)


def test_launch_parsers_accept_only_frozen_owned_ports(prepared):
    h = prepared
    for module_name, port in [("run_mock_logistics", 18101), ("run_logistics_mcp", 18102), ("run_logistics_demo", 18103)]:
        module = api("scripts." + module_name)
        args = module.parser().parse_args(["--ownership", str(h.manifest), "--port", str(port)])
        assert args.ownership == h.manifest and args.port == port
        with pytest.raises(ValueError):
            module.create_app(h.manifest, port + 5)


def test_occupied_port_does_not_overwrite_owned_process_record(prepared, monkeypatch):
    h = prepared
    before = h.manifest.read_bytes()

    class Occupied:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 18101)
            raise OSError("address already in use")

    assert hasattr(h.module, "serve_owned_app"), "missing owned port reservation before process record"
    monkeypatch.setattr(h.module.socket, "socket", lambda: Occupied())
    with pytest.raises(OSError, match="address already in use"):
        h.module.serve_owned_app(object(), h.manifest, "api", 18101)
    assert h.manifest.read_bytes() == before


def test_owned_listener_can_restart_after_time_wait_without_reusing_live_port(prepared, monkeypatch):
    import socket

    import uvicorn

    module = prepared.module
    operations = []

    class Listener:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, level, option, value):
            operations.append((level, option, value))

        def bind(self, address):
            if (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) not in operations:
                raise OSError("TIME_WAIT requires address reuse")
            operations.append("bound")

        def listen(self, backlog):
            operations.append("listening")

    class Server:
        def __init__(self, config):
            pass

        def run(self, *, sockets):
            assert len(sockets) == 1
            operations.append("served")

    monkeypatch.setattr(module.socket, "socket", lambda *args: Listener())
    monkeypatch.setattr(uvicorn, "Server", Server)
    module.serve_owned_app(object(), prepared.manifest, "mcp", prepared.identity["ports"]["mcp"])
    assert operations == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1), "bound", "listening", "served"]
