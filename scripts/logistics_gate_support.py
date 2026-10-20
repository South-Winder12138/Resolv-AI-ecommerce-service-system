"""Task-owned logistics resources. Only explicit preparation performs DDL."""

import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from resolv.core.config import ConversationDatabaseSettings, DatabaseSettings
from resolv.db.checkpoints import checkpoint_session, setup_checkpoint_tables
from resolv.db.engine import create_session_factory
from resolv.repositories.conversation import fingerprint
from resolv.schemas.logistics import LogisticsBackendProfile
from resolv.services.logistics_dataset import load_dataset, strict_json_loads
from resolv.services.logistics_mcp_server import logistics_tool_definition

SOURCE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = SOURCE_ROOT / ".superpowers/sdd/2026-10-20-resolv-logistics-mcp"
REPOSITORY_ROOT = RESOURCE_ROOT.parents[2]
DATABASE = "resolv_conversations_smoke_logistics_mcp_20261020"
OWNER = "resolv-logistics-mcp-v1"
KIND = "logistics-mcp-20261020-v1"
FROZEN = ("kind", "owner_id", "resource_id", "source_root", "source_revision", "resource_root", "conversation_database", "server_fingerprint", "ports", "dataset_hash", "profile_hash", "business_db", "api_fault")


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temporary.open("x") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_revision(root=SOURCE_ROOT):
    exported = root / ".logistics-source.json"
    if exported.is_file():
        value = strict_json_loads(exported.read_bytes())["revision"]
    else:
        value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Invalid source revision")
    return value


def _plain_path(path):
    path = Path(path).absolute()
    if path != path.resolve() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Symlinked owned path")
    return path


def _identity(path):
    path = _plain_path(path)
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino}


def _read_json(path):
    path = _plain_path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("Owned file permissions changed")
    return strict_json_loads(path.read_bytes())


def validate_logistics_ownership(path: Path) -> dict:
    return _validate_logistics_ownership(path, source_root=str(SOURCE_ROOT), revision=source_revision())


def _validate_logistics_ownership(path, *, source_root, revision):
    path = _plain_path(path)
    if path != RESOURCE_ROOT.resolve() / "resource-ownership.json":
        raise ValueError("Foreign logistics ownership path")
    obj = _read_json(path)
    if (
        obj.get("kind") != KIND or obj.get("owner_id") != OWNER or obj.get("state") != "ready"
        or obj.get("conversation_database") != DATABASE
        or obj.get("source_root") != source_root
        or obj.get("source_revision") != revision
        or obj.get("resource_root") != str(path.parent)
        or obj.get("business_db") != str(path.parent / "business.db")
        or not re.fullmatch(r"[0-9a-f]{32}", obj.get("resource_id", ""))
        or not re.fullmatch(r"[0-9a-f]{64}", obj.get("server_fingerprint", ""))
        or obj.get("api_fault") != "api_timeout"
        or set(obj.get("ports", {})) != {"api", "mcp", "resolv"}
        or any(type(port) is not int or not 1024 <= port <= 65535 for port in obj["ports"].values())
        or len(set(obj["ports"].values())) != 3
    ):
        raise ValueError("Foreign logistics resource identity")
    marker = _read_json(path.parent / ".logistics-resource-owner.json")
    if marker != {"resource_id": obj["resource_id"], "identity_hash": fingerprint({k: obj[k] for k in FROZEN})}:
        raise ValueError("Swapped logistics manifest")
    profile_data = _read_json(path.parent / "profile.json")
    profile = LogisticsBackendProfile.model_validate_json(json.dumps(profile_data))
    dataset = load_dataset(SOURCE_ROOT / "logistics/datasets/mock-logistics-v1.json")
    tool = logistics_tool_definition()
    if (
        profile.fingerprint != obj["profile_hash"] or profile.dataset_hash != dataset.dataset_hash
        or profile.dataset_hash != obj["dataset_hash"] or profile.dataset_version != dataset.dataset_version
        or profile.mcp_url != f"http://127.0.0.1:{obj['ports']['mcp']}/mcp"
        or profile.api_url != f"http://127.0.0.1:{obj['ports']['api']}"
        or profile.input_schema_hash != fingerprint(tool.input_schema)
        or profile.output_schema_hash != fingerprint(tool.output_schema)
        or _identity(obj["business_db"]) != obj.get("business_identity")
    ):
        raise ValueError("Logistics profile or resource drift")
    _read_json(path.parent / "budget.json")
    return obj


def rebind_logistics_source(path, *, previous_revision, previous_source_root, allow_terminal_budget=False):
    """Explicit code-only handoff; preserve database, ports, profile and closed evidence."""
    path = _plain_path(path)
    with path.with_suffix(".rebind.lock").open("a+") as lock, (path.parent / "budget.lock").open("a+") as budget_lock, (path.parent / ".logistics-process.lock").open("a+") as process_lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fcntl.flock(budget_lock, fcntl.LOCK_EX)
        fcntl.flock(process_lock, fcntl.LOCK_EX)
        obj = _validate_logistics_ownership(path, source_root=previous_source_root, revision=previous_revision)
        budget = _read_json(path.parent / "budget.json")
        unused = budget.get("state") == "unopened" and not budget.get("calls")
        terminal = allow_terminal_budget and budget.get("state") in {"closed", "completed"}
        if not (unused or terminal) or budget.get("retrievals") != 0:
            raise ValueError("Source rebind requires unopened/unused model budget")
        for label in ("C1", "C2", "C3"):
            child_path = path.parent / f"budget-{label}.json"
            if child_path.exists() and _read_json(child_path).get("state") not in {"closed", "completed"}:
                raise ValueError(f"{label} must be terminal before source handoff")
        for process in obj.get("processes", {}).values():
            try:
                os.kill(process["pid"], 0)
            except ProcessLookupError:
                continue
            raise ValueError("Source rebind requires all owned processes stopped")
        engine = _open_owned_conversation(obj)
        engine.dispose()
        import sqlite3

        with sqlite3.connect(Path(obj["business_db"]).as_uri() + "?mode=ro", uri=True) as connection:
            if connection.execute("SELECT resource_id,owner_id FROM logistics_gate_ownership").fetchall() != [(obj["resource_id"], OWNER)]:
                raise ValueError("Foreign source rebind SQLite")
        history = {"previous_revision": previous_revision, "previous_source_root": previous_source_root,
            "revision": source_revision(), "source_root": str(SOURCE_ROOT), "resource_id": obj["resource_id"], "terminal_budget_unchanged": bool(terminal)}
        obj.update(source_revision=source_revision(), source_root=str(SOURCE_ROOT))
        obj.setdefault("source_rebindings", []).append(history)
        save_json(path.parent / ("source-rebind-" + uuid4().hex + ".json"), history)
        save_json(path.parent / ".logistics-resource-owner.json", {"resource_id": obj["resource_id"], "identity_hash": fingerprint({k: obj[k] for k in FROZEN})})
        save_json(path, obj)
        return validate_logistics_ownership(path)


def owned_profile(path):
    validate_logistics_ownership(path)
    return LogisticsBackendProfile.model_validate_json((Path(path).parent / "profile.json").read_bytes())


def _allocate_ports():
    with ExitStack() as stack:
        ports = {}
        for name in ("api", "mcp", "resolv"):
            listener = stack.enter_context(socket.socket())
            listener.bind(("127.0.0.1", 0))
            ports[name] = listener.getsockname()[1]
        return ports


def _administrative_identity():
    from scripts.refund_gate_support import admin_mysql

    value = admin_mysql("SELECT SHA2(@@server_uuid,256);")
    if len(value) != 1 or not re.fullmatch(r"[0-9a-f]{64}", value[0]):
        raise ValueError("Cannot identify logistics database server")
    return value[0]


def _conversation_url():
    from sqlalchemy.engine import make_url

    settings = ConversationDatabaseSettings(_env_file=REPOSITORY_ROOT / ".env")
    return make_url(settings.conversation_database_url.get_secret_value()).set(database=DATABASE)


def _prepare_mysql(obj):
    from alembic import command
    from alembic.config import Config
    from scripts.refund_gate_support import _engine, admin_mysql

    if _administrative_identity() != obj["server_fingerprint"]:
        raise ValueError("Administrative server changed")
    if admin_mysql(f"SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME='{DATABASE}';"):
        raise ValueError("Refusing existing unowned logistics database")
    url = _conversation_url()
    # Identify the existing project account without exposing credentials.
    engine = _engine(url.set(database="resolv_conversations"))
    try:
        with engine.connect() as c:
            account = c.scalar(text("SELECT CURRENT_USER()"))
    finally:
        engine.dispose()
    if not re.fullmatch(r"[A-Za-z0-9_-]+@[A-Za-z0-9_.%:-]+", account):
        raise ValueError("Unsupported project SQL account")
    user, host = account.split("@")
    admin_mysql(
        f"CREATE DATABASE `{DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;"
        f"CREATE TABLE `{DATABASE}`.logistics_gate_ownership(resource_id VARCHAR(32) PRIMARY KEY, owner_id VARCHAR(64) NOT NULL);"
        f"INSERT INTO `{DATABASE}`.logistics_gate_ownership VALUES ('{obj['resource_id']}','{OWNER}');"
        f"GRANT SELECT,INSERT,UPDATE,DELETE,CREATE,ALTER,INDEX,REFERENCES ON `{DATABASE}`.* TO '{user}'@'{host}';"
    )
    engine = _open_owned_conversation(obj)
    try:
        cfg = Config(str(SOURCE_ROOT / "alembic.conversation.ini"))
        with engine.begin() as connection:
            cfg.attributes["connection"] = connection
            command.upgrade(cfg, "20261020_0006")
        setup_checkpoint_tables(ConversationDatabaseSettings(_env_file=None, conversation_database_url=url.render_as_string(hide_password=False)))
    finally:
        engine.dispose()


def _open_owned_conversation(obj):
    from scripts.refund_gate_support import _engine, server_fingerprint

    engine = _engine(_conversation_url())
    try:
        if server_fingerprint(engine) != obj["server_fingerprint"]:
            raise ValueError("Owned logistics SQL server changed")
        with engine.connect() as c:
            if c.execute(text("SELECT resource_id,owner_id FROM logistics_gate_ownership")).all() != [(obj["resource_id"], OWNER)]:
                raise ValueError("Foreign logistics SQL ownership")
        return engine
    except BaseException:
        engine.dispose()
        raise


def prepare_logistics_demo(workspace: Path, database: str) -> dict:
    from resolv.db.engine import create_db_engine
    from resolv.db.init_db import init_db
    from resolv.models import Order, Shipment

    workspace = _plain_path(workspace)
    if workspace != RESOURCE_ROOT.resolve() or database != DATABASE or not workspace.is_dir():
        raise ValueError("Explicit task workspace/database required")
    if any((workspace / name).exists() for name in ("resource-ownership.json", ".logistics-resource-owner.json", "business.db", "profile.json", "budget.json")):
        raise ValueError("Refusing existing or unowned logistics resources")
    dataset = load_dataset(SOURCE_ROOT / "logistics/datasets/mock-logistics-v1.json")
    ports, tool = _allocate_ports(), logistics_tool_definition()
    profile = LogisticsBackendProfile(
        backend="mock-logistics-mcp-v1", source="mock-logistics-v1",
        dataset_version=dataset.dataset_version, dataset_hash=dataset.dataset_hash,
        mcp_url=f"http://127.0.0.1:{ports['mcp']}/mcp", api_url=f"http://127.0.0.1:{ports['api']}",
        input_schema_hash=fingerprint(tool.input_schema), output_schema_hash=fingerprint(tool.output_schema),
    )
    obj = {
        "kind": KIND, "owner_id": OWNER, "resource_id": uuid4().hex, "state": "initializing",
        "resource_root": str(workspace), "source_root": str(SOURCE_ROOT), "source_revision": source_revision(),
        "conversation_database": DATABASE, "server_fingerprint": _administrative_identity(),
        "ports": ports, "dataset_hash": dataset.dataset_hash, "profile_hash": profile.fingerprint,
        "business_db": str(workspace / "business.db"), "api_fault": "api_timeout", "processes": {},
    }
    os.chmod(workspace, 0o700)
    save_json(workspace / "resource-ownership.json", obj)
    save_json(workspace / ".logistics-resource-owner.json", {"resource_id": obj["resource_id"], "identity_hash": fingerprint({k: obj[k] for k in FROZEN})})
    save_json(workspace / "profile.json", profile.model_dump(mode="json"))
    save_json(workspace / "budget.json", {"state": "unopened", "max_model_calls": 48, "retrievals": 0, "calls": [], "cases": {}})
    _prepare_mysql(obj)
    engine = create_db_engine("sqlite:///" + obj["business_db"])
    try:
        init_db(engine)
        with engine.begin() as c:
            c.exec_driver_sql("CREATE TABLE logistics_gate_ownership(resource_id TEXT PRIMARY KEY,owner_id TEXT NOT NULL)")
            c.execute(text("INSERT INTO logistics_gate_ownership VALUES (:id,:owner)"), {"id": obj["resource_id"], "owner": OWNER})
        with create_session_factory(engine).begin() as session:
            for i, state in [(1, "shipped"), (2, "processing"), (3, "shipped")]:
                order = Order(order_no=f"MOCK00{i}", status=state)
                session.add(order)
                session.flush()
                if i != 2:
                    session.add(Shipment(order_id=order.id, tracking_no=f"MOCKSF00{i}", status="unknown", location=None))
    finally:
        engine.dispose()
    os.chmod(obj["business_db"], 0o600)
    obj.update(state="ready", business_identity=_identity(obj["business_db"]))
    save_json(workspace / "resource-ownership.json", obj)
    validate_logistics_ownership(workspace / "resource-ownership.json")
    return obj


@dataclass
class LogisticsOwnedResources:
    identity: dict
    conversation_factory: object
    business_factory: object
    checkpoint_factory: object


@contextmanager
def owned_logistics_resources(path: Path):
    from resolv.db.engine import create_db_engine
    from resolv.services.routed_runtime import validate_conversation_runtime_store

    obj = validate_logistics_ownership(path)
    conversation = _open_owned_conversation(obj)
    business = create_db_engine("sqlite:///" + obj["business_db"])
    try:
        with business.connect() as c:
            if c.execute(text("SELECT resource_id,owner_id FROM logistics_gate_ownership")).all() != [(obj["resource_id"], OWNER)]:
                raise ValueError("Foreign logistics SQLite ownership")
        factory = create_session_factory(conversation)
        validate_conversation_runtime_store(factory, require_logistics=True)
        # Construction does not connect; the explicit checkpoint factory is
        # invoked only by the actual conversation executor.
        settings = ConversationDatabaseSettings(_env_file=None, conversation_database_url=_conversation_url().render_as_string(hide_password=False))
        yield LogisticsOwnedResources(obj, factory, create_session_factory(business), partial(checkpoint_session, settings))
    finally:
        business.dispose()
        conversation.dispose()


def validate_service_port(path, role, port):
    obj = validate_logistics_ownership(path)
    if obj["ports"][role] != port:
        raise ValueError("Service port differs from frozen profile")
    return obj


def record_service_process(path, role, port):
    path = Path(path)
    with (path.parent / ".logistics-process.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        obj = validate_service_port(path, role, port)
        obj["processes"][role] = {"pid": os.getpid(), "port": port, "source_revision": obj["source_revision"]}
        save_json(path, obj)


def serve_owned_app(app, path, role, port):
    import uvicorn

    validate_service_port(path, role, port)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(128)
        record_service_process(path, role, port)
        audited = LogisticsHTTPAudit(app, role=role, path=Path(path).parent / (role + "-http.jsonl"))
        server = uvicorn.Server(uvicorn.Config(audited, host="127.0.0.1", port=port))
        server.run(sockets=[listener])


def append_json_line(path, value):
    with Path(path).open("a") as handle:
        os.chmod(path, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_http_events(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class LogisticsHTTPAudit:
    def __init__(self, app, *, role, path):
        self.app, self.role, self.path = app, role, path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            chunk = await receive()
            chunks.append(chunk)
            size += len(chunk.get("body", b""))
            if size > 65536:
                await send({"type": "http.response.start", "status": 413, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            if not chunk.get("more_body"):
                break
        rpc = None
        try:
            body = strict_json_loads(b"".join(c.get("body", b"") for c in chunks))
            if body.get("method") in {"initialize", "notifications/initialized", "notifications/cancelled", "tools/list", "tools/call"}:
                rpc = body["method"]
        except (ValueError, TypeError, RuntimeError):
            rpc = None
        record = {"event_id": uuid4().hex, "role": self.role, "pid": os.getpid(), "method": scope["method"], "path": scope["path"][:256], "rpc_method": rpc}
        append_json_line(self.path, {**record, "phase": "start", "at": time.monotonic()})

        async def replay():
            return chunks.pop(0) if chunks else await receive()

        try:
            await self.app(scope, replay, send)
        finally:
            append_json_line(self.path, {**record, "phase": "finish", "at": time.monotonic()})


def network_events(ownership):
    root = Path(ownership).parent
    return [e for role in ("api", "mcp", "resolv") for e in read_http_events(root / (role + "-http.jsonl"))]


def new_logistics_report_dir(ownership, report_dir):
    obj = validate_logistics_ownership(ownership)
    target = _plain_path(report_dir)
    if target.parent != Path(obj["resource_root"]) or target.exists():
        raise ValueError("A new direct child report directory is required")
    target.mkdir(mode=0o700)
    return target


def validate_logistics_artifact(ownership, path):
    obj = validate_logistics_ownership(ownership)
    target = _plain_path(path)
    if Path(obj["resource_root"]) not in target.parents or target.parent == Path(obj["resource_root"]):
        raise ValueError("Fault artifact must remain in an owned report directory")
    if not target.parent.is_dir():
        raise ValueError("Fault artifact parent is not prepared")
    return target


def snapshot_logistics_protection(*, layout=None):
    from scripts.refund_gate_support import snapshot_formal_resources

    baseline = strict_json_loads((RESOURCE_ROOT / "implementation-baseline.json").read_bytes())
    paths = set(baseline["protected"]) | {".env", "pyproject.toml", "uv.lock", "frontend/package-lock.json", "knowledge/active-release.json"}
    # Preserve prior task budgets/selectors and public acceptance evidence.
    for path in (REPOSITORY_ROOT / ".superpowers/sdd").rglob("*.json"):
        if RESOURCE_ROOT not in path.parents and any(word in path.name for word in ("budget", "selector", "ownership", "ledger")):
            paths.add(str(path.relative_to(REPOSITORY_ROOT)))
    for path in (REPOSITORY_ROOT / "docs/superpowers/acceptance").rglob("*"):
        if path.is_file() and "2026-10-20-logistics-mcp" not in str(path):
            paths.add(str(path.relative_to(REPOSITORY_ROOT)))
    import hashlib

    files = {p: hashlib.sha256((REPOSITORY_ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)}
    # Enumerate again: a prior layout would silently omit newly added objects.
    settings = DatabaseSettings(_env_file=REPOSITORY_ROOT / ".env")
    # The unset default is relative to config.py, which lives in the export.
    # Formal preservation must still target the original repository's file.
    business_url = settings.database_url if "database_url" in settings.model_fields_set else "sqlite:///" + str(REPOSITORY_ROOT / "data/resolv.db")
    formal = snapshot_formal_resources(business_database_url=business_url)
    return {"files": files, "formal": formal}


@contextmanager
def logistics_child(ownership, report_dir, *, role, worker_module=None, extra=()):
    obj = validate_logistics_ownership(ownership)
    module = worker_module or {"api": "scripts.run_mock_logistics", "mcp": "scripts.run_logistics_mcp", "resolv": "scripts.run_logistics_demo"}[role]
    log = Path(report_dir) / (role + "-" + uuid4().hex + ".log")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT / "src") + os.pathsep + str(SOURCE_ROOT)
    with log.open("x") as output:
        os.chmod(log, 0o600)
        child = subprocess.Popen([sys.executable, "-m", module, "--ownership", str(ownership), "--port", str(obj["ports"][role]), *extra], cwd=SOURCE_ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            import httpx2

            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise ValueError("Owned service exited before readiness; inspect private process log")
                try:
                    with httpx2.Client(trust_env=False, timeout=0.5) as http:
                        response = http.get(f"http://127.0.0.1:{obj['ports'][role]}" + ("/health" if role == "resolv" else "/healthz" if role == "api" else "/mcp"))
                    if response.status_code == (405 if role == "mcp" else 200):
                        break
                except httpx2.HTTPError:
                    time.sleep(0.05)
            else:
                raise ValueError("Owned service readiness deadline exceeded")
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
