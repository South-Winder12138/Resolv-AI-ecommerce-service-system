"""Task-owned resources and read-only preservation for the knowledge gates."""

import json
import os
import re
import subprocess
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

from resolv.core.config import (
    ConversationDatabaseSettings,
    KnowledgeDatabaseSettings,
    MilvusSettings,
    Settings,
)
from resolv.db.checkpoints import setup_checkpoint_tables
from resolv.db.engine import create_db_engine
from resolv.db.init_db import init_db
from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, milvus_client
from resolv.services.knowledge_release import (
    POLICY_PATHS,
    ROOT,
    WORKSPACE,
    database_identity,
)

CONVERSATION_DB = "resolv_conversations_smoke_knowledge_20261015"
OWNER_KIND = "knowledge-branch-20261015-v1"
OWNERSHIP = ROOT / WORKSPACE / "resource-ownership.json"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_fingerprints(extra=()):
    names = [
        "pyproject.toml",
        "uv.lock",
        "frontend/package-lock.json",
        *POLICY_PATHS,
        "knowledge/file-config.formal.toml",
        *extra,
    ]
    names += [
        str(p.relative_to(ROOT))
        for folder in ("src", "migrations")
        for p in (ROOT / folder).rglob("*.py")
    ]
    names += [
        str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*knowledge*.py")
    ]
    names += [
        "tests/knowledge_branch_fixtures.py",
        "tests/branch_fixtures.py",
        "tests/understanding_fixtures.py",
        "tests/test_context.py",
        "tests/test_understanding_context.py",
        "tests/conversation_fixtures.py",
        "tests/hybrid_fixtures.py",
    ]
    return {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(names))
    }


def configured_urls():
    return (
        make_url(KnowledgeDatabaseSettings().knowledge_database_url.get_secret_value()),
        make_url(
            ConversationDatabaseSettings().conversation_database_url.get_secret_value()
        ),
    )


def validate_ownership(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / WORKSPACE).resolve()):
        raise ValueError("Ownership manifest is outside task")
    obj = json.loads(path.read_text())
    if (
        obj.get("kind") != OWNER_KIND
        or not re.fullmatch("[0-9a-f]{32}", obj.get("owner", ""))
        or obj.get("conversation_database") != CONVERSATION_DB
        or obj.get("knowledge_database") != "mhknow_gate_" + obj.get("owner", "")[:16]
        or obj.get("collection") != "mhknow_gate_" + obj.get("owner", "")[:16]
        or obj.get("business_db") != str((ROOT / WORKSPACE / "business.db").resolve())
    ):
        raise ValueError("Foreign resource ownership")
    return obj


def _engine(url):
    return create_db_engine(
        url.render_as_string(hide_password=False),
        hide_parameters=True,
        pool_pre_ping=True,
    )


def _owned_database(url, obj, *, initialize=False):
    admin = _engine(url._replace(database=None))
    try:
        with admin.begin() as c:
            exists = (
                c.execute(
                    text(
                        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=:name"
                    ),
                    {"name": url.database},
                ).scalar()
                is not None
            )
            if not exists:
                if not initialize:
                    raise ValueError("Owned resource missing")
                if not re.fullmatch("[a-z_][a-z0-9_]*", url.database):
                    raise ValueError("Unsafe owned name")
                c.exec_driver_sql(
                    f"CREATE DATABASE `{url.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
                )
        engine = _engine(url)
        try:
            with engine.begin() as c:
                if not exists:
                    c.exec_driver_sql(
                        "CREATE TABLE knowledge_gate_ownership (owner VARCHAR(32) PRIMARY KEY, kind VARCHAR(64) NOT NULL)"
                    )
                    c.execute(
                        text(
                            "INSERT INTO knowledge_gate_ownership(owner,kind) VALUES (:owner,:kind)"
                        ),
                        {"owner": obj["owner"], "kind": OWNER_KIND},
                    )
                rows = c.execute(
                    text("SELECT owner,kind FROM knowledge_gate_ownership")
                ).all()
                if rows != [(obj["owner"], OWNER_KIND)]:
                    raise ValueError("Database belongs to another task")
            return engine
        except BaseException:
            engine.dispose()
            raise
    finally:
        admin.dispose()


def database_exists(url):
    engine = _engine(url._replace(database=None))
    try:
        with engine.connect() as connection:
            return (
                connection.execute(
                    text(
                        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=:name"
                    ),
                    {"name": url.database},
                ).scalar()
                is not None
            )
    finally:
        engine.dispose()


def owned_bootstrap_sql(obj, accounts):
    owner = obj.get("owner", "")
    if (
        obj.get("kind") != OWNER_KIND
        or not re.fullmatch("[0-9a-f]{32}", owner)
        or obj.get("knowledge_database") != "mhknow_gate_" + owner[:16]
        or obj.get("conversation_database") != CONVERSATION_DB
        or obj.get("collection") != "mhknow_gate_" + owner[:16]
    ):
        raise ValueError("Administrative bootstrap requires exact owned resources")
    principals = []
    for account in set(accounts):
        user, separator, host = account.partition("@")
        if (
            separator != "@"
            or not re.fullmatch("[A-Za-z0-9_]+", user)
            or not re.fullmatch("[%A-Za-z0-9_.:-]+", host)
        ):
            raise ValueError("Unsafe existing MySQL account identity")
        principals.append("'" + user + "'@'" + host + "'")
    if not principals:
        raise ValueError("Existing configured accounts are required")
    statements = []
    for name in (obj["knowledge_database"], obj["conversation_database"]):
        statements.extend(
            [
                f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;",
                f"CREATE TABLE `{name}`.knowledge_gate_ownership (owner VARCHAR(32) PRIMARY KEY, kind VARCHAR(64) NOT NULL);",
                f"INSERT INTO `{name}`.knowledge_gate_ownership(owner,kind) VALUES ('{owner}','{OWNER_KIND}');",
            ]
        )
        grant_name = name.replace("_", "\\_")
        for principal in principals:
            statements.append(
                f"GRANT ALL PRIVILEGES ON `{grant_name}`.* TO {principal};"
            )
    return "\n".join(statements)


def _admin_mysql(sql):
    # Password stays in the existing container's process environment, never argv/output.
    result = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "resolv-knowledge-mysql-1",
            "sh",
            "-c",
            'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" exec mysql -uroot --batch --raw --skip-column-names',
        ],
        input=sql,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        save_json(
            ROOT / WORKSPACE / "resource-admin-failure-private.json",
            {"exit_code": result.returncode, "stderr": result.stderr},
        )
        raise ValueError("Task database administration failed")
    return result.stdout.strip().splitlines()


def admin_preflight(obj, urls):
    # Check local admin reaches exactly the configured server before scoped DDL.
    fingerprints, accounts = [], []
    for url in urls:
        engine = _engine(url)
        try:
            fingerprints.append(
                database_identity(engine, url.database)["server_fingerprint"]
            )
            with engine.connect() as connection:
                accounts.append(
                    connection.execute(text("SELECT CURRENT_USER()")).scalar_one()
                )
        finally:
            engine.dispose()
    # Escaped database identifiers follow the existing server's wildcard mode.
    # Never change machine/server settings as part of project resource setup.
    if _admin_mysql("SELECT @@partial_revokes;") != ["0"]:
        raise ValueError(
            "Unexpected database privilege interpretation; no settings changed"
        )
    sql = owned_bootstrap_sql(obj, accounts)
    identity = _admin_mysql("SELECT @@server_uuid;")
    if len(identity) != 1 or any(
        value != sha256(identity[0].encode()).hexdigest() for value in fingerprints
    ):
        raise ValueError(
            "Administrative server identity differs from configured stores"
        )
    names = (obj["knowledge_database"], obj["conversation_database"])
    existing = _admin_mysql(
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME IN ('"
        + "','".join(names)
        + "');"
    )
    if existing:
        raise ValueError("Refusing an existing unowned database found by administrator")
    return sql


def initialize_resources(conversation_database, business_db):
    if (
        conversation_database != CONVERSATION_DB
        or Path(business_db).resolve() != (ROOT / WORKSPACE / "business.db").resolve()
    ):
        raise ValueError("Only approved task resources allowed")
    if OWNERSHIP.exists() or Path(business_db).exists():
        raise ValueError("Existing resources require explicit ownership manifest")
    knowledge_url, conversation_url = configured_urls()
    if (knowledge_url.database, conversation_url.database) != (
        "resolv_knowledge",
        "resolv_conversations",
    ):
        raise ValueError("Configured formal target mismatch")
    owner = uuid4().hex
    obj = {
        "kind": OWNER_KIND,
        "owner": owner,
        "conversation_database": CONVERSATION_DB,
        "knowledge_database": "mhknow_gate_" + owner[:16],
        "collection": "mhknow_gate_" + owner[:16],
        "business_db": str(Path(business_db).resolve()),
        "provider_cases": [],
    }
    # Read every target before creating or claiming the first resource.
    for url in (
        knowledge_url.set(database=obj["knowledge_database"]),
        conversation_url.set(database=CONVERSATION_DB),
    ):
        if database_exists(url):
            raise ValueError("Refusing an existing unowned database")
    with milvus_client(MilvusSettings()) as client:
        if client.has_collection(
            collection_name=obj["collection"], timeout=MILVUS_TIMEOUT_SECONDS
        ):
            raise ValueError("Refusing an existing unowned collection")
    bootstrap_sql = admin_preflight(obj, (knowledge_url, conversation_url))
    # No external resource is claimed by name alone on subsequent invocations.
    save_json(OWNERSHIP, obj)
    _admin_mysql(bootstrap_sql)
    engines = []
    try:
        for label, url, ini in [
            (
                "knowledge",
                knowledge_url.set(database=obj["knowledge_database"]),
                "alembic.knowledge.ini",
            ),
            (
                "conversation",
                conversation_url.set(database=CONVERSATION_DB),
                "alembic.conversation.ini",
            ),
        ]:
            engine = _owned_database(url, obj, initialize=True)
            engines.append(engine)
            obj[label + "_server"] = database_identity(engine, url.database)[
                "server_fingerprint"
            ]
            cfg = Config(str(ROOT / ini))
            with engine.begin() as c:
                cfg.attributes["connection"] = c
                command.upgrade(cfg, "head")
        settings = ConversationDatabaseSettings(
            _env_file=None,
            conversation_database_url=conversation_url.set(
                database=CONVERSATION_DB
            ).render_as_string(hide_password=False),
        )
        setup_checkpoint_tables(settings)
        business = _engine(make_url("sqlite:///" + obj["business_db"]))
        engines.append(business)
        init_db(business)
        save_json(OWNERSHIP, obj)
    finally:
        for engine in engines:
            engine.dispose()
    return obj


@contextmanager
def owned_resources(path):
    obj = validate_ownership(path)
    k, c = configured_urls()
    engines = []
    try:
        for label, url in [
            ("knowledge", k.set(database=obj["knowledge_database"])),
            ("conversation", c.set(database=obj["conversation_database"])),
        ]:
            engine = _owned_database(url, obj)
            engines.append(engine)
            if (
                database_identity(engine, url.database)["server_fingerprint"]
                != obj[label + "_server"]
            ):
                raise ValueError("Owned database server changed")
        if not Path(obj["business_db"]).is_file():
            raise ValueError("Owned business DB missing")
        yield obj, engines[0], engines[1]
    finally:
        for engine in engines:
            engine.dispose()


@contextmanager
def runtime_environment(obj, *, formal=False):
    k, c = configured_urls()
    values = {
        "CONVERSATION_DATABASE_URL": c.set(
            database=obj["conversation_database"]
        ).render_as_string(hide_password=False),
        "KNOWLEDGE_DATABASE_URL": k.set(
            database="resolv_knowledge" if formal else obj["knowledge_database"]
        ).render_as_string(hide_password=False),
        "DATABASE_URL": "sqlite:///" + obj["business_db"],
        "UNDERSTANDING_ENABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def formal_snapshot(layout=None):
    result = {}
    k, c = configured_urls()
    for label, url in [
        ("knowledge", k),
        ("conversation", c),
        ("business", make_url(Settings().database_url)),
    ]:
        engine = _engine(url)
        try:
            inspector = inspect(engine)
            tables = (
                {
                    table: [col["name"] for col in inspector.get_columns(table)]
                    for table in inspector.get_table_names()
                }
                if layout is None
                else layout[label]["columns"]
            )
            rows = {}
            with engine.connect() as conn:
                for table, columns in tables.items():
                    if table.endswith("alembic_version"):
                        continue
                    # Use captured old columns for incremental-migration preservation.
                    values = conn.exec_driver_sql(
                        "SELECT "
                        + ",".join("`" + col + "`" for col in columns)
                        + " FROM `"
                        + table
                        + "`"
                    ).all()
                    encoded = sorted(
                        json.dumps(
                            list(row),
                            ensure_ascii=False,
                            default=lambda v: (
                                v.hex() if isinstance(v, bytes) else str(v)
                            ),
                            sort_keys=True,
                        )
                        for row in values
                    )
                    rows[table] = {
                        "count": len(encoded),
                        "sha256": sha256("\n".join(encoded).encode()).hexdigest(),
                    }
            result[label] = {"columns": tables, "tables": rows}
        finally:
            engine.dispose()
    with milvus_client(MilvusSettings()) as client:
        names = (
            sorted(client.list_collections(timeout=MILVUS_TIMEOUT_SECONDS))
            if layout is None
            else layout["milvus"]["names"]
        )
        state = {}
        for name in names:
            desc = client.describe_collection(
                collection_name=name, timeout=MILVUS_TIMEOUT_SECONDS
            )
            count = client.query(
                collection_name=name,
                filter="",
                output_fields=["count(*)"],
                consistency_level="Strong",
                timeout=MILVUS_TIMEOUT_SECONDS,
            )
            data = client.query(
                collection_name=name,
                filter="",
                output_fields=["*"],
                limit=16384,
                consistency_level="Strong",
                timeout=MILVUS_TIMEOUT_SECONDS,
            )
            if len(data) != count[0]["count(*)"]:
                raise ValueError("Incomplete preserved collection snapshot")
            encoded = sorted(
                json.dumps(row, sort_keys=True, default=str) for row in data
            )
            state[name] = {
                "schema_sha256": sha256(
                    json.dumps(desc, sort_keys=True, default=str).encode()
                ).hexdigest(),
                "count": count,
                "rows_sha256": sha256("\n".join(encoded).encode()).hexdigest(),
            }
        result["milvus"] = {"names": names, "collections": state}
    return result


def begin_provider_case(obj, case_id):
    # Persist before sending; failed real attempts also consume the eight-case budget.
    current = validate_ownership(OWNERSHIP)
    if (
        current["owner"] != obj["owner"]
        or len(current["provider_cases"]) >= 8
        or case_id in current["provider_cases"]
    ):
        raise ValueError("Real provider case budget or duplicate exceeded")
    current["provider_cases"].append(case_id)
    save_json(OWNERSHIP, current)
