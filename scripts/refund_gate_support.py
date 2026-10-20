"""Refund-only resource ownership, bounded call ledger and formal read preservation."""

import fcntl
import json
import os
import re
import sqlite3
import subprocess
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import make_url

from resolv.core.config import (
    ConversationDatabaseSettings,
    KnowledgeDatabaseSettings,
    MilvusSettings,
    Settings,
)
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.services.hybrid_fingerprints import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ".superpowers/sdd/2026-10-16-resolv-refund-flow"
KIND = "refund-flow-20261016-v1"
EXPECTED_SERVER = "ddf9166a802d9c308fdc4de5ad6bbbab8c26b556c54bf8e83a06b1977b8e46d9"
LIMITS = {
    "resolution": 10,
    "intents": 10,
    "refund_query": 11,
    "refund_decision": 63,
    "refund_summary": 2,
}
R1_LIMITS = {"refund_query": 1, "refund_decision": 3, "refund_summary": 2}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temp.open("x") as handle:
        os.chmod(temp, 0o600)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def new_manifest(workspace, *, owner, server_fingerprint):
    workspace = Path(workspace).resolve()
    if (
        workspace != (ROOT / WORKSPACE).resolve()
        or not re.fullmatch("[0-9a-f]{32}", owner)
        or not re.fullmatch("[0-9a-f]{64}", server_fingerprint)
    ):
        raise ValueError("Invalid refund resource identity")
    return {
        "kind": KIND,
        "owner": owner,
        "server_fingerprint": server_fingerprint,
        "conversation_database": "mhrefund_c_" + owner[:16],
        "knowledge_database": "mhrefund_k_" + owner[:16],
        "collection": "mhrefund_k_" + owner[:16],
        "business_db": str(workspace / ("business-" + owner + ".db")),
        "state": "initializing",
        "retests": {},
    }


def validate_manifest(manifest):
    path = Path(manifest).resolve()
    if (
        path.parent != (ROOT / WORKSPACE).resolve()
        or path.name != "resource-ownership.json"
    ):
        raise ValueError("Foreign refund ownership path")
    obj = json.loads(path.read_text())
    expected = new_manifest(
        path.parent,
        owner=obj.get("owner", ""),
        server_fingerprint=obj.get("server_fingerprint", ""),
    )
    if any(
        obj.get(k) != expected[k]
        for k in (
            "kind",
            "owner",
            "server_fingerprint",
            "conversation_database",
            "knowledge_database",
            "collection",
            "business_db",
        )
    ):
        raise ValueError("Foreign refund resource identity")
    if obj.get("state") not in {"initializing", "ready"} or not isinstance(
        obj.get("retests"), dict
    ):
        raise ValueError("Invalid refund resource state")
    return obj


def file_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Owned business file missing or symlinked")
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino}


def initialize_business_ownership(engine, obj):
    with engine.begin() as c:
        c.exec_driver_sql(
            "CREATE TABLE refund_gate_ownership(owner TEXT PRIMARY KEY, kind TEXT NOT NULL)"
        )
        c.execute(
            text("INSERT INTO refund_gate_ownership(owner,kind) VALUES(:owner,:kind)"),
            {"owner": obj["owner"], "kind": KIND},
        )


def _verify_business(obj):
    path = Path(obj["business_db"])
    if file_identity(path) != obj.get("business_identity"):
        raise ValueError("Owned business file changed")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as c:
        if c.execute("SELECT owner,kind FROM refund_gate_ownership").fetchall() != [
            (obj["owner"], KIND)
        ]:
            raise ValueError("Foreign business ownership")


def configured_urls():
    k = make_url(KnowledgeDatabaseSettings().knowledge_database_url.get_secret_value())
    c = make_url(
        ConversationDatabaseSettings().conversation_database_url.get_secret_value()
    )
    if (k.database, c.database) != ("resolv_knowledge", "resolv_conversations"):
        raise ValueError("Configured formal databases differ")
    return k, c


def _engine(url):
    url = make_url(url) if isinstance(url, str) else url
    return create_db_engine(
        url.render_as_string(hide_password=False),
        hide_parameters=True,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5, "read_timeout": 30, "write_timeout": 30}
        if url.get_backend_name() == "mysql"
        else {},
    )


def server_fingerprint(engine):
    with engine.connect() as c:
        return sha256(c.scalar(text("SELECT @@server_uuid")).encode()).hexdigest()


def open_owned_database(url, obj, label):
    engine = _engine(url.set(database=obj[label + "_database"]))
    try:
        if server_fingerprint(engine) != obj["server_fingerprint"]:
            raise ValueError("Owned server changed")
        with engine.connect() as c:
            if c.execute(
                text("SELECT owner,kind FROM refund_gate_ownership")
            ).all() != [(obj["owner"], KIND)]:
                raise ValueError("Foreign database ownership")
        return engine
    except BaseException:
        engine.dispose()
        raise


@dataclass
class RefundOwnedResources:
    manifest: Path
    identity: dict
    knowledge_engine: object
    conversation_engine: object
    business_engine: object

    @property
    def knowledge_factory(self):
        return create_session_factory(self.knowledge_engine)

    @property
    def conversation_factory(self):
        return create_session_factory(self.conversation_engine)

    @property
    def business_factory(self):
        return create_session_factory(self.business_engine)


@contextmanager
def owned_refund_resources(manifest):
    manifest = Path(manifest).resolve()
    obj = validate_manifest(manifest)
    _verify_business(obj)
    k, c = configured_urls()
    engines = []
    try:
        knowledge = open_owned_database(k, obj, "knowledge")
        engines.append(knowledge)
        conversation = open_owned_database(c, obj, "conversation")
        engines.append(conversation)
        business = _engine(make_url("sqlite:///" + obj["business_db"]))
        engines.append(business)
        yield RefundOwnedResources(manifest, obj, knowledge, conversation, business)
    finally:
        for engine in engines:
            engine.dispose()


def admin_mysql(sql):
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
        text=True,
        capture_output=True,
        check=False,
        timeout=45,
    )
    if result.returncode:
        raise ValueError("Scoped refund resource administration failed")
    return result.stdout.strip().splitlines()


def bootstrap_sql(obj, accounts):
    expected = new_manifest(
        Path(obj["business_db"]).parent,
        owner=obj["owner"],
        server_fingerprint=obj["server_fingerprint"],
    )
    if any(
        obj.get(k) != expected[k]
        for k in ("kind", "conversation_database", "knowledge_database")
    ):
        raise ValueError("Foreign administrative target")
    principals = []
    for account in sorted(set(accounts)):
        if not re.fullmatch(r"[A-Za-z0-9_]+@[%A-Za-z0-9_.:-]+", account):
            raise ValueError("Invalid existing account")
        user, host = account.split("@")
        principals.append("'" + user + "'@'" + host + "'")
    if not principals:
        raise ValueError("Existing accounts required")
    statements = []
    for name in (obj["conversation_database"], obj["knowledge_database"]):
        statements.extend(
            [
                f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;",
                f"CREATE TABLE `{name}`.refund_gate_ownership(owner VARCHAR(32) PRIMARY KEY,kind VARCHAR(64) NOT NULL);",
                f"INSERT INTO `{name}`.refund_gate_ownership(owner,kind) VALUES ('{obj['owner']}','{KIND}');",
            ]
        )
        escaped = name.replace("_", "\\_")
        statements.extend(
            f"GRANT ALL PRIVILEGES ON `{escaped}`.* TO {principal};"
            for principal in principals
        )
    return "\n".join(statements)


def initialize_refund_resources(*, workspace):
    from alembic import command
    from alembic.config import Config

    from resolv.db.checkpoints import setup_checkpoint_tables
    from resolv.db.init_db import init_db

    workspace = Path(workspace).resolve()
    if (
        workspace != (ROOT / WORKSPACE).resolve()
        or (workspace / "resource-ownership.json").exists()
    ):
        raise ValueError("Existing/foreign resources require original manifest")
    k, c = configured_urls()
    owner = uuid4().hex
    obj = new_manifest(workspace, owner=owner, server_fingerprint=EXPECTED_SERVER)
    if Path(obj["business_db"]).exists():
        raise ValueError("Business file already exists")
    accounts = []
    for url in (k, c):
        engine = _engine(url)
        try:
            if server_fingerprint(engine) != EXPECTED_SERVER:
                raise ValueError("Configured server changed")
            with engine.connect() as connection:
                accounts.append(connection.scalar(text("SELECT CURRENT_USER()")))
        finally:
            engine.dispose()
    if admin_mysql("SELECT @@partial_revokes;") != ["0"] or admin_mysql(
        "SELECT SHA2(@@server_uuid,256);"
    ) != [EXPECTED_SERVER]:
        raise ValueError("Administrative server mismatch")
    names = (obj["conversation_database"], obj["knowledge_database"])
    if admin_mysql(
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME IN ('"
        + "','".join(names)
        + "');"
    ):
        raise ValueError("Refusing existing unowned database")
    manifest = workspace / "resource-ownership.json"
    save_json(manifest, obj)  # Evidence of ownership intent precedes scoped DDL.
    admin_mysql(bootstrap_sql(obj, accounts))
    engines = []
    try:
        for label, url, ini in [
            ("conversation", c, "alembic.conversation.ini"),
            ("knowledge", k, "alembic.knowledge.ini"),
        ]:
            engine = open_owned_database(url, obj, label)
            engines.append(engine)
            cfg = Config(str(ROOT / ini))
            with engine.begin() as connection:
                cfg.attributes["connection"] = connection
                command.upgrade(cfg, "head")
        setup_checkpoint_tables(
            ConversationDatabaseSettings(
                _env_file=None,
                conversation_database_url=c.set(
                    database=obj["conversation_database"]
                ).render_as_string(hide_password=False),
            )
        )
        business = _engine(make_url("sqlite:///" + obj["business_db"]))
        engines.append(business)
        init_db(business)
        initialize_business_ownership(business, obj)
        obj["business_identity"] = file_identity(obj["business_db"])
        obj["state"] = "ready"
        save_json(manifest, obj)
    finally:
        for engine in engines:
            engine.dispose()
    # Caller owns the returned context, not any open connections here.
    return RefundOwnedResources(manifest, obj, None, None, None)


@contextmanager
def manifest_lock(manifest):
    manifest = Path(manifest).resolve()
    validate_manifest(manifest)
    with manifest.with_suffix(".lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield validate_manifest(manifest)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _ledger_path(manifest):
    return Path(manifest).with_name("provider-calls.json")


def budget_ledger(manifest):
    obj = validate_manifest(manifest)
    path = _ledger_path(manifest)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data.get("owner") != obj["owner"] or data.get("hash") != canonical_hash(
        data.get("calls")
    ):
        raise ValueError("Refund budget ledger changed")
    return data["calls"]


def _save_ledger(manifest, calls, obj):
    save_json(
        _ledger_path(manifest),
        {"owner": obj["owner"], "hash": canonical_hash(calls), "calls": calls},
    )


@dataclass(frozen=True)
class InvocationReservation:
    manifest: Path
    invocation_id: str


def authorize_case_retest(manifest, *, case_id, source_hash):
    with manifest_lock(manifest) as obj:
        calls = budget_ledger(manifest)
        previous = [c for c in calls if c["case_id"] == case_id and c["attempt"] == 1]
        first = obj.get("semantic_attempts", {}).get(case_id + ":1")
        previous_hashes = {c["source_hash"] for c in previous}
        if first:
            previous_hashes.add(first["source_hash"])
        if (
            not re.fullmatch("F[1-8]", case_id)
            or not re.fullmatch("[0-9a-f]{64}", source_hash)
            or not previous_hashes
            or source_hash in previous_hashes
            or case_id in obj["retests"]
            or len(obj["retests"]) >= 2
        ):
            raise ValueError("Unauthorized extra semantic attempt")
        obj["retests"][case_id] = source_hash
        save_json(manifest, obj)


def reserve_refund_case_attempt(
    manifest, *, case_id, attempt, source_hash, continuation=None
):
    """An attempt is consumed even when startup fails before its first model."""
    if continuation == "native-return":
        from scripts.refund_native_regression import reserve_case

        return reserve_case(
            manifest, case_id=case_id, attempt=attempt, source_hash=source_hash
        )
    if continuation in {
        "round-control",
        "output-repair",
        "output-completeness",
        "context-identity",
        "chain-validation",
    }:
        from scripts.refund_round_regression import reserve_case

        return reserve_case(
            manifest,
            case_id=case_id,
            attempt=attempt,
            source_hash=source_hash,
            scope=continuation,
        )
    if continuation is not None:
        from scripts.refund_semantic_continuation import require, reserve_case

        require(continuation in {"C2", "C3"})
        return reserve_case(
            manifest,
            case_id=case_id,
            attempt=attempt,
            source_hash=source_hash,
            continuation=continuation,
        )
    with manifest_lock(manifest) as obj:
        from scripts.refund_continuation import authorized, grant

        c1 = authorized(obj, case_id, attempt, source_hash)
        g = grant(obj) if c1 else None
        if (
            not re.fullmatch("F[1-8]", case_id)
            or type(attempt) is not int
            or (not c1 and attempt not in (1, 2))
            or not re.fullmatch("[0-9a-f]{64}", source_hash)
            or (not c1 and attempt == 2 and obj["retests"].get(case_id) != source_hash)
            or (
                g is not None
                and (
                    not g["r1_passed"] or case_id != f"F{len(g['completed_cases']) + 1}"
                )
            )
        ):
            raise ValueError("Unapproved refund case attempt")
        attempts = obj.setdefault("semantic_attempts", {})
        key = case_id + ":" + str(attempt)
        if key in attempts or len(attempts) >= (18 if c1 else 10):
            raise ValueError("Refund case attempt already consumed")
        attempts[key] = {"source_hash": source_hash, "status": "started"}
        save_json(manifest, obj)


def reserve_refund_retrieval(manifest, *, label, source_hash, continuation=None):
    if continuation == "native-return":
        from scripts.refund_native_regression import reserve_retrieval

        return reserve_retrieval(manifest, label=label, source_hash=source_hash)
    if continuation in {
        "round-control",
        "output-repair",
        "output-completeness",
        "context-identity",
        "chain-validation",
    }:
        from scripts.refund_round_regression import reserve_retrieval

        return reserve_retrieval(
            manifest, label=label, source_hash=source_hash, scope=continuation
        )
    if continuation is not None:
        from scripts.refund_semantic_continuation import require, reserve_retrieval

        require(continuation in {"C2", "C3"})
        return reserve_retrieval(
            manifest, label=label, source_hash=source_hash, continuation=continuation
        )
    with manifest_lock(manifest) as obj:
        from scripts.refund_continuation import ATTEMPTS, authorized

        parts = label.split(":")
        c1 = (
            len(parts) == 2
            and parts[1].isdigit()
            and authorized(obj, parts[0], int(parts[1]), source_hash)
        )
        allowed = {f"formal{i}" for i in range(1, 4)} | {
            f"boundary{i}" for i in range(1, 4)
        }
        allowed |= {f"F{i}:{attempt}" for i in range(1, 9) for attempt in (1, 2)}
        if c1:
            allowed |= {f"{c}:{a}" for c, a in ATTEMPTS.items()}
        used = obj.setdefault("retrievals", {})
        if (
            label not in allowed
            or label in used
            or len(used) >= 16
            or not re.fullmatch("[0-9a-f]{64}", source_hash)
        ):
            raise ValueError("Refund retrieval budget exhausted or duplicate")
        if (
            not c1
            and label.endswith(":2")
            and obj["retests"].get(label[:-2]) != source_hash
        ) or (
            c1
            and (
                label not in obj.get("semantic_attempts", {})
                or sum(v.get("continuation") == "C1" for v in used.values()) >= 8
            )
        ):
            raise ValueError("Unapproved retrieval retest")
        used[label] = {"source_hash": source_hash, "outcome": "reserved"}
        if c1:
            used[label]["continuation"] = "C1"
        save_json(manifest, obj)


def finish_refund_retrieval(manifest, *, label, outcome):
    with manifest_lock(manifest) as obj:
        row = obj.get("retrievals", {}).get(label)
        if (
            row is None
            or row["outcome"] != "reserved"
            or outcome not in {"succeeded", "failed", "unknown"}
        ):
            raise ValueError("Invalid retrieval terminal event")
        row["outcome"] = outcome
        save_json(manifest, obj)


def reserve_refund_call(
    manifest, *, purpose, case_id, attempt, source_hash, continuation=None
):
    if continuation == "native-return":
        from scripts.refund_native_regression import reserve_call

        return reserve_call(
            manifest,
            purpose=purpose,
            case_id=case_id,
            attempt=attempt,
            source_hash=source_hash,
        )
    if continuation in {
        "round-control",
        "output-repair",
        "output-completeness",
        "context-identity",
        "chain-validation",
    }:
        from scripts.refund_round_regression import reserve_call

        return reserve_call(
            manifest,
            purpose=purpose,
            case_id=case_id,
            attempt=attempt,
            source_hash=source_hash,
            scope=continuation,
        )
    if continuation is not None:
        from scripts.refund_semantic_continuation import require, reserve_call

        require(continuation in {"C2", "C3"})
        return reserve_call(
            manifest,
            purpose=purpose,
            case_id=case_id,
            attempt=attempt,
            source_hash=source_hash,
            continuation=continuation,
        )
    with manifest_lock(manifest) as obj:
        from scripts.refund_continuation import authorized, grant

        c1 = authorized(obj, case_id, attempt, source_hash)
        g = grant(obj)
        calls = budget_ledger(manifest)
        if (
            purpose not in LIMITS
            or not re.fullmatch("[0-9a-f]{64}", source_hash)
            or type(attempt) is not int
        ):
            raise ValueError("Invalid refund call")
        total = Counter(c["purpose"] for c in calls)
        case = [c for c in calls if c["case_id"] == case_id and c["attempt"] == attempt]
        if any(c["source_hash"] != source_hash for c in case):
            raise ValueError("One attempt cannot mix source candidates")
        used = Counter(c["purpose"] for c in case)
        if len(calls) >= 96 or total[purpose] >= LIMITS[purpose]:
            raise ValueError("Refund purpose budget exhausted")
        if g is not None and (
            not c1
            or sum(c.get("continuation") == "C1" for c in calls) >= 82
            or (case_id == "R1" and not g["transport_started"])
            or (
                case_id != "R1"
                and (
                    not g["r1_passed"]
                    or f"{case_id}:{attempt}" not in obj.get("semantic_attempts", {})
                )
            )
        ):
            raise ValueError("C1 call not authorized")
        if case_id == "R1":
            if (
                (attempt != 1 and not c1)
                or purpose not in R1_LIMITS
                or used[purpose]
                >= (7 if c1 and purpose == "refund_decision" else R1_LIMITS[purpose])
            ):
                raise ValueError("R1 budget exhausted")
        elif re.fullmatch("F[1-8]", case_id):
            if (
                (not c1 and attempt not in (1, 2))
                or (
                    not c1
                    and attempt == 2
                    and obj["retests"].get(case_id) != source_hash
                )
                or purpose == "refund_summary"
                or used[purpose] >= (6 if purpose == "refund_decision" else 1)
            ):
                raise ValueError("R2 attempt budget exhausted")
        else:
            raise ValueError("Unapproved refund case")
        entry = {
            "invocation_id": uuid4().hex,
            "purpose": purpose,
            "case_id": case_id,
            "attempt": attempt,
            "source_hash": source_hash,
            "outcome": "reserved",
            "input_tokens": None,
            "output_tokens": None,
        }
        calls.append(entry)
        if c1:
            entry["continuation"] = "C1"
        _save_ledger(manifest, calls, obj)
        return InvocationReservation(Path(manifest).resolve(), entry["invocation_id"])


def authorize_refund_continuation(manifest, *, approval_commit, source_hash):
    from scripts.refund_continuation import authorize

    return authorize(manifest, approval_commit=approval_commit, source_hash=source_hash)


def reserve_refund_transport_attempt(manifest, *, source_hash):
    from scripts.refund_continuation import start_transport

    return start_transport(manifest, source_hash=source_hash)


def finish_continuation_r1(manifest, *, source_hash, passed):
    from scripts.refund_continuation import finish_r1

    return finish_r1(manifest, source_hash=source_hash, passed=passed)


def halt_refund_continuation(manifest, *, source_hash, case_id):
    from scripts.refund_continuation import halt

    return halt(manifest, source_hash=source_hash, case_id=case_id)


def finish_refund_call(
    reservation,
    *,
    outcome,
    input_tokens=None,
    output_tokens=None,
    error_code=None,
    input_hash=None,
):
    with manifest_lock(reservation.manifest) as obj:
        calls = budget_ledger(reservation.manifest)
        entry = next(
            (c for c in calls if c["invocation_id"] == reservation.invocation_id), None
        )
        if (
            entry is None
            or entry["outcome"] != "reserved"
            or outcome not in {"succeeded", "failed", "unknown"}
        ):
            raise ValueError("Immutable refund call result")
        if any(
            v is not None and (type(v) is not int or v < 0)
            for v in (input_tokens, output_tokens)
        ):
            raise ValueError("Invalid usage")
        entry.update(
            outcome=outcome,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error_code=error_code,
            input_hash=input_hash,
        )
        _save_ledger(reservation.manifest, calls, obj)


def new_report_dir(manifest, target):
    validate_manifest(manifest)
    target = Path(target).resolve()
    workspace = (ROOT / WORKSPACE).resolve()
    if target.exists() or target == workspace or not target.is_relative_to(workspace):
        raise ValueError("Report directory must be new and owned")
    target.mkdir(parents=True, exist_ok=False)
    return target


def source_fingerprints(names):
    return {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(names))
    }


def committed_sources(names):
    actual = source_fingerprints(names)
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    for name, digest in actual.items():
        committed = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", commit + ":" + name]
        )
        if sha256(committed).hexdigest() != digest:
            raise ValueError("Gate source differs from committed candidate")
    return commit, actual


def preservation_value(value):
    """Logical Milvus values, preserving protobuf sequences rather than repr()."""
    import math
    from collections.abc import Mapping, Sequence

    if value is None or type(value) in {str, bool}:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite preservation value")
        return float(value)
    if isinstance(value, Mapping):
        if not all(type(k) in {str, int} for k in value):
            raise TypeError("Unsupported preservation key")
        result = {str(k): preservation_value(v) for k, v in value.items()}
        if len(result) != len(value):
            raise ValueError("Ambiguous preservation key")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [preservation_value(v) for v in value]
    if type(value).__module__.startswith("google.") and type(value).__name__ in {
        "RepeatedScalarContainer",
        "RepeatedCompositeContainer",
    }:
        return [preservation_value(v) for v in value]
    if type(value).__module__.startswith("numpy"):
        import numpy as np

        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            return preservation_value(value.tolist())
        if isinstance(value, np.generic) and value.dtype.kind in "biuf":
            return preservation_value(value.item())
    raise TypeError("Unsupported preservation value")


def snapshot_formal_resources(*, include_milvus=True, layout=None, business_database_url=None):
    from sqlalchemy import inspect

    k, c = configured_urls()
    result = {}
    for label, url in [
        ("knowledge", k),
        ("conversation", c),
        ("business", make_url(business_database_url or Settings().database_url)),
    ]:
        if label == "business" and not Path(url.database).is_file():
            raise ValueError("Formal business file missing")
        engine = _engine(url)
        try:
            columns = (
                {
                    t: [v["name"] for v in inspect(engine).get_columns(t)]
                    for t in inspect(engine).get_table_names()
                }
                if layout is None
                else layout[label]["columns"]
            )
            rows = {}
            with engine.connect() as connection:
                for table, names in columns.items():
                    if table.endswith("alembic_version"):
                        continue
                    values = connection.exec_driver_sql(
                        "SELECT "
                        + ",".join("`" + n + "`" for n in names)
                        + " FROM `"
                        + table
                        + "`"
                    ).all()
                    encoded = sorted(
                        json.dumps(
                            list(row),
                            sort_keys=True,
                            default=lambda v: (
                                v.hex() if isinstance(v, bytes) else str(v)
                            ),
                        )
                        for row in values
                    )
                    rows[table] = {
                        "count": len(encoded),
                        "hash": sha256("\n".join(encoded).encode()).hexdigest(),
                    }
            result[label] = {"columns": columns, "rows": rows}
        finally:
            engine.dispose()
    if include_milvus:
        from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, milvus_client

        with milvus_client(MilvusSettings()) as client:
            names = (
                sorted(client.list_collections(timeout=MILVUS_TIMEOUT_SECONDS))
                if layout is None
                else layout["milvus"]["names"]
            )
            collections = {}
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
                )[0]["count(*)"]
                data = client.query(
                    collection_name=name,
                    filter="",
                    output_fields=["*"],
                    limit=16384,
                    consistency_level="Strong",
                    timeout=MILVUS_TIMEOUT_SECONDS,
                )
                if len(data) != count:
                    raise ValueError("Formal collection snapshot incomplete")
                collections[name] = {
                    "schema_hash": canonical_hash(preservation_value(desc)),
                    "rows_hash": canonical_hash(
                        sorted(preservation_value(data), key=canonical_hash)
                    ),
                    "count": count,
                }
            result["milvus"] = {"names": names, "collections": collections}
    return result
