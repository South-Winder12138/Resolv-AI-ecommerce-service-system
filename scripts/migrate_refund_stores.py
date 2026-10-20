"""Explicit backed-up additive migration. Never imported by application startup."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import base64
import fcntl
import json
import os
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from scripts.migrate_refund_business import (
    file_hash,
    migrate_business,
    validate_refund_schema,
)
from scripts.refund_gate_support import (
    ROOT,
    WORKSPACE,
    _engine,
    configured_urls,
    file_identity,
    owned_refund_resources,
    save_json,
    server_fingerprint,
)
from sqlalchemy import MetaData, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from resolv.core.config import DatabaseSettings
from resolv.db.conversation_base import ConversationBase
from resolv.db.engine import create_db_engine
from resolv.models import Order, Refund
from resolv.models.conversation import Conversation  # noqa: F401 — register old schema
from resolv.models.knowledge_branch import ConversationKnowledgeRun  # noqa: F401
from resolv.models.refund_application import RefundApplication, RefundSubmissionReceipt
from resolv.models.refund_flow import ConversationRefundOffer  # noqa: F401
from resolv.models.understanding import ConversationUnderstanding  # noqa: F401
from resolv.repositories.refund_application import RefundApplicationRepository
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.refund_registration import LOCAL_SOURCE, RefundRegistrationService

REVISION = "20261016_0005"
PREVIOUS = "20261015_0004"
CONFIG = ROOT / "alembic.conversation.ini"
REFUND_TABLES = {
    "conversation_refund_" + n
    for n in ("runs", "offers", "submissions", "request_aliases")
}
BUSINESS_TABLES = {"orders", "refunds", "shipments", "tool_execution_receipts"}
# This migrator targets 0004 -> 0005. Later ORM registrations must not expand
# its prerequisites or authorize additional migrations.
PREVIOUS_TABLES = {
    "conversations", "conversation_runs", "conversation_messages",
    "conversation_actions", "conversation_summaries",
    "conversation_understandings", "conversation_understanding_requests",
    "conversation_knowledge_runs", "conversation_knowledge_review_items",
}


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class StoreTarget:
    engine: object
    business_db: Path
    identity: dict
    offline_test_boundary: bool = False


@contextmanager
def target_context(business_db, ownership=None):
    path = Path(business_db)
    require(
        not path.is_symlink() and path.is_file(),
        "Existing direct business file required",
    )
    path = path.resolve()
    if ownership is not None:
        with owned_refund_resources(ownership) as r:
            require(
                path == Path(r.identity["business_db"]).resolve(),
                "Foreign owned business target",
            )
            yield StoreTarget(
                r.conversation_engine,
                path,
                {
                    "scope": "refund-owned",
                    "ownership": str(Path(ownership).resolve()),
                    "owner": r.identity["owner"],
                    "database": r.identity["conversation_database"],
                    "server_fingerprint": r.identity["server_fingerprint"],
                },
            )
    else:
        configured = make_url(DatabaseSettings().database_url)
        require(
            configured.get_backend_name() == "sqlite"
            and Path(configured.database).resolve()
            == path
            == (ROOT / "data/resolv.db").resolve(),
            "Unknown formal business target",
        )
        _, url = configured_urls()
        engine = _engine(url)
        try:
            yield StoreTarget(
                engine,
                path,
                {
                    "scope": "formal",
                    "ownership": None,
                    "database": url.database,
                    "server_fingerprint": server_fingerprint(engine),
                },
            )
        finally:
            engine.dispose()


def tagged(value):
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "type": "bytes",
            "value": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": str(value)}
    if type(value) in (str, int, float, bool):
        return {"type": type(value).__name__, "value": value}
    raise TypeError("Unsupported backup value type")


def _quoted(connection, name):
    return connection.dialect.identifier_preparer.quote_identifier(name)


def _snapshot_connection(connection, columns=None):
    schema = inspect(connection)
    selected = columns or {
        t: [c["name"] for c in schema.get_columns(t)]
        for t in sorted(schema.get_table_names())
    }
    result = {}
    for table, names in selected.items():
        rows = connection.exec_driver_sql(
            "SELECT "
            + ",".join(_quoted(connection, c) for c in names)
            + " FROM "
            + _quoted(connection, table)
        ).all()
        encoded = [[tagged(v) for v in row] for row in rows]
        result[table] = sorted(encoded, key=canonical_hash)
    return {"columns": selected, "rows": result}


def business_snapshot(path, columns=None):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "Missing direct business target")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as c:
        c.execute("BEGIN")
        require(
            c.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
            "Business integrity failed",
        )
        require(
            c.execute("PRAGMA foreign_key_check").fetchall() == [],
            "Business foreign keys failed",
        )
        selected = columns or {
            r[0]: [
                col[1]
                for col in c.execute(
                    'PRAGMA table_info("' + r[0].replace('"', '""') + '")'
                )
            ]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        }
        rows = {
            table: sorted(
                [
                    [tagged(v) for v in row]
                    for row in c.execute(
                        "SELECT "
                        + ",".join('"' + n.replace('"', '""') + '"' for n in names)
                        + ' FROM "'
                        + table.replace('"', '""')
                        + '"'
                    )
                ],
                key=canonical_hash,
            )
            for table, names in selected.items()
        }
        schema = {
            r[0]: r[1]
            for r in c.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        }
        return {"columns": selected, "rows": rows, "schema": schema}


def conversation_snapshot(target, columns=None):
    require(
        target.engine.dialect.name == "mysql" or target.offline_test_boundary,
        "Real MySQL required outside isolated tests",
    )
    with target.engine.connect() as c:
        if c.dialect.name == "mysql":
            c = c.execution_options(isolation_level="REPEATABLE READ")
            c.exec_driver_sql("SET TRANSACTION READ ONLY")
            c.exec_driver_sql("START TRANSACTION WITH CONSISTENT SNAPSHOT")
        value = _snapshot_connection(c, columns)
        versions = c.execute(
            text("SELECT version_num FROM conversation_alembic_version")
        ).all()
        require(len(versions) == 1, "Exactly one known conversation revision required")
        version = versions[0][0]
        schemas = {}
        for table in value["columns"]:
            if c.dialect.name == "mysql":
                schemas[table] = c.exec_driver_sql(
                    "SHOW CREATE TABLE " + _quoted(c, table)
                ).one()[1]
            else:
                schemas[table] = c.scalar(
                    text(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"
                    ),
                    {"name": table},
                )
        nonterminal = c.execute(
            text(
                "SELECT status,COUNT(*) FROM conversation_runs WHERE status NOT IN ('completed','failed') GROUP BY status"
            )
        ).all()
        active = c.scalar(
            text(
                "SELECT COUNT(*) FROM conversation_runs WHERE status IN ('running','finalizing') AND execution_owner_id IS NOT NULL AND lease_expires_at>:now"
            ),
            {"now": datetime.now(UTC).replace(tzinfo=None)},
        )
        c.rollback()
    return value | {
        "revision": version,
        "schema": schemas,
        "nonterminal": {r[0]: r[1] for r in nonterminal},
        "active_leases": active,
    }


def _config(connection):
    cfg = Config(str(CONFIG))
    cfg.attributes["connection"] = connection
    return cfg


def upgrade_conversation(target, revision):
    require(revision == REVISION, "Exact approved revision required")
    with target.engine.begin() as c:
        command.upgrade(_config(c), revision)


def schema_check(target, business_db):
    with target.engine.connect() as c:
        version = c.scalar(text("SELECT version_num FROM conversation_alembic_version"))
        tables = set(inspect(c).get_table_names())
        conversation_ready = version == REVISION and REFUND_TABLES <= tables
        if conversation_ready:
            approved = MetaData()
            for name in sorted(PREVIOUS_TABLES | REFUND_TABLES):
                ConversationBase.metadata.tables[name].to_metadata(approved)
            context = MigrationContext.configure(c, opts={
                "compare_type": True,
                "include_name": lambda name, kind, parents: kind != "table" or name in approved.tables,
            })
            require(not compare_metadata(context, approved), "Approved conversation schema differs")
    state = business_snapshot(business_db)
    business_ready = {"refund_applications", "refund_submission_receipts"} <= set(
        state["columns"]
    )
    if business_ready:
        engine = create_db_engine("sqlite:///" + str(Path(business_db).resolve()))
        try:
            with engine.connect() as c:
                validate_refund_schema(c)
            with Session(engine) as session:
                registration = RefundRegistrationService(session)
                for old, order in session.execute(
                    select(Refund, Order).join(Order, Order.id == Refund.order_id)
                ):
                    require(
                        registration._consistent(LOCAL_SOURCE, order.order_no)
                        is not None,
                        "Missing legacy mapping",
                    )
                repo = RefundApplicationRepository(session)
                for app in session.scalars(select(RefundApplication)):
                    registration._consistent(app.order_source, app.order_id)
                    repo.record(app)
                for receipt in session.scalars(select(RefundSubmissionReceipt)):
                    repo.validated_receipt(receipt)
        finally:
            engine.dispose()
    return {
        "ready": conversation_ready and business_ready,
        "conversation_revision": version,
        "business_ready": business_ready,
    }


def _backup_location(path):
    path = Path(path).resolve()
    require(
        any(
            path.is_relative_to((ROOT / base).resolve())
            and path != (ROOT / base).resolve()
            for base in (WORKSPACE, "data/refund-backups")
        ),
        "Backup directory must be task-private",
    )
    return path


def preflight_refund_stores(
    *,
    business_db,
    expected_server_fingerprint,
    expected_business_sha256,
    backup_dir,
    ownership=None,
):
    with target_context(business_db, ownership) as target:
        require(
            target.identity["server_fingerprint"] == expected_server_fingerprint
            and file_hash(business_db) == expected_business_sha256,
            "Target fingerprint mismatch",
        )
        business = business_snapshot(business_db)
        require(
            BUSINESS_TABLES <= set(business["columns"]),
            "Legacy business tables missing",
        )
        conversation = conversation_snapshot(target)
        require(
            conversation["revision"] in (PREVIOUS, REVISION),
            "Unknown conversation revision",
        )
        old_tables = PREVIOUS_TABLES
        require(
            old_tables <= set(conversation["columns"]),
            "Legacy conversation tables missing",
        )
        for table in old_tables:
            expected = set(ConversationBase.metadata.tables[table].columns.keys())
            if table == "conversation_runs" and conversation["revision"] == PREVIOUS:
                expected.discard("refund_config")
            require(
                expected <= set(conversation["columns"][table]),
                "Legacy conversation columns missing",
            )
        if conversation["revision"] == PREVIOUS:
            require(
                not REFUND_TABLES.intersection(conversation["columns"])
                and "refund_config" not in conversation["columns"]["conversation_runs"],
                "Incomplete prior DDL requires inspection",
            )
        else:
            schema_check(target, business_db)
        backup_dir = _backup_location(backup_dir)
        require(not backup_dir.exists(), "Backup directory already exists")
        backup_dir.mkdir(parents=True, mode=0o700)
        mysql_backup = backup_dir / "conversation-logical.json"
        save_json(
            mysql_backup,
            {
                "format": "refund-mysql-logical-v1",
                "identity": target.identity,
                "snapshot": conversation,
            },
        )
        sqlite_backup = backup_dir / "business.db"
        sqlite_backup.touch(mode=0o600, exist_ok=False)
        with (
            closing(
                sqlite3.connect(
                    Path(business_db).resolve().as_uri() + "?mode=ro", uri=True
                )
            ) as source,
            closing(sqlite3.connect(sqlite_backup)) as dest,
        ):
            source.backup(dest)
        sqlite_backup.chmod(0o600)
        with sqlite_backup.open("rb") as f:
            os.fsync(f.fileno())
        require(
            business_snapshot(sqlite_backup) == business
            and business_snapshot(business_db) == business,
            "Business changed during backup including WAL",
        )
        require(
            conversation_snapshot(target) == conversation,
            "Conversation changed during backup",
        )
        result = {
            "format": "refund-store-preflight-v1",
            "status": "ready",
            "business_db": str(Path(business_db).resolve()),
            "business_identity": file_identity(business_db),
            "identity": target.identity,
            "business": business,
            "conversation": conversation,
            "business_main_sha256": expected_business_sha256,
            "backups": {
                "business": {
                    "path": str(sqlite_backup),
                    "sha256": file_hash(sqlite_backup),
                },
                "conversation": {
                    "path": str(mysql_backup),
                    "sha256": file_hash(mysql_backup),
                },
            },
            "receipt_path": str(backup_dir / "migration-receipt.json"),
        }
        result["fingerprint"] = canonical_hash(result)
        save_json(backup_dir / "preflight.json", result)
        return result


def _validate_preflight(preflight):
    require(
        preflight.get("format") == "refund-store-preflight-v1"
        and preflight.get("status") == "ready"
        and preflight.get("fingerprint")
        == canonical_hash({k: v for k, v in preflight.items() if k != "fingerprint"}),
        "Invalid preflight receipt",
    )
    directory = _backup_location(Path(preflight["receipt_path"]).parent)
    require(
        Path(preflight["receipt_path"]) == directory / "migration-receipt.json",
        "Foreign migration receipt path",
    )
    for label, name in (
        ("business", "business.db"),
        ("conversation", "conversation-logical.json"),
    ):
        info = preflight["backups"][label]
        path = Path(info["path"])
        require(
            path == directory / name
            and not path.is_symlink()
            and path.is_file()
            and file_hash(path) == info["sha256"]
            and path.stat().st_mode & 0o777 == 0o600,
            "Backup changed or not private",
        )
    require(
        business_snapshot(directory / "business.db") == preflight["business"],
        "Business backup content mismatch",
    )
    backup = json.loads((directory / "conversation-logical.json").read_text())
    require(
        backup.get("snapshot") == preflight["conversation"]
        and backup.get("identity") == preflight["identity"],
        "Conversation backup content mismatch",
    )
    return directory


def migrate_refund_stores(preflight, *, conversation_revision=REVISION):
    require(conversation_revision == REVISION, "Exact approved revision required")
    directory = _validate_preflight(preflight)
    with (directory / "apply.lock").open("a+") as lock:
        os.chmod(directory / "apply.lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _apply_locked(preflight, directory)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def migration_lock(target):
    if target.offline_test_boundary:
        from types import SimpleNamespace

        yield SimpleNamespace(assert_owned=lambda: None)
    else:
        from resolv.services.refund_submission import mysql_submission_lock

        key = "migrate-" + canonical_hash(target.identity)[:40]
        with mysql_submission_lock(target.engine, key) as lock:
            yield lock


def _original_rows(p, target, path):
    require(
        business_snapshot(path, columns=p["business"]["columns"])["rows"]
        == p["business"]["rows"],
        "Original business rows changed",
    )
    original = {
        k: v
        for k, v in p["conversation"]["rows"].items()
        if k != "conversation_alembic_version"
    }
    observed = conversation_snapshot(target, columns=p["conversation"]["columns"])[
        "rows"
    ]
    require(
        {k: v for k, v in observed.items() if k != "conversation_alembic_version"}
        == original,
        "Original conversation rows changed",
    )


def _apply_locked(p, directory):
    with (
        target_context(
            Path(p["business_db"]), p["identity"].get("ownership")
        ) as target,
        migration_lock(target) as lock,
    ):
        lock.assert_owned()
        return _apply_target(p, directory, target, lock)


def _apply_target(p, directory, target, lock):
    path = Path(p["business_db"])
    receipt_path = Path(p["receipt_path"])
    require(
        target.identity == p["identity"]
        and file_identity(path) == p["business_identity"],
        "Target identity changed",
    )
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        require(
            saved.get("status") == "completed"
            and saved.get("preflight_hash") == p["fingerprint"],
            "Incomplete prior migration requires inspection",
        )
        require(
            saved.get("business_post_hash") == canonical_hash(business_snapshot(path))
            and saved.get("conversation_post_hash")
            == canonical_hash(conversation_snapshot(target)),
            "Completed migration target changed",
        )
        require(schema_check(target, path)["ready"], "Completed schema/mapping invalid")
        apply_backup = Path(saved.get("business_migration", {}).get("backup_path", ""))
        require(
            apply_backup == directory / "business-apply.db"
            and not apply_backup.is_symlink()
            and apply_backup.is_file()
            and file_hash(apply_backup) == saved["business_migration"]["backup_hash"]
            and apply_backup.stat().st_mode & 0o777 == 0o600,
            "Completed SQLite backup changed",
        )
        require(
            business_snapshot(apply_backup) == p["business"],
            "Completed backup content changed",
        )
        _original_rows(p, target, path)
        return saved
    before = conversation_snapshot(target)
    require(
        before == p["conversation"] and business_snapshot(path) == p["business"],
        "Target changed since preflight",
    )
    require(before["active_leases"] == 0, "Active writers require explicit quiescence")
    receipt = {
        "format": "refund-store-migration-v1",
        "status": "started",
        "preflight_hash": p["fingerprint"],
        "conversation_status": "not_started",
        "business_status": "not_started",
        "atomic_across_stores": False,
    }
    save_json(receipt_path, receipt)
    try:
        receipt["conversation_status"] = "started"
        save_json(receipt_path, receipt)
        lock.assert_owned()
        upgrade_conversation(target, REVISION)
        lock.assert_owned()
        receipt["conversation_status"] = "completed"
        save_json(receipt_path, receipt)
        # Check every original table using its original column list.
        after = conversation_snapshot(target, columns=before["columns"])
        original = {
            k: v
            for k, v in before["rows"].items()
            if k != "conversation_alembic_version"
        }
        require(
            {
                k: v
                for k, v in after["rows"].items()
                if k != "conversation_alembic_version"
            }
            == original,
            "Legacy conversation rows changed",
        )
        receipt["conversation_status"] = "completed"
        receipt["business_status"] = "started"
        save_json(receipt_path, receipt)
        require(
            business_snapshot(path) == p["business"],
            "Business changed before SQLite migration",
        )
        migration = migrate_business(
            path,
            expected_sha256=file_hash(path),
            backup=directory / "business-apply.db",
        )
        receipt.update(
            business_status="completed",
            business_migration=migration.model_dump(mode="json"),
        )
        save_json(receipt_path, receipt)
        lock.assert_owned()
        require(
            business_snapshot(path, columns=p["business"]["columns"])["rows"]
            == p["business"]["rows"],
            "Legacy business rows changed",
        )
        require(schema_check(target, path)["ready"], "Migrated schema/mapping invalid")
        receipt.update(
            status="completed",
            business_status="completed",
            business_migration=migration.model_dump(mode="json"),
            business_post_hash=canonical_hash(business_snapshot(path)),
            conversation_post_hash=canonical_hash(conversation_snapshot(target)),
        )
    except Exception as exc:  # noqa: BLE001 — never publish driver secrets or claim DDL rollback
        for store in ("conversation", "business"):
            if receipt[store + "_status"] == "started":
                receipt[store + "_status"] = "unknown"
        try:
            receipt["observed_conversation_revision"] = conversation_snapshot(target)[
                "revision"
            ]
        except Exception:  # noqa: BLE001 — unknown means no proof, never rollback
            receipt["observed_conversation_revision"] = None
        receipt.update(
            status="failed",
            error_type=type(exc).__name__,
            error_code="refund_store_migration_failed",
        )
        save_json(receipt_path, receipt)
        raise ValueError(
            "Migration incomplete; inspect independent receipts before any further action"
        ) from None
    save_json(receipt_path, receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "apply"), required=True)
    p.add_argument("--business-db", type=Path)
    p.add_argument("--expected-server-fingerprint")
    p.add_argument("--expected-business-sha256")
    p.add_argument("--backup-dir", type=Path)
    p.add_argument("--ownership", type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args(argv)
    require(not a.report.exists(), "Report already exists")
    try:
        if a.mode == "preflight":
            require(
                a.business_db
                and a.expected_server_fingerprint
                and a.expected_business_sha256
                and a.backup_dir
                and not a.preflight,
                "Explicit preflight identity required",
            )
            result = preflight_refund_stores(
                business_db=a.business_db,
                expected_server_fingerprint=a.expected_server_fingerprint,
                expected_business_sha256=a.expected_business_sha256,
                backup_dir=a.backup_dir,
                ownership=a.ownership,
            )
        else:
            require(
                a.preflight
                and not any(
                    (
                        a.business_db,
                        a.expected_server_fingerprint,
                        a.expected_business_sha256,
                        a.backup_dir,
                        a.ownership,
                    )
                ),
                "Apply consumes only retained preflight",
            )
            result = migrate_refund_stores(json.loads(a.preflight.read_text()))
        save_json(a.report, result)
        print(json.dumps({"status": result["status"], "report": str(a.report)}))
        return 0
    except Exception as exc:  # noqa: BLE001 — safe CLI failure evidence only
        save_json(
            a.report,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "code": "refund_store_migration_failed",
            },
        )
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
