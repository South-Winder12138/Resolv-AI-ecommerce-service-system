"""Explicit additive SQLite refund migration; never called by application startup."""

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from sqlalchemy import UniqueConstraint, inspect
from sqlalchemy.orm import Session

from resolv.db.engine import create_db_engine
from resolv.models.refund_application import RefundApplication, RefundSubmissionReceipt
from resolv.schemas.refund_submission import MigrationReceipt
from resolv.services.refund_registration import RefundRegistrationService


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(connection, tables):
    return {table: sorted(repr(tuple(row)) for row in connection.exec_driver_sql('SELECT * FROM "' + table + '"'))
            for table in tables}


def validate_refund_schema(connection):
    schema = inspect(connection)
    for model in (RefundApplication, RefundSubmissionReceipt):
        table = model.__table__
        if {c["name"] for c in schema.get_columns(table.name)} != set(table.columns.keys()):
            raise ValueError("Incompatible refund schema columns")
        if schema.get_pk_constraint(table.name)["constrained_columns"] != [c.name for c in table.primary_key]:
            raise ValueError("Incompatible refund schema primary key")
        unique = {tuple(c["column_names"]) for c in schema.get_unique_constraints(table.name)}
        required = {tuple(c.columns.keys()) for c in table.constraints if isinstance(c, UniqueConstraint)}
        if not required <= unique:
            raise ValueError("Incompatible refund schema unique constraints")
        foreign = {(tuple(f["constrained_columns"]), f["referred_table"], tuple(f["referred_columns"]))
                   for f in schema.get_foreign_keys(table.name)}
        expected = {(tuple(c.name for c in fk.columns), fk.referred_table.name,
                     tuple(e.column.name for e in fk.elements)) for fk in table.foreign_key_constraints}
        if foreign != expected:
            raise ValueError("Incompatible refund schema foreign keys")


def migrate_business(path: Path, *, expected_sha256: str, backup: Path) -> MigrationReceipt:
    path, backup = Path(path).resolve(), Path(backup).resolve()
    if not path.is_file() or file_hash(path) != expected_sha256 or backup.exists() or path == backup:
        raise ValueError("Business target/backup identity mismatch")
    # The backup is a consistent SQLite snapshot, including WAL, not a copy of
    # a possibly stale main file. Its actual bytes receive their own checksum.
    backup.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
        required = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"orders", "shipments", "refunds", "tool_execution_receipts"} <= required:
            raise ValueError("Explicit legacy business schema required")
        backup.touch(mode=0o600, exist_ok=False)
        with closing(sqlite3.connect(backup)) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Invalid backup")
            preserved_tables = tuple(sorted(required & {"orders", "shipments", "refunds", "tool_execution_receipts", "refund_applications", "refund_submission_receipts"}))
            backup_rows = {table: sorted(repr(tuple(row)) for row in target.execute('SELECT * FROM "' + table + '"'))
                           for table in preserved_tables}
    backup.chmod(0o600)
    engine = create_db_engine("sqlite:///" + str(path))
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                if file_hash(path) != expected_sha256:
                    raise ValueError("Business target changed before migration")
                if _rows(connection, preserved_tables) != backup_rows:
                    raise ValueError("Business target changed after backup (including WAL)")
                before = _rows(connection, ("orders", "shipments", "refunds", "tool_execution_receipts"))
                if connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall():
                    raise ValueError("Legacy foreign key violation")
                for model in (RefundApplication, RefundSubmissionReceipt):
                    model.__table__.create(connection, checkfirst=True)
                validate_refund_schema(connection)
                with Session(bind=connection) as session:
                    RefundRegistrationService(session).backfill_legacy()
                    session.flush()
                if before != _rows(connection, before):
                    raise ValueError("Legacy rows changed")
                count = connection.exec_driver_sql("SELECT COUNT(*) FROM refund_applications").scalar_one()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return MigrationReceipt(path=str(path), backup_path=str(backup), backup_hash=file_hash(backup),
                                before_hash=expected_sha256, after_hash=file_hash(path),
                                legacy_count=len(before["refunds"]), application_count=count)
    finally:
        engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--business-db", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = migrate_business(args.business_db, expected_sha256=args.expected_sha256, backup=args.backup)
    print(json.dumps(receipt.model_dump(mode="json")))


if __name__ == "__main__":
    sys.exit(main())
