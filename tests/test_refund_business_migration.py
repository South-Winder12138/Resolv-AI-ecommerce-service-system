"""Explicit SQLite upgrade preserves original rows, relations and receipts."""

import hashlib
import sqlite3

import pytest


def old_database(path):
    with sqlite3.connect(path) as conn:
        conn.executescript("""
CREATE TABLE orders(id INTEGER PRIMARY KEY,order_no VARCHAR(32) UNIQUE NOT NULL,status VARCHAR(32) NOT NULL,created_at DATETIME NOT NULL);
CREATE TABLE shipments(id INTEGER PRIMARY KEY,order_id INTEGER REFERENCES orders(id));
CREATE TABLE refunds(id INTEGER PRIMARY KEY,order_id INTEGER NOT NULL UNIQUE REFERENCES orders(id),status VARCHAR(32) NOT NULL,reason VARCHAR(500) NOT NULL,created_at DATETIME NOT NULL,updated_at DATETIME NOT NULL);
CREATE TABLE tool_execution_receipts(action_id TEXT PRIMARY KEY,result JSON);
INSERT INTO orders VALUES(7,'OLD001','delivered','2026-01-01 00:00:00');
INSERT INTO refunds VALUES(3,7,'rejected','原申请原因','2026-01-02 00:00:00','2026-01-02 00:00:00');
INSERT INTO tool_execution_receipts VALUES('old-action','{"status":"pending"}');
""")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_backfill_is_unknown_and_idempotent(tmp_path):
    from scripts.migrate_refund_business import migrate_business

    path, backup = tmp_path / "business.db", tmp_path / "backup.db"
    old_database(path)
    old_hash = digest(path)
    receipt = migrate_business(path, expected_sha256=old_hash, backup=backup)
    assert digest(backup) == receipt.backup_hash and receipt.legacy_count == 1
    with sqlite3.connect(backup) as saved:
        assert saved.execute("SELECT id,order_id,status,reason FROM refunds").fetchall() == [(3, 7, "rejected", "原申请原因")]
    with sqlite3.connect(path) as conn:
        before = conn.execute("SELECT * FROM refunds").fetchall()
        app = conn.execute("SELECT application_id,origin,money,subject_ref,legacy_refund_id FROM refund_applications").fetchone()
        assert app[1:] == ("legacy", None, None, 3)
        assert conn.execute("SELECT * FROM tool_execution_receipts").fetchall() == [('old-action', '{"status":"pending"}')]
    migrate_business(path, expected_sha256=digest(path), backup=tmp_path / "second-backup.db")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM refunds").fetchall() == before
        assert conn.execute("SELECT COUNT(*) FROM refund_applications").fetchone() == (1,)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_refuses_wrong_hash_and_does_not_create_target(tmp_path):
    from scripts.migrate_refund_business import migrate_business

    path = tmp_path / "missing.db"
    with pytest.raises(ValueError):
        migrate_business(path, expected_sha256="a" * 64, backup=tmp_path / "backup.db")
    assert not path.exists()
    old_database(path)
    before = digest(path)
    with pytest.raises(ValueError):
        migrate_business(path, expected_sha256="a" * 64, backup=tmp_path / "backup.db")
    assert digest(path) == before


def test_migration_rejects_missing_unique_boundary(tmp_path):
    from scripts.migrate_refund_business import migrate_business

    path = tmp_path / "bad-schema.db"
    old_database(path)
    migrate_business(path, expected_sha256=digest(path), backup=tmp_path / "before.db")
    with sqlite3.connect(path) as conn:
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name='refund_applications'").fetchone()[0]
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE refund_applications")
        ddl = ddl.replace('CONSTRAINT uq_refund_application_order UNIQUE (order_source, order_id),', '')
        conn.execute(ddl)
    with pytest.raises(ValueError, match="Incompatible refund schema"):
        migrate_business(path, expected_sha256=digest(path), backup=tmp_path / "bad-before.db")


def test_migration_failure_rolls_back_new_tables(tmp_path, monkeypatch):
    from scripts.migrate_refund_business import migrate_business

    from resolv.services.refund_registration import RefundRegistrationService

    path = tmp_path / "rollback.db"
    old_database(path)

    def fail(self):
        raise RuntimeError("injected backfill failure")

    monkeypatch.setattr(RefundRegistrationService, "backfill_legacy", fail)
    with pytest.raises(RuntimeError, match="backfill failure"):
        migrate_business(path, expected_sha256=digest(path), backup=tmp_path / "rollback-backup.db")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'refund_%' AND name != 'refunds'").fetchall() == []
        assert conn.execute("SELECT status,reason FROM refunds").fetchall() == [("rejected", "原申请原因")]


def test_legacy_rejected_application_cannot_be_overwritten_after_migration(tmp_path):
    from scripts.migrate_refund_business import migrate_business

    from resolv.db.engine import create_db_engine, create_session_factory
    from tests.test_refund_registration import command, register

    path = tmp_path / "rejected.db"
    old_database(path)
    migrate_business(path, expected_sha256=digest(path), backup=tmp_path / "rejected-backup.db")
    engine = create_db_engine("sqlite:///" + str(path))
    try:
        result = register(create_session_factory(engine), command(namespace="local-business-v1", order_id="OLD001"))
        assert not result.created and result.status == "rejected" and result.reason == "原申请原因"
        assert result.money is None
    finally:
        engine.dispose()


def test_migration_rejects_wal_change_after_consistent_backup(tmp_path, monkeypatch):
    import scripts.migrate_refund_business as migration

    path = tmp_path / "wal.db"
    old_database(path)
    writer = sqlite3.connect(path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        original = migration.create_db_engine

        def changed(url):
            writer.execute("UPDATE refunds SET reason='concurrent update' WHERE id=3")
            writer.commit()
            return original(url)

        monkeypatch.setattr(migration, "create_db_engine", changed)
        with pytest.raises(ValueError, match="changed"):
            migration.migrate_business(path, expected_sha256=digest(path), backup=tmp_path / "wal-backup.db")
    finally:
        writer.close()
