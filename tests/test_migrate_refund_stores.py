"""Offline dual-store orchestration: real SQLite, explicit simulated MySQL boundary."""

import importlib
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from resolv.db.engine import create_db_engine
from tests.test_refund_business_migration import digest, old_database

ROOT = Path(__file__).resolve().parents[1]


def runner():
    assert importlib.util.find_spec("scripts.migrate_refund_stores"), (
        "missing explicit dual-store migrator"
    )
    return importlib.import_module("scripts.migrate_refund_stores")


@pytest.fixture
def stores(tmp_path, monkeypatch):
    from types import SimpleNamespace

    m = runner()
    business = tmp_path / "business.db"
    old_database(business)
    engine = create_db_engine("sqlite:///" + str(tmp_path / "conversation.db"))
    with engine.begin() as c:
        config = Config(str(ROOT / "alembic.conversation.ini"))
        config.attributes["connection"] = c
        command.upgrade(config, "20261015_0004")
        c.execute(text("CREATE TABLE untouched(payload TEXT)"))
        c.execute(
            text("INSERT INTO untouched VALUES (:v)"),
            {"v": '{"huge":9007199254740993}'},
        )
    monkeypatch.setattr(m, "ROOT", tmp_path)
    target = m.StoreTarget(
        engine=engine,
        business_db=business,
        identity={
            "database": "offline-owned",
            "server_fingerprint": "e" * 64,
            "scope": "offline-boundary",
        },
        offline_test_boundary=True,
    )

    @contextmanager
    def selected(path, ownership=None):
        if Path(path).resolve() != business.resolve():
            raise ValueError("Unknown target")
        yield target

    monkeypatch.setattr(m, "target_context", selected)
    yield SimpleNamespace(
        module=m,
        business=business,
        target=target,
        engine=engine,
        backup=tmp_path / m.WORKSPACE / "migration-backup",
    )
    engine.dispose()


def prepare(h):
    return h.module.preflight_refund_stores(
        business_db=h.business,
        expected_server_fingerprint="e" * 64,
        expected_business_sha256=digest(h.business),
        backup_dir=h.backup,
    )


def test_explicit_migration_identity_backup_and_idempotency(stores):
    h = stores
    m = h.module
    before = m.business_snapshot(h.business)
    p = prepare(h)
    assert p["status"] == "ready"
    assert m.business_snapshot(h.business) == before
    first = m.migrate_refund_stores(p)
    assert first["status"] == "completed"
    assert m.migrate_refund_stores(p) == first
    assert (
        m.business_snapshot(h.business, columns=before["columns"])["rows"]
        == before["rows"]
    )
    with h.engine.connect() as c:
        assert (
            c.scalar(text("SELECT payload FROM untouched"))
            == '{"huge":9007199254740993}'
        )
    for info in p["backups"].values():
        assert (Path(info["path"]).stat().st_mode & 0o777) == 0o600


def test_refund_migrator_keeps_0005_scope_when_logistics_is_registered(stores):
    from resolv.models.logistics import ConversationLogisticsAttempt

    assert "conversation_logistics_attempts" in ConversationLogisticsAttempt.metadata.tables
    receipt = stores.module.migrate_refund_stores(prepare(stores))
    assert receipt["status"] == "completed"
    with stores.engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM conversation_alembic_version")) == "20261016_0005"
        assert "conversation_logistics_attempts" not in inspect(connection).get_table_names()


def test_schema_check_never_migrates_or_seeds(stores):
    h = stores
    m = h.module
    with h.engine.connect() as c:
        before = set(inspect(c).get_table_names())
    status = m.schema_check(h.target, h.business)
    assert status["ready"] is False
    with h.engine.connect() as c:
        assert set(inspect(c).get_table_names()) == before
    assert set(m.business_snapshot(h.business)["columns"]) == {
        "orders",
        "refunds",
        "shipments",
        "tool_execution_receipts",
    }


@pytest.mark.parametrize(
    "change",
    [
        "wrong_server",
        "wrong_hash",
        "unknown_target",
        "backup_tamper",
        "target_change",
        "unknown_revision",
        "partial_receipt",
    ],
)
def test_migration_refuses_unproven_or_changed_target(stores, change):
    h = stores
    m = h.module
    if change in {"wrong_server", "wrong_hash", "unknown_target"}:
        with pytest.raises(ValueError):
            m.preflight_refund_stores(
                business_db=h.business
                if change != "unknown_target"
                else h.business.with_name("other.db"),
                expected_server_fingerprint=("f" if change == "wrong_server" else "e")
                * 64,
                expected_business_sha256="a" * 64
                if change == "wrong_hash"
                else digest(h.business),
                backup_dir=h.backup,
            )
        assert not h.backup.exists()
        return
    p = prepare(h)
    if change == "backup_tamper":
        Path(p["backups"]["business"]["path"]).write_bytes(b"tampered")
    elif change == "target_change":
        import sqlite3

        with sqlite3.connect(h.business) as c:
            c.execute("UPDATE refunds SET reason='changed'")
    elif change == "partial_receipt":
        m.save_json(Path(p["receipt_path"]), {"status": "started"})
    with pytest.raises(ValueError):
        m.migrate_refund_stores(
            p,
            conversation_revision="head"
            if change == "unknown_revision"
            else "20261016_0005",
        )


def test_mysql_acknowledgement_loss_records_partial_progress(stores, monkeypatch):
    import json

    h = stores
    m = h.module
    p = prepare(h)
    original = m.upgrade_conversation

    def lost_ack(target, revision):
        original(target, revision)
        raise OSError("lost DDL acknowledgement")

    monkeypatch.setattr(m, "upgrade_conversation", lost_ack)
    with pytest.raises(ValueError):
        m.migrate_refund_stores(p)
    receipt = json.loads(Path(p["receipt_path"]).read_text())
    assert receipt["status"] == "failed" and receipt["conversation_status"] == "unknown"
    assert receipt["observed_conversation_revision"] == "20261016_0005"
    assert receipt["business_status"] == "not_started"
    assert "refund_applications" not in m.business_snapshot(h.business)["columns"]
    with pytest.raises(ValueError):
        m.migrate_refund_stores(p)


def test_completed_receipt_rechecks_second_backup_and_original_rows(stores):
    h = stores
    m = h.module
    p = prepare(h)
    receipt = m.migrate_refund_stores(p)
    Path(receipt["business_migration"]["backup_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        m.migrate_refund_stores(p)


def test_wal_changes_cannot_hide_behind_unchanged_main_file(stores):
    import sqlite3

    h = stores
    m = h.module
    with sqlite3.connect(h.business) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("UPDATE orders SET status='shipped'")
        live.commit()
        p = prepare(h)
        before = digest(h.business)
        live.execute("UPDATE refunds SET reason='WAL changed'")
        live.commit()
        assert digest(h.business) == before
        with pytest.raises(ValueError):
            m.migrate_refund_stores(p)


def test_wal_checkpoint_without_logical_change_preserves_apply(stores):
    import sqlite3

    h = stores
    m = h.module
    with sqlite3.connect(h.business) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("UPDATE orders SET status='shipped'")
        live.commit()
        p = prepare(h)
        live.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        first = m.migrate_refund_stores(p)
        assert first["status"] == "completed" and m.migrate_refund_stores(p) == first


def test_sqlite_commit_remains_recorded_if_later_check_fails(stores, monkeypatch):
    import json

    h = stores
    m = h.module
    p = prepare(h)

    def failed_check(*args):
        raise OSError("post-commit connection lost")

    monkeypatch.setattr(m, "schema_check", failed_check)
    with pytest.raises(ValueError):
        m.migrate_refund_stores(p)
    receipt = json.loads(Path(p["receipt_path"]).read_text())
    assert receipt["status"] == "failed" and receipt["business_status"] == "completed"
    assert Path(receipt["business_migration"]["backup_path"]).is_file()
    assert "refund_applications" in m.business_snapshot(h.business)["columns"]


def test_completed_post_hash_cannot_hide_changed_original_rows(stores):
    import sqlite3

    h = stores
    m = h.module
    p = prepare(h)
    receipt = m.migrate_refund_stores(p)
    with sqlite3.connect(h.business) as c:
        c.execute("UPDATE refunds SET reason='altered'")
        c.execute("UPDATE refund_applications SET reason='altered'")
    receipt["business_post_hash"] = m.canonical_hash(m.business_snapshot(h.business))
    m.save_json(Path(p["receipt_path"]), receipt)
    with pytest.raises(ValueError):
        m.migrate_refund_stores(p)


@pytest.mark.parametrize(
    "change", ["active_writer", "unknown_revision", "missing_legacy_table"]
)
def test_preflight_reports_nonterminal_and_apply_refuses_active_writer(
    stores, monkeypatch, change
):
    h = stores
    m = h.module
    if change == "active_writer":
        original = m.conversation_snapshot
        monkeypatch.setattr(
            m,
            "conversation_snapshot",
            lambda *a, **kw: (
                original(*a, **kw) | {"active_leases": 1, "nonterminal": {"running": 1}}
            ),
        )
        p = prepare(h)
        assert p["conversation"]["nonterminal"] == {"running": 1}
        with pytest.raises(ValueError):
            m.migrate_refund_stores(p)
        with h.engine.connect() as c:
            assert (
                c.scalar(text("SELECT version_num FROM conversation_alembic_version"))
                == "20261015_0004"
            )
    else:
        if change == "unknown_revision":
            with h.engine.begin() as c:
                c.execute(
                    text(
                        "UPDATE conversation_alembic_version SET version_num='unknown'"
                    )
                )
        else:
            import sqlite3

            with sqlite3.connect(h.business) as c:
                c.execute("DROP TABLE shipments")
        with pytest.raises(ValueError):
            prepare(h)
        assert not h.backup.exists()


def test_preflight_refuses_old_revision_with_missing_known_column(stores):
    h = stores
    with h.engine.begin() as c:
        c.execute(text("ALTER TABLE conversation_messages DROP COLUMN text"))
    with pytest.raises(ValueError):
        prepare(h)
    assert not h.backup.exists()
