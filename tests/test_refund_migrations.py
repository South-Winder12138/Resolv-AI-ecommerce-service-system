"""Additive, explicit conversation-domain migration only."""

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from resolv.db.engine import create_db_engine
from tests.test_conversation_migrations import ROOT, upgrade


def test_additive_refund_migration_preserves_old_rows(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as c:
        config = Config(str(ROOT / "alembic.conversation.ini"))
        config.attributes["connection"] = c
        command.upgrade(config, "20261015_0004")
        c.execute(text("CREATE TABLE untouched (payload TEXT)"))
        original = '{"schema_version":2,"amount":9007199254740993,"score":0.12345678901234567}'
        c.execute(text("INSERT INTO untouched VALUES (:v)"), {"v": original})
        before = set(inspect(c).get_table_names())
        config = upgrade(c)
        assert set(inspect(c).get_table_names()) - before == {
            "conversation_refund_runs", "conversation_refund_offers", "conversation_refund_submissions", "conversation_refund_request_aliases", "conversation_logistics_attempts"}
        assert {v["name"]:v for v in inspect(c).get_columns("conversation_runs")}["refund_config"]["nullable"]
        upgrade(c)
        assert c.scalar(text("SELECT payload FROM untouched")) == original
        assert c.scalar(text("SELECT count(*) FROM conversation_refund_runs")) == 0
        command.check(config)
    engine.dispose()
