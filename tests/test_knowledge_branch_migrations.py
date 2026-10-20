"""Only additive conversation-domain schema changes; real SQLite DDL."""

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from resolv.db.engine import create_db_engine
from tests.test_conversation_migrations import ROOT, upgrade


def test_knowledge_tables_and_nullable_config_are_additive_and_repeatable(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE untouched (id INTEGER PRIMARY KEY)"))
        conn.execute(text("INSERT INTO untouched VALUES (7)"))
        config = Config(str(ROOT / "alembic.conversation.ini"))
        config.attributes["connection"] = conn
        command.upgrade(config, "20261014_0003")
        before_tables = set(inspect(conn).get_table_names())
        config = upgrade(conn)
        assert set(inspect(conn).get_table_names()) - before_tables == {
            "conversation_knowledge_runs",
            "conversation_knowledge_review_items",
            "conversation_refund_runs",
            "conversation_refund_offers",
            "conversation_refund_submissions",
            "conversation_refund_request_aliases",
            "conversation_logistics_attempts",
        }
        cols = {c["name"]: c for c in inspect(conn).get_columns("conversation_runs")}
        assert cols["knowledge_config"]["nullable"]
        assert conn.scalar(text("SELECT id FROM untouched")) == 7
        first = [
            (name, inspect(conn).get_columns(name)) for name in sorted(before_tables)
        ]
        upgrade(conn)
        second = [
            (name, inspect(conn).get_columns(name)) for name in sorted(before_tables)
        ]
        assert str(first) == str(second)
        command.check(config)
        fks = inspect(conn).get_foreign_keys("conversation_knowledge_review_items")
        assert any(f["referred_table"] == "conversation_knowledge_runs" for f in fks)
    engine.dispose()
