from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from resolv.db.engine import create_db_engine

ROOT = Path(__file__).resolve().parents[1]


def test_upgrade_0005_adds_only_attempts_table_and_preserves_old_rows(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    config = Config(str(ROOT / "alembic.conversation.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "20261016_0005")
        before = set(inspect(connection).get_table_names())
        connection.execute(text("INSERT INTO conversations (conversation_id, owner_id, thread_id, title, next_message_seq, status, created_at, updated_at) VALUES ('old', 'local', 'old-thread', '旧会话', 1, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
        connection.execute(text("CREATE TABLE retained_probe (id INTEGER PRIMARY KEY, body TEXT)"))
        connection.execute(text("INSERT INTO retained_probe VALUES (1, '旧记录')"))
        command.upgrade(config, "head")
        assert set(inspect(connection).get_table_names()) - before == {
            "retained_probe", "conversation_logistics_attempts",
        }
        assert connection.scalar(text("SELECT body FROM retained_probe")) == "旧记录"
        assert connection.scalar(text("SELECT title FROM conversations WHERE conversation_id='old'")) == "旧会话"
        assert connection.scalar(text("SELECT version_num FROM conversation_alembic_version")) == "20261020_0006"
        constraints = inspect(connection).get_unique_constraints("conversation_logistics_attempts")
        assert {"action_id", "attempt_no"} in [set(c["column_names"]) for c in constraints]
        action_checks = inspect(connection).get_check_constraints("conversation_actions")
        execution = next(c["sqltext"] for c in action_checks if c["name"] == "ck_action_execution")
        assert execution == "execution_status IN ('pending','completed')"
        command.check(config)
        command.upgrade(config, "head")
        assert connection.scalar(text("SELECT body FROM retained_probe")) == "旧记录"
    engine.dispose()
