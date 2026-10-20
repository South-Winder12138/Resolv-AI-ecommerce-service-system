from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from resolv.db.engine import create_db_engine

ROOT = Path(__file__).resolve().parents[1]


def upgrade(connection):
    config = Config(str(ROOT / "alembic.conversation.ini"))
    config.attributes["connection"] = connection
    command.upgrade(config, "head")
    return config


def test_upgrade_creates_seven_domain_tables_only(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as connection:
        upgrade(connection)
    assert set(inspect(engine).get_table_names()) == {
        "conversations",
        "conversation_messages",
        "conversation_runs",
        "conversation_actions",
        "conversation_summaries",
        "conversation_alembic_version",
        "conversation_understandings",
        "conversation_understanding_requests",
        "conversation_knowledge_runs",
        "conversation_knowledge_review_items",
        "conversation_refund_runs",
        "conversation_refund_offers",
        "conversation_refund_submissions",
        "conversation_refund_request_aliases",
        "conversation_logistics_attempts",
    }
    engine.dispose()


def test_second_upgrade_preserves_messages(tmp_path):
    from importlib import import_module

    from resolv.db.engine import create_session_factory

    models = import_module("resolv.models.conversation")
    engine = create_db_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as connection:
        upgrade(connection)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        session.add(
            models.Conversation(conversation_id="c", thread_id="t", title="hello")
        )
        session.flush()
        session.add(
            models.ConversationRun(
                run_id="r",
                conversation_id="c",
                request_id="req",
                request_fingerprint="f" * 64,
                question_message_id="m",
                answer_message_id="a",
            )
        )
        session.flush()
        session.add(
            models.ConversationMessage(
                message_id="m",
                conversation_id="c",
                run_id="r",
                seq=1,
                role="user",
                text="原问题",
            )
        )
    with engine.begin() as connection:
        upgrade(connection)
        assert (
            connection.scalar(text("SELECT text FROM conversation_messages"))
            == "原问题"
        )
    engine.dispose()


def test_migration_ignores_checkpoint_and_knowledge_tables(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as connection:
        for table in ("checkpoints", "knowledge_documents", "orders"):
            connection.execute(text(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)"))
            connection.execute(text(f"INSERT INTO {table} VALUES (7)"))
        config = upgrade(connection)
        command.check(config)
        for table in ("checkpoints", "knowledge_documents", "orders"):
            assert connection.scalar(text(f"SELECT id FROM {table}")) == 7
    engine.dispose()


def test_routed_upgrade_backfills_old_rows_and_constrains_understanding(tmp_path):
    import pytest
    from sqlalchemy import MetaData, Table
    from sqlalchemy.exc import IntegrityError

    engine = create_db_engine(f"sqlite:///{tmp_path / 'old.db'}")
    config = Config(str(ROOT / "alembic.conversation.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "20261013_0002")
        metadata = MetaData()
        c = Table("conversations", metadata, autoload_with=connection)
        r = Table("conversation_runs", metadata, autoload_with=connection)
        m = Table("conversation_messages", metadata, autoload_with=connection)
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        connection.execute(
            c.insert().values(
                conversation_id="c",
                owner_id="local",
                thread_id="t",
                title="旧会话",
                next_message_seq=2,
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            r.insert().values(
                run_id="r",
                conversation_id="c",
                request_id="q",
                request_fingerprint="f" * 64,
                mode="agent",
                status="completed",
                attempt=1,
                execution_epoch=1,
                question_message_id="m",
                answer_message_id="a",
                history_cutoff_seq=0,
                summary_version=0,
                memory_prepared=False,
                knowledge_snapshot=[],
                knowledge_prepared=False,
                model_name="fake",
                prompt_version="v1",
                system_prompt_snapshot="",
                tool_schema_fingerprint="",
                state_schema_version=1,
                count_config={},
                retryable=False,
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            m.insert().values(
                message_id="m",
                conversation_id="c",
                run_id="r",
                seq=1,
                role="user",
                text="旧问题",
                created_at=now,
            )
        )
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
        columns = {
            column["name"]: column
            for column in inspect(connection).get_columns("conversation_runs")
        }
        assert "orchestration_version" in columns
        r = Table("conversation_runs", MetaData(), autoload_with=connection)
        old = connection.execute(r.select()).mappings().one()
        assert old["orchestration_version"] == "legacy-v1"
        assert old["routing_input"] == {} and old["model_audits"] == []
        assert old["understanding_id"] is None and old["result_json"] is None
        assert old["knowledge_config"] is None
        assert (
            connection.scalar(text("SELECT text FROM conversation_messages"))
            == "旧问题"
        )
        assert {"understanding_id"} in [
            set(x["column_names"])
            for x in inspect(connection).get_unique_constraints("conversation_runs")
        ]
        assert any(
            fk["constrained_columns"] == ["understanding_id"]
            and fk["referred_table"] == "conversation_understandings"
            for fk in inspect(connection).get_foreign_keys("conversation_runs")
        )
        with connection.begin_nested(), pytest.raises(IntegrityError):
            connection.execute(r.update().values(understanding_id="missing"))
        command.upgrade(config, "head")
        command.check(config)
    engine.dispose()
