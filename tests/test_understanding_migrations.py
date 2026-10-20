from sqlalchemy import inspect, text

from resolv.db.engine import create_db_engine
from tests.test_conversation_migrations import upgrade


def test_understanding_migration_is_additive_and_repeatable(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'understanding-migrations.db'}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE external_checkpoint (id INTEGER PRIMARY KEY)"))
        connection.execute(text("INSERT INTO external_checkpoint VALUES (7)"))
        upgrade(connection)
        upgrade(connection)
        assert connection.scalar(text("SELECT id FROM external_checkpoint")) == 7
    tables = inspect(engine).get_table_names()
    assert "conversation_understandings" in tables
    assert "conversation_understanding_requests" in tables
    engine.dispose()
