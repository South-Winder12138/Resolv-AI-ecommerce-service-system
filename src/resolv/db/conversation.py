from contextlib import contextmanager

from resolv.core.config import ConversationDatabaseSettings
from resolv.db.engine import create_db_engine


def create_conversation_engine(settings: ConversationDatabaseSettings):
    return create_db_engine(
        settings.conversation_database_url.get_secret_value(),
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={"connect_timeout": 10},
    )


@contextmanager
def conversation_engine(settings: ConversationDatabaseSettings):
    engine = create_conversation_engine(settings)
    try:
        yield engine
    finally:
        engine.dispose()
