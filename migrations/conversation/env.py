from alembic import context

from resolv.core.config import ConversationDatabaseSettings
from resolv.db.conversation import conversation_engine
from resolv.models.conversation import Conversation
from resolv.models.knowledge_branch import ConversationKnowledgeRun  # noqa: F401
from resolv.models.logistics import ConversationLogisticsAttempt  # noqa: F401
from resolv.models.refund_flow import ConversationRefundRun  # noqa: F401
from resolv.models.understanding import ConversationUnderstanding  # noqa: F401

config = context.config
target_metadata = Conversation.metadata


def include_name(name, type_, parent_names):
    return type_ != "table" or name in target_metadata.tables


def migrate(connection=None):
    options = {
        "target_metadata": target_metadata,
        "version_table": "conversation_alembic_version",
        "include_name": include_name,
        "compare_type": True,
    }
    if connection is None:
        options.update(
            url=ConversationDatabaseSettings().conversation_database_url.get_secret_value(),
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )
    else:
        options["connection"] = connection
    context.configure(**options)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    migrate()
elif config.attributes.get("connection") is not None:
    migrate(config.attributes["connection"])
else:
    with (
        conversation_engine(ConversationDatabaseSettings()) as engine,
        engine.connect() as connection,
    ):
        migrate(connection)
