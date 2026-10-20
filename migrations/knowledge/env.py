"""Explicit knowledge-only migration execution; application startup imports none."""

from alembic import context
from sqlalchemy import Connection

from resolv.core.config import KnowledgeDatabaseSettings
from resolv.db.knowledge import KnowledgeStoreError, knowledge_engine
from resolv.models.knowledge import KnowledgeDocument
from resolv.models.knowledge_hybrid import KnowledgeIndexVersion

assert KnowledgeIndexVersion.metadata is KnowledgeDocument.metadata

config = context.config
target_metadata = KnowledgeDocument.metadata


def include_name(name, type_, parent_names):
    return type_ != "table" or name in target_metadata.tables


def run_migrations_offline() -> None:
    settings = KnowledgeDatabaseSettings()
    context.configure(
        url=settings.knowledge_database_url.get_secret_value(),
        target_metadata=target_metadata, literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="knowledge_alembic_version", include_name=include_name,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def migrate_connection(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata,
        version_table="knowledge_alembic_version", include_name=include_name,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    supplied = config.attributes.get("connection")
    if supplied is not None:
        migrate_connection(supplied)
        return
    with knowledge_engine(KnowledgeDatabaseSettings()) as engine:
        connection = engine.connect()
        body_error: BaseException | None = None
        try:
            migrate_connection(connection)
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                connection.close()
            except Exception as exc:
                if body_error is None:
                    raise KnowledgeStoreError("Knowledge migration connection cleanup failed.") from exc
                body_error.add_note("Knowledge migration connection cleanup also failed.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
