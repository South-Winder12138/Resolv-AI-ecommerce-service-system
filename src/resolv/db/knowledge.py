"""Explicit caller-owned knowledge database resources; importing connects nowhere."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine

from resolv.core.config import KnowledgeDatabaseSettings
from resolv.db.engine import create_db_engine


class KnowledgeStoreError(RuntimeError):
    """Knowledge database creation, operations, or cleanup failed."""


def create_knowledge_engine(settings: KnowledgeDatabaseSettings) -> Engine:
    try:
        return create_db_engine(
            settings.knowledge_database_url.get_secret_value(),
            pool_pre_ping=True, hide_parameters=True,
            connect_args={"connect_timeout": 10},
        )
    except Exception as exc:
        raise KnowledgeStoreError("Knowledge database engine creation failed.") from exc


@contextmanager
def knowledge_engine(settings: KnowledgeDatabaseSettings) -> Iterator[Engine]:
    engine = create_knowledge_engine(settings)
    body_error: BaseException | None = None
    try:
        yield engine
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            engine.dispose()
        except Exception as exc:
            if body_error is None:
                raise KnowledgeStoreError("Knowledge database cleanup failed.") from exc
            body_error.add_note("Knowledge database engine cleanup also failed.")
