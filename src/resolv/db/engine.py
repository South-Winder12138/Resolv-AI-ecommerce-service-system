from functools import cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def create_db_engine(
    database_url: str, *, pool_pre_ping: bool = False,
    hide_parameters: bool = False, connect_args: dict[str, object] | None = None,
) -> Engine:
    if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
        Path(database_url.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    engine = create_engine(
        database_url, pool_pre_ping=pool_pre_ping,
        hide_parameters=hide_parameters, connect_args=connect_args or {},
    )
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(
            dbapi_connection: object, _connection_record: object
        ) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)


@cache
def get_session_factory(database_url: str) -> sessionmaker[Session]:
    """Reuse an Engine/factory for a URL; each call still opens a fresh Session."""
    if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
        database_file = Path(database_url.removeprefix("sqlite:///"))
        if not database_file.is_file():
            raise RuntimeError(
                "Database not found; initialize and seed it before invoking Tools."
            )
    return create_session_factory(create_db_engine(database_url))
