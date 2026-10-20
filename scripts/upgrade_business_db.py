"""Explicit additive SQLite receipt upgrade; never reset or seed existing orders."""

from sqlalchemy import inspect

from resolv.core.config import DatabaseSettings
from resolv.db.engine import create_db_engine
from resolv.models.refund import Refund
from resolv.models.tool_execution import ToolExecutionReceipt


def upgrade(engine):
    if engine.dialect.name != "sqlite":
        raise ValueError("business receipt upgrade requires SQLite")
    inspector = inspect(engine)
    if not {"orders", "shipments"} <= set(inspector.get_table_names()):
        raise ValueError(
            "initialize empty business databases explicitly before upgrading"
        )
    # Older initialized installations predate the existing Refund mapping.
    # Explicitly add its empty table; never seed or rebuild legacy business rows.
    Refund.__table__.create(engine, checkfirst=True)
    table = ToolExecutionReceipt.__table__
    table.create(engine, checkfirst=True)
    columns = inspect(engine).get_columns(table.name)
    if {column["name"] for column in columns} != set(table.columns.keys()) or inspect(
        engine
    ).get_pk_constraint(table.name)["constrained_columns"] != ["action_id"]:
        raise ValueError("existing receipt schema is incompatible")


def main():
    engine = create_db_engine(DatabaseSettings().database_url)
    try:
        upgrade(engine)
    finally:
        engine.dispose()
    print("SQLite receipt schema upgraded; existing business rows preserved.")


if __name__ == "__main__":
    main()
