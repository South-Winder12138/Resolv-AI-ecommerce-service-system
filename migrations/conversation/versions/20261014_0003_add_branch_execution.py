"""Add routed execution fields without altering saver or business tables."""

import sqlalchemy as sa
from alembic import op

revision = "20261014_0003"
down_revision = "20261013_0002"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    sqlite = connection.dialect.name == "sqlite"
    with op.batch_alter_table("conversation_runs") as batch:
        batch.add_column(sa.Column("understanding_id", sa.String(64), nullable=True))
        batch.add_column(
            sa.Column("orchestration_version", sa.String(24), nullable=True)
        )
        batch.add_column(sa.Column("routing_input", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("result_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("business_config", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("model_audits", sa.JSON(), nullable=True))
    table = sa.table(
        "conversation_runs",
        sa.column("orchestration_version", sa.String(24)),
        sa.column("routing_input", sa.JSON()),
        sa.column("model_audits", sa.JSON()),
    )
    op.execute(
        table.update().values(
            orchestration_version="legacy-v1", routing_input={}, model_audits=[]
        )
    )
    if sqlite:
        # Set this after the backfill starts the actual DBAPI transaction: a
        # standalone SQLite DDL statement resets defer_foreign_keys on commit.
        connection.exec_driver_sql("PRAGMA defer_foreign_keys=ON")
    with op.batch_alter_table("conversation_runs") as batch:
        batch.alter_column(
            "orchestration_version", existing_type=sa.String(24), nullable=False
        )
        batch.alter_column("routing_input", existing_type=sa.JSON(), nullable=False)
        batch.alter_column("model_audits", existing_type=sa.JSON(), nullable=False)
        batch.create_unique_constraint("uq_run_understanding", ["understanding_id"])
        batch.create_foreign_key(
            "fk_run_understanding",
            "conversation_understandings",
            ["understanding_id"],
            ["understanding_id"],
        )
    if sqlite:
        if connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("branch migration foreign key validation failed")
        # SQLite retains pending DROP-table checks after replacement even when
        # foreign_key_check is empty. Clear them only after validating every FK.
        connection.exec_driver_sql("PRAGMA defer_foreign_keys=OFF")


def downgrade():
    with op.batch_alter_table("conversation_runs") as batch:
        batch.drop_constraint("fk_run_understanding", type_="foreignkey")
        batch.drop_constraint("uq_run_understanding", type_="unique")
        for name in (
            "model_audits",
            "business_config",
            "result_json",
            "routing_input",
            "orchestration_version",
            "understanding_id",
        ):
            batch.drop_column(name)
