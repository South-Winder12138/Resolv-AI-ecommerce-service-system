"""Require exact pending/done status even with MySQL PAD SPACE comparisons."""

from alembic import op

revision = "20261012_0002"
down_revision = "20261012_0001"
branch_labels = None
depends_on = None


def _replace_status_check(expression: str) -> None:
    if op.get_context().dialect.name == "mysql":
        # One atomic ALTER retains the old check if existing rows fail validation.
        op.execute(
            "ALTER TABLE knowledge_chunks "
            "DROP CHECK ck_knowledge_chunks_status, "
            "ADD CONSTRAINT ck_knowledge_chunks_status CHECK (" + expression + ")"
        )
        return
    with op.batch_alter_table("knowledge_chunks") as batch:
        batch.drop_constraint("ck_knowledge_chunks_status", type_="check")
        batch.create_check_constraint("ck_knowledge_chunks_status", expression)


def upgrade() -> None:
    _replace_status_check(
        "index_status IN ('pending', 'done') "
        "AND length(index_status) = length(trim(index_status))"
    )


def downgrade() -> None:
    _replace_status_check("index_status IN ('pending', 'done')")
