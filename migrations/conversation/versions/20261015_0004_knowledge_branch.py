"""Add frozen knowledge config and stages without modifying existing records."""

import sqlalchemy as sa
from alembic import op

revision = "20261015_0004"
down_revision = "20261014_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "conversation_runs", sa.Column("knowledge_config", sa.JSON(), nullable=True)
    )
    columns = []
    for stage in ("query", "retrieval", "check", "answer_input", "answer"):
        columns.extend(
            [
                sa.Column(
                    f"{stage}_payload", sa.JSON(none_as_null=True), nullable=True
                ),
                sa.Column(f"{stage}_input_hash", sa.String(64), nullable=True),
                sa.Column(f"{stage}_output_hash", sa.String(64), nullable=True),
            ]
        )
    options = {
        "mysql_engine": "InnoDB",
        "mysql_charset": "utf8mb4",
        "mysql_collate": "utf8mb4_bin",
    }
    op.create_table(
        "conversation_knowledge_runs",
        sa.Column(
            "run_id",
            sa.String(64),
            sa.ForeignKey("conversation_runs.run_id"),
            primary_key=True,
        ),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("conversations.conversation_id"),
            nullable=False,
        ),
        sa.Column(
            "understanding_id",
            sa.String(64),
            sa.ForeignKey("conversation_understandings.understanding_id"),
            nullable=False,
        ),
        sa.Column(
            "primary_task_id",
            sa.String(64),
            sa.ForeignKey("conversation_understanding_requests.task_id"),
            nullable=False,
        ),
        sa.Column("primary_hash", sa.String(64), nullable=False),
        sa.Column("primary_json", sa.JSON(), nullable=False),
        sa.Column("profile_json", sa.JSON(), nullable=False),
        sa.Column("profile_hash", sa.String(64), nullable=False),
        *columns,
        sa.Column("audits", sa.JSON(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        **options,
    )
    op.create_table(
        "conversation_knowledge_review_items",
        sa.Column(
            "run_id",
            sa.String(64),
            sa.ForeignKey("conversation_knowledge_runs.run_id"),
            primary_key=True,
        ),
        sa.Column("item_json", sa.JSON(), nullable=False),
        sa.Column("item_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        **options,
    )


def downgrade():
    op.drop_table("conversation_knowledge_review_items")
    op.drop_table("conversation_knowledge_runs")
    op.drop_column("conversation_runs", "knowledge_config")
