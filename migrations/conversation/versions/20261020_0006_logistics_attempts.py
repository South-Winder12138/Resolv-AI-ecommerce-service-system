"""Add remote logistics read attempts without changing old actions or rows."""

import sqlalchemy as sa
from alembic import op

revision = "20261020_0006"
down_revision = "20261016_0005"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "conversation_logistics_attempts",
        sa.Column("attempt_id", sa.String(32), primary_key=True),
        sa.Column("action_id", sa.String(64), sa.ForeignKey("conversation_actions.action_id"), nullable=False),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("conversation_runs.run_id"), nullable=False),
        sa.Column("conversation_id", sa.String(64), sa.ForeignKey("conversations.conversation_id"), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("execution_epoch", sa.Integer(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("arguments_hash", sa.String(64), nullable=False),
        sa.Column("tracking_no", sa.String(64), nullable=False),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("envelope", sa.JSON(), nullable=True),
        sa.Column("projection", sa.JSON(), nullable=True),
        sa.Column("result_hash", sa.String(64), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.UniqueConstraint("action_id", "attempt_no", name="uq_logistics_action_attempt"),
        sa.CheckConstraint("state IN ('prepared','completed','failed','unknown')", name="ck_logistics_attempt_state"),
        sa.CheckConstraint("attempt_no > 0", name="ck_logistics_attempt_no"),
        mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin",
    )


def downgrade():
    op.drop_table("conversation_logistics_attempts")
