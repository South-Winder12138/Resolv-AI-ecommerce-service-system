"""Add refund stages, immutable offers and independently leased submissions."""

import sqlalchemy as sa
from alembic import op

revision = "20261016_0005"
down_revision = "20261015_0004"
branch_labels = None
depends_on = None
OPTIONS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_bin"}


def col(name, kind, *, nullable=False, fk=None, primary_key=False):
    args = [sa.ForeignKey(fk)] if fk else []
    return sa.Column(name, kind, *args, nullable=nullable, primary_key=primary_key)


def upgrade():
    op.add_column("conversation_runs", sa.Column("refund_config", sa.JSON(), nullable=True))
    op.create_table("conversation_refund_runs",
        col("run_id", sa.String(64), fk="conversation_runs.run_id", primary_key=True),
        col("owner_id", sa.String(64)), col("conversation_id", sa.String(64), fk="conversations.conversation_id"),
        col("understanding_id", sa.String(64), fk="conversation_understandings.understanding_id"),
        col("primary_task_id", sa.String(64), fk="conversation_understanding_requests.task_id"),
        col("primary_hash", sa.String(64)), col("primary_json", sa.JSON()), col("profile_json", sa.JSON()), col("profile_hash", sa.String(64)),
        col("stage_data", sa.JSON()), col("model_requests", sa.JSON()), col("observations", sa.JSON()), col("audits", sa.JSON()),
        col("revision", sa.Integer()), col("error_code", sa.String(64), nullable=True), col("retryable", sa.Boolean()), col("step_limit", sa.Boolean()),
        col("created_at", sa.DateTime()), col("updated_at", sa.DateTime()), **OPTIONS)
    op.create_table("conversation_refund_offers",
        col("offer_id", sa.String(64), primary_key=True), col("run_id", sa.String(64), fk="conversation_refund_runs.run_id"),
        col("owner_id", sa.String(64)), col("conversation_id", sa.String(64), fk="conversations.conversation_id"),
        col("payload", sa.JSON()), col("payload_hash", sa.String(64)), col("status", sa.String(32)), col("invalid_reason", sa.String(64), nullable=True),
        col("created_at", sa.DateTime()), col("expires_at", sa.DateTime()), sa.UniqueConstraint("run_id", name="uq_refund_offer_run"), **OPTIONS)
    op.create_table("conversation_refund_submissions",
        col("command_id", sa.String(64), primary_key=True), col("offer_id", sa.String(64), fk="conversation_refund_offers.offer_id"),
        col("owner_id", sa.String(64)), col("conversation_id", sa.String(64), fk="conversations.conversation_id"), col("request_hash", sa.String(64)),
        col("status", sa.String(24)), col("execution_epoch", sa.Integer()), col("execution_owner_id", sa.String(64), nullable=True),
        col("lease_expires_at", sa.DateTime(), nullable=True), col("revalidations", sa.JSON()), col("receipt", sa.JSON(none_as_null=True), nullable=True),
        col("receipt_hash", sa.String(64), nullable=True), col("error_code", sa.String(64), nullable=True), col("retryable", sa.Boolean()),
        col("created_at", sa.DateTime()), col("updated_at", sa.DateTime()), sa.UniqueConstraint("offer_id", name="uq_refund_submission_offer"), **OPTIONS)
    op.create_table("conversation_refund_request_aliases",
        col("owner_id", sa.String(64), primary_key=True), col("request_id", sa.String(128), primary_key=True),
        col("command_id", sa.String(64), fk="conversation_refund_submissions.command_id"), col("request_hash", sa.String(64)),
        col("created_at", sa.DateTime()), **OPTIONS)


def downgrade():
    for name in ("request_aliases", "submissions", "offers", "runs"):
        op.drop_table("conversation_refund_"+name)
    op.drop_column("conversation_runs", "refund_config")
