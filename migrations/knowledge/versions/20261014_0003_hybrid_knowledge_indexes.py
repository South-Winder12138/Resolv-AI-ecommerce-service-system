"""Add independent hybrid generations without changing original knowledge rows."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import LONGTEXT

revision = "20261014_0003"
down_revision = "20261012_0002"
branch_labels = None
depends_on = None
OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
}


def hash_check(field, table):
    return sa.CheckConstraint(
        f"length({field}) = 64 AND length(trim({field})) = 64",
        name=f"ck_{table}_{field}",
    )


def upgrade():
    op.create_table(
        "knowledge_index_versions",
        sa.Column("version_id", sa.String(128), primary_key=True),
        sa.Column("collection_name", sa.String(255), nullable=False),
        sa.Column(
            "profile_json", sa.Text().with_variant(LONGTEXT(), "mysql"), nullable=False
        ),
        sa.Column("profile_hash", sa.String(64), nullable=False),
        sa.Column(
            "manifest_json", sa.Text().with_variant(LONGTEXT(), "mysql"), nullable=False
        ),
        sa.Column("manifest_hash", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False, server_default="building"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("ready_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "collection_name", name="uq_knowledge_index_versions_collection"
        ),
        sa.CheckConstraint(
            "length(trim(version_id)) > 0 AND length(version_id) = length(trim(version_id))",
            name="ck_hybrid_version_id",
        ),
        sa.CheckConstraint(
            "length(trim(collection_name)) > 0 AND length(collection_name) = length(trim(collection_name))",
            name="ck_hybrid_collection",
        ),
        hash_check("profile_hash", "hybrid_version"),
        hash_check("manifest_hash", "hybrid_version"),
        sa.CheckConstraint(
            "state IN ('building','ready') AND length(state) = length(trim(state))",
            name="ck_hybrid_version_state",
        ),
        sa.CheckConstraint(
            "(state = 'building' AND ready_at IS NULL) OR (state = 'ready' AND ready_at IS NOT NULL)",
            name="ck_hybrid_version_ready",
        ),
        **OPTIONS,
    )
    op.create_table(
        "knowledge_chunk_indexes",
        sa.Column("index_version", sa.String(128), primary_key=True),
        sa.Column("chunk_id", sa.String(64), primary_key=True),
        *(
            sa.Column(f, sa.String(64), nullable=False)
            for f in (
                "document_id",
                "document_hash",
                "content_hash",
                "section_instance_id",
                "index_hash",
            )
        ),
        sa.Column("expandable", sa.Integer(), nullable=False),
        sa.Column(
            "index_status", sa.String(16), nullable=False, server_default="pending"
        ),
        sa.Column("indexed_hash", sa.String(64), nullable=True),
        sa.Column("indexed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("last_error", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["index_version"],
            ["knowledge_index_versions.version_id"],
            name="fk_hybrid_chunk_version",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["knowledge_chunks.chunk_id"],
            name="fk_hybrid_chunk_source",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["knowledge_documents.document_id"],
            name="fk_hybrid_chunk_document",
            ondelete="RESTRICT",
        ),
        *(
            hash_check(f, "hybrid_chunk")
            for f in (
                "chunk_id",
                "document_id",
                "document_hash",
                "content_hash",
                "section_instance_id",
                "index_hash",
            )
        ),
        sa.CheckConstraint(
            "indexed_hash IS NULL OR (length(indexed_hash) = 64 AND length(trim(indexed_hash)) = 64)",
            name="ck_hybrid_indexed_hash",
        ),
        sa.CheckConstraint(
            "index_status IN ('pending','done') AND length(index_status) = length(trim(index_status))",
            name="ck_hybrid_chunk_state",
        ),
        sa.CheckConstraint(
            "(index_status = 'pending' AND indexed_hash IS NULL AND indexed_at IS NULL) OR (index_status = 'done' AND indexed_hash IS NOT NULL AND indexed_hash = index_hash AND indexed_at IS NOT NULL)",
            name="ck_hybrid_chunk_done",
        ),
        sa.CheckConstraint("attempt_count >= 0", name="ck_hybrid_chunk_attempts"),
        sa.CheckConstraint("expandable IN (0,1)", name="ck_hybrid_chunk_expandable"),
        **OPTIONS,
    )
    op.create_index(
        "ix_hybrid_pending",
        "knowledge_chunk_indexes",
        ["index_version", "index_status", "chunk_id"],
    )


def downgrade():
    op.drop_index("ix_hybrid_pending", table_name="knowledge_chunk_indexes")
    op.drop_table("knowledge_chunk_indexes")
    op.drop_table("knowledge_index_versions")
