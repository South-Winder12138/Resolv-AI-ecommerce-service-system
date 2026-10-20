"""Create authoritative knowledge tables, separate from business metadata."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import LONGTEXT

revision = "20261012_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_documents",
        sa.Column("document_id", sa.String(64), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text().with_variant(LONGTEXT(), "mysql"), nullable=False),
        sa.Column("ingest_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.CheckConstraint("length(document_id) = 64", name="ck_knowledge_documents_id"),
        sa.CheckConstraint("length(ingest_hash) = 64", name="ck_knowledge_documents_hash"),
        sa.CheckConstraint("length(trim(source)) > 0", name="ck_knowledge_documents_source"),
        sa.CheckConstraint("length(trim(content_type)) > 0", name="ck_knowledge_documents_type"),
        mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin",
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("chunk_id", sa.String(64), primary_key=True),
        sa.Column("document_id", sa.String(64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), server_default=sa.text("('')"), nullable=False),
        sa.Column("questions", sa.Text(), server_default=sa.text("('')"), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("section_path", sa.Text(), server_default=sa.text("('')"), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("is_key_clause", sa.Integer(), nullable=False),
        sa.Column("index_status", sa.String(16), server_default="pending", nullable=False),
        sa.Column("index_hash", sa.String(64), nullable=False),
        sa.Column("indexed_hash", sa.String(64), nullable=True),
        sa.Column("indexed_collection", sa.String(255), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=False), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.document_id"],
                                name="fk_knowledge_chunks_document", ondelete="RESTRICT"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_knowledge_chunks_document_index"),
        sa.CheckConstraint("length(chunk_id) = 64", name="ck_knowledge_chunks_id"),
        sa.CheckConstraint("length(index_hash) = 64", name="ck_knowledge_chunks_hash"),
        sa.CheckConstraint("indexed_hash IS NULL OR length(indexed_hash) = 64",
                           name="ck_knowledge_chunks_indexed_hash"),
        sa.CheckConstraint("length(trim(source)) > 0", name="ck_knowledge_chunks_source"),
        sa.CheckConstraint("length(trim(content_type)) > 0", name="ck_knowledge_chunks_type"),
        sa.CheckConstraint("length(trim(answer)) > 0", name="ck_knowledge_chunks_answer"),
        sa.CheckConstraint("length(trim(text)) > 0", name="ck_knowledge_chunks_text"),
        sa.CheckConstraint("chunk_index >= 0", name="ck_knowledge_chunks_index"),
        sa.CheckConstraint("index_status IN ('pending', 'done')", name="ck_knowledge_chunks_status"),
        sa.CheckConstraint("is_key_clause IN (0, 1)", name="ck_knowledge_chunks_key_clause"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_knowledge_chunks_attempts"),
        sa.CheckConstraint(
            "index_status = 'pending' OR (indexed_hash IS NOT NULL "
            "AND indexed_hash = index_hash AND indexed_collection IS NOT NULL "
            "AND length(trim(indexed_collection)) > 0 AND indexed_at IS NOT NULL)",
            name="ck_knowledge_chunks_done",
        ),
        mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin",
    )
    op.create_index("ix_knowledge_chunks_status_id", "knowledge_chunks", ["index_status", "chunk_id"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunks_status_id", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.drop_table("knowledge_documents")
