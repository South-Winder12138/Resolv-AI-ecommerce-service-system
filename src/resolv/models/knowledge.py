"""Authoritative knowledge records; no vectors or resource initialization."""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from resolv.db.knowledge_base import KnowledgeBase
from resolv.models.order import utc_now

_TABLE_OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
}


class KnowledgeDocument(KnowledgeBase):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        CheckConstraint("length(document_id) = 64", name="ck_knowledge_documents_id"),
        CheckConstraint("length(ingest_hash) = 64", name="ck_knowledge_documents_hash"),
        CheckConstraint("length(trim(source)) > 0", name="ck_knowledge_documents_source"),
        CheckConstraint("length(trim(content_type)) > 0", name="ck_knowledge_documents_type"),
        _TABLE_OPTIONS,
    )

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"))
    ingest_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utc_now, onupdate=utc_now
    )
    chunks: Mapped[list["KnowledgeChunkRecord"]] = relationship(
        back_populates="document", order_by="KnowledgeChunkRecord.chunk_index",
        passive_deletes="all",
    )


class KnowledgeChunkRecord(KnowledgeBase):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_knowledge_chunks_document_index"),
        CheckConstraint("length(chunk_id) = 64", name="ck_knowledge_chunks_id"),
        CheckConstraint("length(index_hash) = 64", name="ck_knowledge_chunks_hash"),
        CheckConstraint(
            "indexed_hash IS NULL OR length(indexed_hash) = 64",
            name="ck_knowledge_chunks_indexed_hash",
        ),
        CheckConstraint("length(trim(source)) > 0", name="ck_knowledge_chunks_source"),
        CheckConstraint("length(trim(content_type)) > 0", name="ck_knowledge_chunks_type"),
        CheckConstraint("length(trim(answer)) > 0", name="ck_knowledge_chunks_answer"),
        CheckConstraint("length(trim(text)) > 0", name="ck_knowledge_chunks_text"),
        CheckConstraint("chunk_index >= 0", name="ck_knowledge_chunks_index"),
        CheckConstraint(
            "index_status IN ('pending', 'done') "
            "AND length(index_status) = length(trim(index_status))",
            name="ck_knowledge_chunks_status",
        ),
        CheckConstraint("is_key_clause IN (0, 1)", name="ck_knowledge_chunks_key_clause"),
        CheckConstraint("attempt_count >= 0", name="ck_knowledge_chunks_attempts"),
        CheckConstraint(
            "index_status = 'pending' OR (indexed_hash IS NOT NULL "
            "AND indexed_hash = index_hash AND indexed_collection IS NOT NULL "
            "AND length(trim(indexed_collection)) > 0 AND indexed_at IS NOT NULL)",
            name="ck_knowledge_chunks_done",
        ),
        Index("ix_knowledge_chunks_status_id", "index_status", "chunk_id"),
        _TABLE_OPTIONS,
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.document_id", name="fk_knowledge_chunks_document", ondelete="RESTRICT")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text, default="", server_default=sql_text("('')"))
    questions: Mapped[str] = mapped_column(Text, default="", server_default=sql_text("('')"))
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(Text, default="", server_default=sql_text("('')"))
    content_type: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    is_key_clause: Mapped[int] = mapped_column(Integer)
    index_status: Mapped[str] = mapped_column(String(16), default="pending", server_default="pending")
    index_hash: Mapped[str] = mapped_column(String(64))
    indexed_hash: Mapped[str | None] = mapped_column(String(64))
    indexed_collection: Mapped[str | None] = mapped_column(String(255))
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default=sql_text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utc_now, onupdate=utc_now
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")
