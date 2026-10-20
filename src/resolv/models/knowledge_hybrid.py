"""Independent v3 generations; legacy source and v2 confirmation columns are untouched."""

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
    text,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from resolv.db.knowledge_base import KnowledgeBase
from resolv.models.knowledge import _TABLE_OPTIONS
from resolv.models.order import utc_now


def _hash_check(field, table):
    return CheckConstraint(
        f"length({field}) = 64 AND length(trim({field})) = 64",
        name=f"ck_{table}_{field}",
    )


class KnowledgeIndexVersion(KnowledgeBase):
    __tablename__ = "knowledge_index_versions"
    __table_args__ = (
        UniqueConstraint(
            "collection_name", name="uq_knowledge_index_versions_collection"
        ),
        CheckConstraint(
            "length(trim(version_id)) > 0 AND length(version_id) = length(trim(version_id))",
            name="ck_hybrid_version_id",
        ),
        CheckConstraint(
            "length(trim(collection_name)) > 0 AND length(collection_name) = length(trim(collection_name))",
            name="ck_hybrid_collection",
        ),
        _hash_check("profile_hash", "hybrid_version"),
        _hash_check("manifest_hash", "hybrid_version"),
        CheckConstraint(
            "state IN ('building','ready') AND length(state) = length(trim(state))",
            name="ck_hybrid_version_state",
        ),
        CheckConstraint(
            "(state = 'building' AND ready_at IS NULL) OR (state = 'ready' AND ready_at IS NOT NULL)",
            name="ck_hybrid_version_ready",
        ),
        _TABLE_OPTIONS,
    )
    version_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    collection_name: Mapped[str] = mapped_column(String(255))
    profile_json: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"))
    profile_hash: Mapped[str] = mapped_column(String(64))
    manifest_json: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"))
    manifest_hash: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(
        String(16), default="building", server_default="building"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(), default=utc_now, onupdate=utc_now
    )
    ready_at: Mapped[datetime | None] = mapped_column(DateTime())


class KnowledgeChunkIndex(KnowledgeBase):
    __tablename__ = "knowledge_chunk_indexes"
    __table_args__ = (
        *(
            _hash_check(field, "hybrid_chunk")
            for field in (
                "chunk_id",
                "document_id",
                "document_hash",
                "content_hash",
                "section_instance_id",
                "index_hash",
            )
        ),
        CheckConstraint(
            "indexed_hash IS NULL OR (length(indexed_hash) = 64 AND length(trim(indexed_hash)) = 64)",
            name="ck_hybrid_indexed_hash",
        ),
        CheckConstraint(
            "index_status IN ('pending','done') AND length(index_status) = length(trim(index_status))",
            name="ck_hybrid_chunk_state",
        ),
        CheckConstraint(
            "(index_status = 'pending' AND indexed_hash IS NULL AND indexed_at IS NULL) OR "
            "(index_status = 'done' AND indexed_hash IS NOT NULL AND indexed_hash = index_hash AND indexed_at IS NOT NULL)",
            name="ck_hybrid_chunk_done",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_hybrid_chunk_attempts"),
        CheckConstraint("expandable IN (0,1)", name="ck_hybrid_chunk_expandable"),
        Index("ix_hybrid_pending", "index_version", "index_status", "chunk_id"),
        _TABLE_OPTIONS,
    )
    index_version: Mapped[str] = mapped_column(
        ForeignKey(
            "knowledge_index_versions.version_id",
            name="fk_hybrid_chunk_version",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey(
            "knowledge_chunks.chunk_id",
            name="fk_hybrid_chunk_source",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey(
            "knowledge_documents.document_id",
            name="fk_hybrid_chunk_document",
            ondelete="RESTRICT",
        )
    )
    document_hash: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64))
    section_instance_id: Mapped[str] = mapped_column(String(64))
    expandable: Mapped[int] = mapped_column(Integer)
    index_hash: Mapped[str] = mapped_column(String(64))
    index_status: Mapped[str] = mapped_column(
        String(16), default="pending", server_default="pending"
    )
    indexed_hash: Mapped[str | None] = mapped_column(String(64))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime())
    attempt_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0")
    )
    last_error: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(), default=utc_now, onupdate=utc_now
    )
