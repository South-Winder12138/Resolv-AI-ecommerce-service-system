"""Explicit short-transaction staging; embedding and vector storage are separate."""
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256

from pymysql.err import IntegrityError as MySQLIntegrityError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from resolv.db.knowledge import KnowledgeStoreError
from resolv.knowledge import (
    chunk_knowledge_document,
    knowledge_index_hash,
    knowledge_ingest_hash,
)
from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.repositories.knowledge import KnowledgeRepository
from resolv.schemas.knowledge import (
    Document,
    KnowledgeChunk,
    KnowledgeFileConfig,
    KnowledgeIndexProfile,
    KnowledgeStageResult,
)


class KnowledgeConflictError(ValueError):
    """Existing authoritative input differs; replacement needs an explicit policy."""


class KnowledgeIngestionService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def stage_document(
        self, document: Document, file_config: KnowledgeFileConfig,
        profile: KnowledgeIndexProfile, *, max_chars: int = 800, overlap: int = 100,
    ) -> KnowledgeStageResult:
        if document.document_id != sha256(document.source.encode("utf-8")).hexdigest():
            raise ValueError("Document identity must match its normalized source.")
        chunks = chunk_knowledge_document(
            document, file_config, max_chars=max_chars, overlap=overlap
        )
        ingest_hash = knowledge_ingest_hash(
            document, file_config, max_chars=max_chars, overlap=overlap
        )
        index_hashes = {chunk.chunk_id: knowledge_index_hash(chunk, profile) for chunk in chunks}
        try:
            result = self._persist(document, file_config, chunks, ingest_hash, index_hashes)
        except IntegrityError as exc:
            if not self._is_unique_violation(exc):
                raise KnowledgeStoreError("Knowledge staging failed due to an integrity constraint.") from exc
            # A competing first insert can win after the initial SELECT. Never
            # reuse the failed transaction or treat incomplete/conflicting data as success.
            try:
                result = self._persist(
                    document, file_config, chunks, ingest_hash, index_hashes,
                    existing_only=True,
                )
            except SQLAlchemyError as confirmation_error:
                raise KnowledgeStoreError("Knowledge staging confirmation failed.") from confirmation_error
            if result is None:
                raise KnowledgeStoreError("Knowledge staging failed; no complete duplicate confirmed.") from exc
        except SQLAlchemyError as exc:
            raise KnowledgeStoreError("Knowledge staging failed; inspect state before retrying.") from exc
        assert result is not None
        return result

    @staticmethod
    def _is_unique_violation(error: IntegrityError) -> bool:
        original = error.orig
        if isinstance(original, sqlite3.IntegrityError):
            return getattr(original, "sqlite_errorcode", None) in {
                sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY, sqlite3.SQLITE_CONSTRAINT_UNIQUE,
            }
        return bool(
            isinstance(original, MySQLIntegrityError)
            and original.args and original.args[0] == 1062
        )

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._session_factory()
        primary_error: BaseException | None = None
        try:
            yield session
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                session.close()
            except Exception as exc:
                if primary_error is None:
                    raise KnowledgeStoreError("Knowledge staging Session cleanup failed.") from exc
                primary_error.add_note("Knowledge staging Session cleanup also failed.")
                if isinstance(primary_error, IntegrityError):
                    # Duplicate recovery must not turn failed resource cleanup into success.
                    raise KnowledgeStoreError(
                        "Knowledge staging failed and Session cleanup also failed."
                    ) from primary_error

    def _persist(
        self, document: Document, file_config: KnowledgeFileConfig,
        chunks: list[KnowledgeChunk], ingest_hash: str, index_hashes: dict[str, str],
        *, existing_only: bool = False,
    ) -> KnowledgeStageResult | None:
        with self._session() as session, session.begin():
            repository = KnowledgeRepository(session)
            existing = repository.get_document(document.document_id)
            if existing is not None:
                records = repository.list_document_chunks(document.document_id)
                self._verify_existing(existing, records, document, file_config, chunks, ingest_hash, index_hashes)
                return self._result(existing, records, "unchanged")
            if existing_only:
                return None
            record = KnowledgeDocument(
                document_id=document.document_id, source=document.source,
                content_type=file_config.content_type, raw_text=document.text,
                ingest_hash=ingest_hash,
            )
            records = [
                KnowledgeChunkRecord(
                    **chunk.model_dump(), index_hash=index_hashes[chunk.chunk_id],
                    index_status="pending", attempt_count=0,
                ) for chunk in chunks
            ]
            record.chunks = records
            repository.add_document(record)
            repository.add_chunks(records)
            return self._result(record, records, "created")

    @staticmethod
    def _verify_existing(
        existing: KnowledgeDocument, records: list[KnowledgeChunkRecord],
        document: Document, file_config: KnowledgeFileConfig, chunks: list[KnowledgeChunk],
        ingest_hash: str, index_hashes: dict[str, str],
    ) -> None:
        same_document = (
            existing.ingest_hash == ingest_hash and existing.source == document.source
            and existing.raw_text == document.text
            and existing.content_type == file_config.content_type
        )
        same_chunks = len(records) == len(chunks) and all(
            all(getattr(record, key) == value for key, value in chunk.model_dump().items())
            and record.index_hash == index_hashes[chunk.chunk_id]
            for record, chunk in zip(records, chunks, strict=True)
        )
        if not same_document or not same_chunks:
            raise KnowledgeConflictError(
                "Existing knowledge differs or is incomplete; explicit replacement/reindex policy is required."
            )

    @staticmethod
    def _result(
        document: KnowledgeDocument, chunks: list[KnowledgeChunkRecord], outcome: str,
    ) -> KnowledgeStageResult:
        return KnowledgeStageResult(
            document_id=document.document_id, source=document.source,
            content_type=document.content_type, ingest_hash=document.ingest_hash,
            outcome=outcome, chunk_count=len(chunks),
            pending_count=sum(chunk.index_status == "pending" for chunk in chunks),
            done_count=sum(chunk.index_status == "done" for chunk in chunks),
        )
