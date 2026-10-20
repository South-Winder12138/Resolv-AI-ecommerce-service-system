"""Index committed snapshots without holding SQL Sessions across external work."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from resolv.db.knowledge import KnowledgeStoreError
from resolv.db.knowledge_index import KnowledgeIndexLock
from resolv.db.milvus import VectorStoreError
from resolv.knowledge import (
    build_embedding_text,
    knowledge_chunk_from_record,
    knowledge_index_entry,
    knowledge_index_hash,
)
from resolv.models.order import utc_now
from resolv.repositories.knowledge import KnowledgeRepository
from resolv.schemas.knowledge import (
    KnowledgeIndexProfile,
    KnowledgeIndexResult,
    KnowledgeIndexSnapshot,
)


class KnowledgeIndexer:
    def __init__(
        self, session_factory: sessionmaker[Session], embedding_service,
        vector_store, profile: KnowledgeIndexProfile, lock: KnowledgeIndexLock,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._profile = profile
        self._lock = lock

    def _validate_resources(self) -> None:
        if (
            self._embedding_service.profile != self._profile
            or type(self._embedding_service.dimension) is not int
            or self._embedding_service.dimension != self._profile.dimension
            or self._vector_store.collection_name != self._profile.collection_name
        ):
            raise ValueError("Knowledge indexing resources must match the declared profile.")

    @contextmanager
    def _session(self, *, write: bool = False) -> Iterator[Session]:
        try:
            session = self._session_factory()
        except SQLAlchemyError as exc:
            raise KnowledgeStoreError("Knowledge indexing Session creation failed.") from exc
        primary_error: BaseException | None = None
        try:
            yield session
            if write:
                session.commit()
        except BaseException as exc:
            primary_error = exc
            try:
                session.rollback()
            except Exception:  # noqa: BLE001 - preserve the operation that triggered rollback
                exc.add_note("Knowledge indexing transaction rollback also failed.")
            if isinstance(exc, SQLAlchemyError):
                primary_error = KnowledgeStoreError(
                    "Knowledge indexing SQL operation failed; inspect state before retrying."
                )
                raise primary_error from exc
            raise
        finally:
            try:
                session.close()
            except Exception as exc:
                if primary_error is None:
                    raise KnowledgeStoreError("Knowledge indexing Session cleanup failed.") from exc
                primary_error.add_note("Knowledge indexing Session cleanup also failed.")

    def run(self, *, batch_size: int = 8) -> KnowledgeIndexResult:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._validate_resources()
        attempted_count = indexed_count = batch_count = 0
        after_id: str | None = None
        while True:
            self._lock.assert_owned()
            with self._session() as session:
                records = KnowledgeRepository(session).list_pending_chunks(after_id, batch_size)
                snapshots = [KnowledgeIndexSnapshot(
                    **knowledge_chunk_from_record(record).model_dump(),
                    index_hash=record.index_hash,
                ) for record in records]
            if not snapshots:
                break
            after_id = snapshots[-1].chunk_id
            try:
                for snapshot in snapshots:
                    if (
                        snapshot.text != build_embedding_text(
                            snapshot.category, snapshot.questions, snapshot.answer
                        )
                        or snapshot.index_hash != knowledge_index_hash(snapshot, self._profile)
                    ):
                        raise ValueError("Knowledge snapshot text or hash does not match its profile.")
                entries = [knowledge_index_entry(snapshot, self._profile) for snapshot in snapshots]
                self._lock.assert_owned()
                with self._session(write=True) as session:
                    count = KnowledgeRepository(session).increment_attempts(snapshots)
                    if count != len(snapshots):
                        raise KnowledgeStoreError("Knowledge snapshots changed before their indexing attempt.")
                attempted_count += len(snapshots)
                batch_count += 1
                # Both preceding Sessions have closed before inference or RPC.
                self._lock.assert_owned()
                vectors = self._embedding_service.embed_documents([snapshot.text for snapshot in snapshots])
                self._lock.assert_owned()
                count = self._vector_store.upsert(entries, vectors)
                if type(count) is not int or count != len(snapshots):
                    raise VectorStoreError("Milvus did not confirm the complete knowledge batch.")
                self._lock.assert_owned()
                with self._session(write=True) as session:
                    count = KnowledgeRepository(session).confirm_indexed(
                        snapshots, self._profile.collection_name, utc_now(),
                    )
                    if count != len(snapshots):
                        raise KnowledgeStoreError("Knowledge snapshots changed before indexing confirmation.")
                    # Recheck immediately before committing guarded confirmation.
                    self._lock.assert_owned()
                indexed_count += len(snapshots)
            except Exception as exc:
                try:
                    self._lock.assert_owned()
                    with self._session(write=True) as session:
                        KnowledgeRepository(session).record_indexing_error(snapshots, exc)
                except Exception:  # noqa: BLE001 - secondary recording must not mask the failure
                    exc.add_note("Knowledge indexing error recording also failed.")
                raise
        return KnowledgeIndexResult(
            attempted_count=attempted_count, indexed_count=indexed_count,
            batch_count=batch_count,
        )
