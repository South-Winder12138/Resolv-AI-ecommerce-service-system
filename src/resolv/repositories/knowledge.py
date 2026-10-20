"""SQL access only; the caller owns Session, transactions and resource lifetime."""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import LargeBinary, cast, select, update
from sqlalchemy.orm import Session

from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.schemas.knowledge import KnowledgeIndexSnapshot


class KnowledgeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_document(self, document: KnowledgeDocument) -> None:
        self.session.add(document)

    def add_chunks(self, chunks: Sequence[KnowledgeChunkRecord]) -> None:
        self.session.add_all(chunks)

    def get_document(self, document_id: str) -> KnowledgeDocument | None:
        return self.session.get(KnowledgeDocument, document_id)

    def list_document_chunks(self, document_id: str) -> list[KnowledgeChunkRecord]:
        return list(self.session.scalars(
            select(KnowledgeChunkRecord)
            .where(KnowledgeChunkRecord.document_id == document_id)
            .order_by(KnowledgeChunkRecord.chunk_index)
        ))

    def get_chunks_by_ids(
        self, chunk_ids: Sequence[str]
    ) -> dict[str, KnowledgeChunkRecord]:
        if not chunk_ids:
            return {}
        chunks = self.session.scalars(
            select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.chunk_id.in_(chunk_ids))
        )
        return {chunk.chunk_id: chunk for chunk in chunks}

    def list_pending_chunks(
        self, after_id: str | None = None, limit: int = 8,
    ) -> list[KnowledgeChunkRecord]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        statement = select(KnowledgeChunkRecord).where(
            KnowledgeChunkRecord.index_status == "pending"
        )
        if after_id is not None:
            statement = statement.where(KnowledgeChunkRecord.chunk_id > after_id)
        return list(self.session.scalars(
            statement.order_by(KnowledgeChunkRecord.chunk_id).limit(limit)
        ))

    @staticmethod
    def _snapshot_update(snapshot: KnowledgeIndexSnapshot):
        # Also guard the plain payload: a caller can edit text/metadata without
        # recomputing index_hash. Such edits must not confirm an older vector.
        # utf8mb4_bin uses PAD SPACE equality for TEXT. Compare UTF-8 bytes so
        # trailing spaces cannot confirm or annotate a different snapshot.
        payload = snapshot.model_dump(exclude={"index_hash"})
        return update(KnowledgeChunkRecord).where(
            KnowledgeChunkRecord.chunk_id == snapshot.chunk_id,
            KnowledgeChunkRecord.index_hash == snapshot.index_hash,
            KnowledgeChunkRecord.index_status == "pending",
            *(cast(getattr(KnowledgeChunkRecord, name), LargeBinary) == value.encode("utf-8")
              if isinstance(value, str) else getattr(KnowledgeChunkRecord, name) == value
              for name, value in payload.items()),
        ).execution_options(synchronize_session=False)

    def increment_attempts(self, snapshots: Sequence[KnowledgeIndexSnapshot]) -> int:
        return sum(self.session.execute(self._snapshot_update(snapshot).values(
            attempt_count=KnowledgeChunkRecord.attempt_count + 1,
        )).rowcount for snapshot in snapshots)

    def confirm_indexed(
        self, snapshots: Sequence[KnowledgeIndexSnapshot], collection_name: str,
        indexed_at: datetime,
    ) -> int:
        return sum(self.session.execute(self._snapshot_update(snapshot).values(
            index_status="done", indexed_hash=snapshot.index_hash,
            indexed_collection=collection_name, indexed_at=indexed_at,
            last_error=None,
        )).rowcount for snapshot in snapshots)

    def record_indexing_error(
        self, snapshots: Sequence[KnowledgeIndexSnapshot], error: object,
    ) -> int:
        # Never store arbitrary exception text, SQL parameters, or source text.
        safe_error = "Knowledge indexing failed; inspect state before retrying."
        return sum(self.session.execute(self._snapshot_update(snapshot).values(
            last_error=safe_error,
        )).rowcount for snapshot in snapshots)
