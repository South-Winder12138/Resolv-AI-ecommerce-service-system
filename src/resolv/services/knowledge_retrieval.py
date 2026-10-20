"""Read-only retrieval from confirmed SQL authority and opaque vector candidates."""

from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from resolv.db.knowledge import KnowledgeStoreError
from resolv.knowledge import (
    build_embedding_text,
    knowledge_chunk_from_record,
    knowledge_index_hash,
)
from resolv.repositories.knowledge import KnowledgeRepository
from resolv.repositories.knowledge_milvus import KnowledgeMilvusVectorStore
from resolv.schemas.knowledge import (
    KnowledgeCandidate,
    KnowledgeIndexProfile,
    KnowledgeSearchHit,
)
from resolv.services.knowledge_embedding import KnowledgeEmbeddingService


class KnowledgeRetriever:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedding_service: KnowledgeEmbeddingService,
        vector_store: KnowledgeMilvusVectorStore,
        profile: KnowledgeIndexProfile,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._profile = profile

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeSearchHit]:
        """Preserve candidate order/score, filtering unconfirmed or changed SQL rows."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonblank string")
        if type(top_k) is not int or not 1 <= top_k <= 16383:
            raise ValueError("top_k must be an integer between 1 and 16383")
        if (
            self._embedding_service.profile != self._profile
            or self._embedding_service.dimension != self._profile.dimension
        ):
            raise ValueError("Knowledge embedding must match the declared index profile.")
        if self._vector_store.collection_name != self._profile.collection_name:
            raise ValueError("Knowledge vector store must match the declared collection.")

        vector = self._embedding_service.embed_query(query)
        candidates = self._vector_store.search(vector, top_k)
        # First provider occurrence owns each ID; never refill filtered candidates.
        unique: dict[str, KnowledgeCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.chunk_id, candidate)
        ordered = list(unique.values())
        if not ordered:
            return []
        return self._read_confirmed(ordered)

    def _read_confirmed(
        self, candidates: list[KnowledgeCandidate]
    ) -> list[KnowledgeSearchHit]:
        session: Session | None = None
        primary_error: BaseException | None = None
        hits: list[KnowledgeSearchHit] = []
        try:
            session = self._session_factory()
            records = KnowledgeRepository(session).get_chunks_by_ids(
                [candidate.chunk_id for candidate in candidates]
            )
            for candidate in candidates:
                record = records.get(candidate.chunk_id)
                if record is None or not (
                    record.index_status == "done"
                    and record.indexed_hash == record.index_hash == candidate.index_hash
                    and record.indexed_collection == self._profile.collection_name
                    and record.indexed_at is not None
                    and record.document_id == candidate.document_id
                    and record.text == build_embedding_text(
                        record.category, record.questions, record.answer
                    )
                ):
                    continue
                try:
                    chunk = knowledge_chunk_from_record(record)
                except ValidationError:
                    # Invalid legacy/corrupted payloads are not authoritative hits.
                    continue
                # DTO normalization must not hide a changed stored payload.
                if any(
                    getattr(record, name) != value
                    for name, value in chunk.model_dump().items()
                ):
                    continue
                if knowledge_index_hash(chunk, self._profile) != candidate.index_hash:
                    continue
                hits.append(KnowledgeSearchHit(**chunk.model_dump(), score=candidate.score))
        except Exception as exc:
            primary_error = KnowledgeStoreError("Knowledge retrieval SQL read failed.")
            raise primary_error from exc
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as exc:
                    if primary_error is None:
                        raise KnowledgeStoreError("Knowledge retrieval Session cleanup failed.") from exc
                    primary_error.add_note("Knowledge retrieval Session cleanup also failed.")
        return hits
