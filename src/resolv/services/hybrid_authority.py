"""One consistent SQL read view freezes valid centers and adjacent source bodies."""

from collections.abc import Sequence

from pydantic import ValidationError

from resolv.knowledge import knowledge_ingest_hash
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.schemas.hybrid_knowledge import (
    FusedHybridCandidate,
    HybridRetrievalError,
    HybridSourceDocument,
    VerifiedHybridCenter,
    VerifiedHybridChunk,
)
from resolv.schemas.knowledge import Document, KnowledgeChunk
from resolv.services.hybrid_fingerprints import content_hash, document_hash, index_hash


class HybridAuthorityReader:
    def __init__(self, session_factory, version_id, profile, *, fault_hook=None):
        self.session_factory = session_factory
        self.version_id = version_id
        self.profile = profile
        self.fault_hook = fault_hook
        self.invalid_count = 0
        self.valid_count = 0

    def prepare(self, fused: Sequence[FusedHybridCandidate]):
        self.invalid_count = 0
        self.valid_count = 0
        try:
            fused = tuple(
                FusedHybridCandidate.model_validate(x.model_dump()) for x in fused
            )
            if len(fused) > 100 or len({x.identity for x in fused}) != len(fused):
                raise HybridRetrievalError("hybrid_fused_contract_invalid")
            with self.session_factory() as session, session.begin():
                repo = HybridIndexRepository(session)
                v = repo.get_version(self.version_id)
                if v.state != "ready":
                    raise HybridRetrievalError("hybrid_version_not_ready")
                if v.profile != self.profile:
                    raise HybridRetrievalError("hybrid_authority_profile_mismatch")
                rows = repo.read_candidate_rows(self.version_id, fused)
                if self.fault_hook:
                    self.fault_hook("after_centers_read")
                documents = repo.read_documents(
                    tuple({row.document_id for row, _ in rows})
                )
                manifest = {d.document_id: d for d in v.manifest.documents}
                valid_docs = set()
                for did, d in documents.items():
                    expected = manifest.get(did)
                    if expected is None:
                        continue
                    plain = Document(document_id=did, source=d.source, text=d.raw_text)
                    src = HybridSourceDocument(
                        document=plain, ingest_hash=d.ingest_hash, chunks=()
                    )
                    if (
                        d.content_type == expected.file_config.content_type
                        and document_hash(src) == expected.document_hash
                        and knowledge_ingest_hash(
                            plain,
                            expected.file_config,
                            max_chars=expected.max_chars,
                            overlap=expected.overlap,
                        )
                        == expected.ingest_hash
                    ):
                        valid_docs.add(did)
                entries = {
                    mc.chunk.chunk_id: (d, mc)
                    for d in v.manifest.documents
                    for mc in d.chunks
                }

                def verify(record, confirmation):
                    pair = entries.get(record.chunk_id)
                    if pair is None or record.document_id not in valid_docs:
                        return None
                    d, mc = pair
                    raw = {
                        name: getattr(record, name)
                        for name in KnowledgeChunk.model_fields
                    }
                    try:
                        chunk = KnowledgeChunk.model_validate(raw)
                    except ValidationError:
                        return None
                    if (
                        chunk.model_dump() != raw
                        or chunk.model_dump() != mc.chunk.model_dump()
                    ):
                        return None
                    expected = index_hash(mc, v.profile, self.version_id)
                    if (
                        confirmation.index_status != "done"
                        or confirmation.indexed_at is None
                        or confirmation.indexed_hash != expected
                        or confirmation.index_hash != expected
                        or confirmation.content_hash != content_hash(chunk)
                        or confirmation.content_hash != mc.content_hash
                        or confirmation.document_hash != d.document_hash
                        or confirmation.document_id != d.document_id
                        or confirmation.section_instance_id != mc.section_instance_id
                        or confirmation.expandable != int(mc.expandable)
                    ):
                        return None
                    return VerifiedHybridChunk(
                        chunk=chunk,
                        document_hash=d.document_hash,
                        content_hash=mc.content_hash,
                        index_hash=expected,
                        section_instance_id=mc.section_instance_id,
                        expandable=mc.expandable,
                    )

                available = {}
                for row, confirmation in rows:
                    verified = verify(row, confirmation)
                    if verified is not None:
                        available[row.chunk_id] = verified
                valid = []
                for candidate in fused:
                    chunk = available.get(candidate.chunk_id)
                    if (
                        chunk is None
                        or candidate.index_version != self.version_id
                        or chunk.index_hash != candidate.index_hash
                        or chunk.chunk.document_id != candidate.document_id
                    ):
                        continue
                    valid.append(
                        VerifiedHybridCenter(
                            verified_chunk=chunk, fused_candidate=candidate
                        )
                    )
                self.valid_count = len(valid)
                self.invalid_count = len(fused) - len(valid)
                centers = tuple(valid[:50])
                chunks = {
                    c.verified_chunk.chunk.chunk_id: c.verified_chunk for c in centers
                }
                positions = tuple(
                    {
                        (
                            c.verified_chunk.chunk.document_id,
                            c.verified_chunk.chunk.chunk_index + delta,
                        )
                        for c in centers
                        if c.verified_chunk.expandable
                        for delta in (-1, 1)
                        if c.verified_chunk.chunk.chunk_index + delta >= 0
                    }
                )
                neighbors = repo.read_neighbor_rows(self.version_id, positions)
                allowed = {
                    (
                        c.verified_chunk.chunk.document_id,
                        c.verified_chunk.section_instance_id,
                    )
                    for c in centers
                    if c.verified_chunk.expandable
                }
                for row, confirmation in neighbors:
                    verified = verify(row, confirmation)
                    if (
                        verified is not None
                        and verified.expandable
                        and (row.document_id, verified.section_instance_id) in allowed
                    ):
                        chunks[row.chunk_id] = verified
                return centers, chunks
        except Exception as exc:
            if isinstance(exc, HybridRetrievalError):
                raise
            raise HybridRetrievalError("hybrid_authority_failed") from exc
