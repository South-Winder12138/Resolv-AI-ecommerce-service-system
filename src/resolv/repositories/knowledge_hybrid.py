"""Versioned SQL access. Caller commits/closes; plain snapshots cross session boundaries."""

from collections.abc import Sequence

from sqlalchemy import LargeBinary, cast, exists, func, select, tuple_, update
from sqlalchemy.orm import Session

from resolv.models.knowledge import KnowledgeChunkRecord, KnowledgeDocument
from resolv.models.knowledge_hybrid import KnowledgeChunkIndex, KnowledgeIndexVersion
from resolv.models.order import utc_now
from resolv.schemas.hybrid_knowledge import (
    HybridCorpusManifest,
    HybridIndexEntry,
    HybridIndexProfile,
    HybridIndexSnapshot,
    HybridRetrievalError,
    HybridSourceDocument,
    HybridVersionSnapshot,
    HybridVersionSummary,
)
from resolv.schemas.knowledge import Document, KnowledgeChunk
from resolv.services.hybrid_fingerprints import (
    index_hash,
    manifest_hash,
    profile_hash,
)
from resolv.services.hybrid_manifest import build_corpus_manifest


def _exact(column, value):
    return (
        cast(column, LargeBinary) == value.encode("utf-8")
        if isinstance(value, str)
        else column == value
    )


class HybridIndexRepository:
    def __init__(self, session: Session):
        self.session = session

    def read_candidate_rows(self, version_id, identities):
        ids = tuple({x.chunk_id for x in identities})
        if not ids:
            return ()
        return tuple(
            self.session.execute(
                select(KnowledgeChunkRecord, KnowledgeChunkIndex)
                .join(
                    KnowledgeChunkIndex,
                    KnowledgeChunkIndex.chunk_id == KnowledgeChunkRecord.chunk_id,
                )
                .where(
                    KnowledgeChunkIndex.index_version == version_id,
                    KnowledgeChunkRecord.chunk_id.in_(ids),
                )
            ).all()
        )

    def read_documents(self, document_ids):
        if not document_ids:
            return {}
        return {
            d.document_id: d
            for d in self.session.scalars(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.document_id.in_(document_ids)
                )
            )
        }

    def read_neighbor_rows(self, version_id, positions):
        if not positions:
            return ()
        return tuple(
            self.session.execute(
                select(KnowledgeChunkRecord, KnowledgeChunkIndex)
                .join(
                    KnowledgeChunkIndex,
                    KnowledgeChunkIndex.chunk_id == KnowledgeChunkRecord.chunk_id,
                )
                .where(
                    KnowledgeChunkIndex.index_version == version_id,
                    tuple_(
                        KnowledgeChunkRecord.document_id,
                        KnowledgeChunkRecord.chunk_index,
                    ).in_(positions),
                )
            ).all()
        )

    def get_version(self, version_id: str) -> HybridVersionSnapshot:
        row = self.session.get(KnowledgeIndexVersion, version_id)
        if row is None:
            raise HybridRetrievalError("hybrid_version_missing")
        try:
            profile = HybridIndexProfile.model_validate_json(row.profile_json)
            manifest = HybridCorpusManifest.model_validate_json(row.manifest_json)
            if (
                profile_hash(profile) != row.profile_hash
                or manifest_hash(manifest) != row.manifest_hash
                or profile.collection_name != row.collection_name
            ):
                raise ValueError("Version fingerprint mismatch")
            return HybridVersionSnapshot(
                version_id=row.version_id,
                profile=profile,
                manifest=manifest,
                state=row.state,
            )
        except Exception as exc:
            raise HybridRetrievalError("hybrid_version_corrupt") from exc

    def summary(self, version_id: str) -> HybridVersionSummary:
        version = self.get_version(version_id)
        counts = dict(
            self.session.execute(
                select(KnowledgeChunkIndex.index_status, func.count())
                .where(KnowledgeChunkIndex.index_version == version_id)
                .group_by(KnowledgeChunkIndex.index_status)
            ).all()
        )
        expected = sum(len(d.chunks) for d in version.manifest.documents)
        return HybridVersionSummary(
            version_id=version_id,
            collection_name=version.profile.collection_name,
            profile_hash=profile_hash(version.profile),
            manifest_hash=manifest_hash(version.manifest),
            state=version.state,
            expected_count=expected,
            pending_count=counts.get("pending", 0),
            done_count=counts.get("done", 0),
        )

    def read_sources(
        self, document_ids: Sequence[str]
    ) -> tuple[HybridSourceDocument, ...]:
        if not document_ids:
            return ()
        documents = self.session.scalars(
            select(KnowledgeDocument).where(
                KnowledgeDocument.document_id.in_(document_ids)
            )
        ).all()
        chunks = self.session.scalars(
            select(KnowledgeChunkRecord)
            .where(KnowledgeChunkRecord.document_id.in_(document_ids))
            .order_by(
                KnowledgeChunkRecord.document_id, KnowledgeChunkRecord.chunk_index
            )
        ).all()
        if len(documents) != len(set(document_ids)):
            raise HybridRetrievalError("hybrid_source_missing")
        result = []
        try:
            for d in sorted(documents, key=lambda d: d.document_id):
                payloads = []
                for c in chunks:
                    if c.document_id != d.document_id:
                        continue
                    raw = {
                        name: getattr(c, name) for name in KnowledgeChunk.model_fields
                    }
                    chunk = KnowledgeChunk.model_validate(raw)
                    if (
                        chunk.model_dump() != raw
                        or c.source != d.source
                        or c.content_type != d.content_type
                    ):
                        raise ValueError("Source normalization or metadata mismatch")
                    payloads.append(chunk)
                result.append(
                    HybridSourceDocument(
                        document=Document(
                            document_id=d.document_id, source=d.source, text=d.raw_text
                        ),
                        ingest_hash=d.ingest_hash,
                        chunks=tuple(payloads),
                    )
                )
            return tuple(result)
        except Exception as exc:
            raise HybridRetrievalError("hybrid_source_invalid") from exc

    def _source_guards(self, version):
        sources = self.read_sources(
            tuple(d.document_id for d in version.manifest.documents)
        )
        source_map = {s.document.document_id: s for s in sources}
        guards = []
        for d in version.manifest.documents:
            src = source_map[d.document_id]
            rebuilt = build_corpus_manifest(
                (src,),
                {src.document.source: d.file_config},
                max_chars=d.max_chars,
                overlap=d.overlap,
            )
            if rebuilt.documents[0] != d:
                raise HybridRetrievalError("hybrid_source_changed")
            guards.append(
                exists(
                    select(KnowledgeDocument.document_id).where(
                        KnowledgeDocument.document_id == d.document_id,
                        _exact(KnowledgeDocument.raw_text, src.document.text),
                        _exact(KnowledgeDocument.source, src.document.source),
                        _exact(KnowledgeDocument.ingest_hash, src.ingest_hash),
                        _exact(
                            KnowledgeDocument.content_type, d.file_config.content_type
                        ),
                    )
                )
            )
            for mc in d.chunks:
                guards.append(
                    exists(
                        select(KnowledgeChunkRecord.chunk_id).where(
                            *(
                                _exact(getattr(KnowledgeChunkRecord, key), value)
                                for key, value in mc.chunk.model_dump().items()
                            )
                        )
                    )
                )
        return guards

    def create_version(
        self,
        version_id: str,
        profile: HybridIndexProfile,
        manifest: HybridCorpusManifest,
    ) -> HybridVersionSummary:
        version = HybridVersionSnapshot(
            version_id=version_id, profile=profile, manifest=manifest, state="building"
        )
        prior = self.session.get(KnowledgeIndexVersion, version_id)
        if prior is not None:
            existing = self.get_version(version_id)
            if existing.profile != profile or existing.manifest != manifest:
                raise HybridRetrievalError("hybrid_version_conflict")
            return self.summary(version_id)
        if self.session.scalar(
            select(KnowledgeIndexVersion.version_id).where(
                KnowledgeIndexVersion.collection_name == profile.collection_name
            )
        ):
            raise HybridRetrievalError("hybrid_collection_already_bound")
        self._source_guards(version)
        self.session.add(
            KnowledgeIndexVersion(
                version_id=version_id,
                collection_name=profile.collection_name,
                profile_json=profile.model_dump_json(),
                profile_hash=profile_hash(profile),
                manifest_json=manifest.model_dump_json(),
                manifest_hash=manifest_hash(manifest),
            )
        )
        self.session.flush()
        for d in manifest.documents:
            for mc in d.chunks:
                self.session.add(
                    KnowledgeChunkIndex(
                        index_version=version_id,
                        chunk_id=mc.chunk.chunk_id,
                        document_id=d.document_id,
                        document_hash=d.document_hash,
                        content_hash=mc.content_hash,
                        section_instance_id=mc.section_instance_id,
                        expandable=int(mc.expandable),
                        index_hash=index_hash(mc, profile, version_id),
                    )
                )
        self.session.flush()
        return self.summary(version_id)

    def list_pending(
        self, version_id: str, after_id: str | None, limit: int
    ) -> tuple[HybridIndexSnapshot, ...]:
        if type(limit) is not int or limit <= 0:
            raise HybridRetrievalError("hybrid_batch_size_invalid")
        v = self.get_version(version_id)
        q = select(KnowledgeChunkIndex).where(
            KnowledgeChunkIndex.index_version == version_id,
            KnowledgeChunkIndex.index_status == "pending",
        )
        if after_id is not None:
            q = q.where(KnowledgeChunkIndex.chunk_id > after_id)
        rows = self.session.scalars(
            q.order_by(KnowledgeChunkIndex.chunk_id).limit(limit)
        ).all()
        entries = {
            mc.chunk.chunk_id: (d, mc) for d in v.manifest.documents for mc in d.chunks
        }
        result = []
        for row in rows:
            if row.chunk_id not in entries:
                raise HybridRetrievalError("hybrid_index_row_corrupt")
            d, mc = entries[row.chunk_id]
            expected = index_hash(mc, v.profile, version_id)
            if (
                row.index_hash,
                row.document_hash,
                row.content_hash,
                row.section_instance_id,
                row.document_id,
                row.expandable,
            ) != (
                expected,
                d.document_hash,
                mc.content_hash,
                mc.section_instance_id,
                d.document_id,
                int(mc.expandable),
            ):
                raise HybridRetrievalError("hybrid_index_row_corrupt")
            result.append(
                HybridIndexSnapshot(
                    entry=HybridIndexEntry(
                        index_version=version_id, index_hash=expected, manifest_chunk=mc
                    ),
                    document_hash=d.document_hash,
                    content_hash=mc.content_hash,
                    ingest_hash=d.ingest_hash,
                )
            )
        return tuple(result)

    def _mutate(self, version_id, snapshots, values):
        v = self.get_version(version_id)
        if v.state != "building":
            raise HybridRetrievalError("hybrid_version_not_building")
        guards = self._source_guards(v)
        expected = {
            mc.chunk.chunk_id: (d, mc) for d in v.manifest.documents for mc in d.chunks
        }
        count = 0
        for snap in snapshots:
            entry = snap.entry
            d, mc = expected[entry.manifest_chunk.chunk.chunk_id]
            if (
                entry.index_version != version_id
                or entry.manifest_chunk != mc
                or snap.document_hash != d.document_hash
                or snap.ingest_hash != d.ingest_hash
                or snap.content_hash != mc.content_hash
                or entry.index_hash != index_hash(mc, v.profile, version_id)
            ):
                raise HybridRetrievalError("hybrid_snapshot_mismatch")
            statement = (
                update(KnowledgeChunkIndex)
                .where(
                    KnowledgeChunkIndex.index_version == version_id,
                    KnowledgeChunkIndex.chunk_id == mc.chunk.chunk_id,
                    KnowledgeChunkIndex.index_status == "pending",
                    KnowledgeChunkIndex.index_hash == entry.index_hash,
                    KnowledgeChunkIndex.document_hash == snap.document_hash,
                    KnowledgeChunkIndex.content_hash == snap.content_hash,
                    *guards,
                )
                .execution_options(synchronize_session=False)
            )
            count += self.session.execute(statement.values(**values(entry))).rowcount
        return count

    def begin_attempt(self, version_id, snapshots) -> int:
        return self._mutate(
            version_id,
            snapshots,
            lambda e: {"attempt_count": KnowledgeChunkIndex.attempt_count + 1},
        )

    def confirm_done(self, version_id, snapshots) -> int:
        return self._mutate(
            version_id,
            snapshots,
            lambda e: {
                "index_status": "done",
                "indexed_hash": e.index_hash,
                "indexed_at": utc_now(),
                "last_error": None,
            },
        )

    def record_error(self, version_id, snapshots, code: str) -> int:
        return self._mutate(
            version_id, snapshots, lambda e: {"last_error": "hybrid_indexing_failed"}
        )

    def mark_ready(self, version_id: str) -> None:
        v = self.get_version(version_id)
        guards = self._source_guards(v)
        rows = self.session.scalars(
            select(KnowledgeChunkIndex).where(
                KnowledgeChunkIndex.index_version == version_id
            )
        ).all()
        expected = {
            mc.chunk.chunk_id: index_hash(mc, v.profile, version_id)
            for d in v.manifest.documents
            for mc in d.chunks
        }
        if len(rows) != len(expected) or any(
            row.chunk_id not in expected
            or row.index_status != "done"
            or row.index_hash != expected[row.chunk_id]
            or row.indexed_hash != row.index_hash
            or row.indexed_at is None
            for row in rows
        ):
            raise HybridRetrievalError("hybrid_version_incomplete")
        if v.state == "ready":
            return
        not_done = exists(
            select(KnowledgeChunkIndex.chunk_id).where(
                KnowledgeChunkIndex.index_version == version_id,
                KnowledgeChunkIndex.index_status != "done",
            )
        )
        changed = self.session.execute(
            update(KnowledgeIndexVersion)
            .where(
                KnowledgeIndexVersion.version_id == version_id,
                KnowledgeIndexVersion.state == "building",
                KnowledgeIndexVersion.profile_hash == profile_hash(v.profile),
                KnowledgeIndexVersion.manifest_hash == manifest_hash(v.manifest),
                ~not_done,
                *guards,
            )
            .values(state="ready", ready_at=utc_now())
            .execution_options(synchronize_session=False)
        ).rowcount
        if changed != 1:
            raise HybridRetrievalError("hybrid_ready_guard_failed")
