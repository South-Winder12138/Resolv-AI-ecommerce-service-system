"""Short SQL transactions around idempotent v3 writes and guarded confirmation."""

from collections.abc import Callable
from contextlib import contextmanager

from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.schemas.hybrid_knowledge import HybridBuildResult, HybridRetrievalError


class HybridKnowledgeIndexer:
    def __init__(
        self,
        session_factory,
        embedding,
        store,
        profile,
        lock,
        *,
        fault_hook: Callable[[str], None] | None = None,
    ):
        self.session_factory = session_factory
        self.embedding = embedding
        self.store = store
        self.profile = profile
        self.lock = lock
        self.fault_hook = fault_hook
        if (
            embedding.profile != profile
            or embedding.dimension != 1024
            or store.profile != profile
        ):
            raise HybridRetrievalError("hybrid_index_resource_mismatch")

    def _hook(self, stage):
        if self.fault_hook is not None:
            self.fault_hook(stage)

    @contextmanager
    def _session(self, *, write=False, before_commit=None, after_commit=None):
        session = None
        primary = None
        try:
            session = self.session_factory()
            yield session
            if write:
                self.lock.assert_owned()
                if before_commit:
                    self._hook(before_commit)
                session.commit()
                if after_commit:
                    self._hook(after_commit)
        except BaseException as exc:
            primary = exc
            if session is not None:
                try:
                    session.rollback()
                except Exception as rollback:  # noqa: BLE001 -- retain the primary operation failure
                    exc.add_note(
                        "Hybrid SQL rollback also failed: " + type(rollback).__name__
                    )
            raise
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as exc:
                    if primary is None:
                        raise HybridRetrievalError("hybrid_sql_close_failed") from exc
                    primary.add_note("Hybrid SQL close also failed.")

    def prepare(self, version_id, manifest):
        try:
            self.lock.assert_owned()
            with self._session(write=True) as session:
                return HybridIndexRepository(session).create_version(
                    version_id, self.profile, manifest
                )
        except Exception as exc:
            if isinstance(exc, HybridRetrievalError):
                raise
            raise HybridRetrievalError("hybrid_prepare_failed") from exc

    def run(self, version_id: str, *, batch_size: int = 8) -> HybridBuildResult:
        if type(batch_size) is not int or batch_size <= 0:
            raise HybridRetrievalError("hybrid_batch_size_invalid")
        attempted = indexed = batches = 0
        snapshots = ()
        try:
            self.lock.assert_owned()
            with self._session() as session:
                repo = HybridIndexRepository(session)
                v = repo.get_version(version_id)
                summary = repo.summary(version_id)
                if v.profile != self.profile:
                    raise HybridRetrievalError("hybrid_index_profile_mismatch")
            # Once any SQL entries are done, never recreate a lost collection.
            if summary.state == "ready" or summary.done_count:
                self.store.validate_existing()
            else:
                self.store.create_or_validate()
            if summary.state == "ready":
                return HybridBuildResult(
                    summary=summary, attempted_count=0, indexed_count=0, batch_count=0
                )
            after = None
            while True:
                self.lock.assert_owned()
                with self._session() as session:
                    snapshots = HybridIndexRepository(session).list_pending(
                        version_id, after, batch_size
                    )
                if not snapshots:
                    break
                after = snapshots[-1].entry.manifest_chunk.chunk.chunk_id
                with self._session(write=True) as session:
                    count = HybridIndexRepository(session).begin_attempt(
                        version_id, snapshots
                    )
                    if count != len(snapshots):
                        raise HybridRetrievalError("hybrid_attempt_guard_failed")
                attempted += len(snapshots)
                batches += 1
                self.lock.assert_owned()
                vectors = self.embedding.embed_documents(
                    tuple(s.entry.manifest_chunk.chunk.text for s in snapshots)
                )
                self.lock.assert_owned()
                self._hook("before_upsert")
                count = self.store.upsert(tuple(s.entry for s in snapshots), vectors)
                self._hook("after_upsert")
                if type(count) is not int or count != len(snapshots):
                    raise HybridRetrievalError("hybrid_partial_upsert")
                self.lock.assert_owned()
                with self._session(
                    write=True,
                    before_commit="before_confirm_commit",
                    after_commit="after_confirm_commit",
                ) as session:
                    count = HybridIndexRepository(session).confirm_done(
                        version_id, snapshots
                    )
                    if count != len(snapshots):
                        raise HybridRetrievalError("hybrid_confirm_guard_failed")
                indexed += len(snapshots)
                snapshots = ()
            self.lock.assert_owned()
            self._hook("before_mark_ready")
            with self._session(write=True) as session:
                repo = HybridIndexRepository(session)
                repo.mark_ready(version_id)
                summary = repo.summary(version_id)
            return HybridBuildResult(
                summary=summary,
                attempted_count=attempted,
                indexed_count=indexed,
                batch_count=batches,
            )
        except Exception as exc:
            if snapshots:
                try:
                    self.lock.assert_owned()
                    with self._session(write=True) as session:
                        HybridIndexRepository(session).record_error(
                            version_id, snapshots, "hybrid_indexing_failed"
                        )
                except Exception as failure:  # noqa: BLE001 -- failed annotation cannot mask uncertain commit
                    exc.add_note(
                        "Hybrid safe-error recording failed: " + type(failure).__name__
                    )
            if isinstance(exc, HybridRetrievalError):
                raise
            raise HybridRetrievalError("hybrid_indexing_failed") from exc
