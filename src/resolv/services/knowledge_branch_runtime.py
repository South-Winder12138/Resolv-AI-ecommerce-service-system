"""Lazy, application-owned version bindings and serialized pinned local inference."""

import json
import math
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import RLock
from time import monotonic

from pydantic import model_validator

from resolv.core.config import KnowledgeDatabaseSettings, MilvusSettings
from resolv.core.understanding_model import unique_object
from resolv.db.knowledge import knowledge_engine
from resolv.db.knowledge_hybrid import consistent_read_factory
from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, milvus_client
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.repositories.knowledge_hybrid_milvus import HybridMilvusStore
from resolv.schemas.hybrid_knowledge import (
    Hash64,
    HybridIndexProfile,
    HybridRetrievalResult,
    Name,
    Version,
)
from resolv.schemas.knowledge_branch import KnowledgeBranchError
from resolv.schemas.knowledge_release import KnowledgeContract, ReleaseSelection
from resolv.services.hybrid_authority import HybridAuthorityReader
from resolv.services.hybrid_embedding import HybridEmbeddingAdapter
from resolv.services.hybrid_fingerprints import index_hash, manifest_hash, profile_hash
from resolv.services.hybrid_reranker import INFERENCE_LOCK, LocalHybridReranker
from resolv.services.hybrid_retrieval import HybridKnowledgeRetriever

DEFAULT_RELEASE_SELECTOR = Path(__file__).resolve().parents[3] / "knowledge/active-release.json"


class _Selector(KnowledgeContract):
    enabled: bool
    config_version: int
    release_id: Version | None = None
    index_version: Version | None = None
    collection: Name | None = None
    profile_hash: Hash64 | None = None
    manifest_hash: Hash64 | None = None

    @model_validator(mode="after")
    def version(self):
        if self.config_version != 1:
            raise ValueError("Unknown selector version")
        if not self.enabled and any(
            getattr(self, key) is not None
            for key in (
                "release_id",
                "index_version",
                "collection",
                "profile_hash",
                "manifest_hash",
            )
        ):
            raise ValueError("Disabled selector carries unexpected identity")
        return self


def read_release_selection(path: Path) -> ReleaseSelection:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ReleaseSelection(status="disabled")
    except (OSError, UnicodeError):
        return ReleaseSelection(
            status="invalid", error_code="knowledge_release_selector_invalid"
        )
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
        selector = _Selector.model_validate_json(json.dumps(value, allow_nan=False))
        if not selector.enabled:
            return ReleaseSelection(status="disabled")
        profile = HybridIndexProfile(collection_name=selector.collection)
        return ReleaseSelection(
            status="configured",
            profile=profile,
            **selector.model_dump(exclude={"enabled"}),
        )
    except (ValueError, TypeError):
        return ReleaseSelection(
            status="invalid", error_code="knowledge_release_selector_invalid"
        )


class _DenseBinding:
    def __init__(self, dense, profile):
        if dense.profile.model_dump(exclude={"collection_name"}) != profile.model_dump(
            exclude={"collection_name"}
        ):
            raise KnowledgeBranchError("knowledge_dense_profile_mismatch")
        self.dense, self.profile = dense, profile
        self.dimension = dense.dimension

    def embed_query(self, text):
        return self.dense.embed_query(text)


class KnowledgeRetrieverPool:
    def __init__(
        self, *, knowledge_settings, milvus_settings, retrieval_profile, cache_dir=None
    ):
        self.knowledge_settings, self.milvus_settings = (
            knowledge_settings,
            milvus_settings,
        )
        self.retrieval_profile, self.cache_dir = retrieval_profile, cache_dir
        self._init_lock, self._resources = RLock(), ExitStack()
        self._factory = self._client = self._dense = self._reranker = None
        self._versions, self._retrievers = {}, {}
        self._closed = False

    @contextmanager
    def _locked(self, timeout=60):
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise KnowledgeBranchError("knowledge_queue_timeout_invalid")
        begin = monotonic()
        if not INFERENCE_LOCK.acquire(timeout=timeout):
            raise KnowledgeBranchError("knowledge_inference_busy", True)
        try:
            with self._init_lock:
                if self._closed:
                    raise KnowledgeBranchError("knowledge_runtime_closed")
                yield monotonic() - begin
        finally:
            INFERENCE_LOCK.release()

    def _connect(self):
        if self._factory is not None:
            return
        with ExitStack() as pending:
            ks = (
                self.knowledge_settings()
                if callable(self.knowledge_settings)
                else self.knowledge_settings
            )
            ms = (
                self.milvus_settings()
                if callable(self.milvus_settings)
                else self.milvus_settings
            )
            engine = pending.enter_context(knowledge_engine(ks))
            client = pending.enter_context(milvus_client(ms))
            factory = consistent_read_factory(engine)
            self._resources.enter_context(pending.pop_all())
            self._factory, self._client = factory, client

    def _version(self, selection):
        selection = ReleaseSelection.model_validate_json(selection.model_dump_json())
        if selection.status != "configured":
            raise KnowledgeBranchError(
                "knowledge_release_disabled"
                if selection.status == "disabled"
                else "knowledge_release_selector_invalid"
            )
        if selection.fingerprint in self._versions:
            return self._versions[selection.fingerprint]
        self._connect()
        with self._factory() as session:
            repo = HybridIndexRepository(session)
            version = repo.get_version(selection.index_version)
            counts = repo.summary(selection.index_version)
        if (
            version.state != "ready"
            or version.profile != selection.profile
            or profile_hash(version.profile) != selection.profile_hash
            or manifest_hash(version.manifest) != selection.manifest_hash
            or counts.pending_count
            or counts.done_count != counts.expected_count
            or not counts.expected_count
        ):
            raise KnowledgeBranchError("knowledge_release_not_ready")
        store = HybridMilvusStore(self._client, version.profile)
        store.validate_existing()
        expected = {
            c.chunk.chunk_id: {
                "chunk_id": c.chunk.chunk_id,
                "document_id": d.document_id,
                "index_version": version.version_id,
                "index_hash": index_hash(c, version.profile, version.version_id),
                "text": c.chunk.text,
            }
            for d in version.manifest.documents
            for c in d.chunks
        }
        actual_count = self._client.query(
            collection_name=selection.collection,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
            timeout=MILVUS_TIMEOUT_SECONDS,
        )
        if actual_count != [{"count(*)": len(expected)}]:
            raise KnowledgeBranchError("knowledge_release_collection_incomplete")
        ids = list(expected)
        for start in range(0, len(ids), 100):
            batch = ids[start : start + 100]
            rows = self._client.get(
                collection_name=selection.collection,
                ids=batch,
                output_fields=[
                    "chunk_id",
                    "document_id",
                    "index_version",
                    "index_hash",
                    "text",
                ],
                consistency_level="Strong",
                timeout=MILVUS_TIMEOUT_SECONDS,
            )
            if (
                len(rows) != len(batch)
                or {r.get("chunk_id") for r in rows} != set(batch)
                or any(r != expected[r["chunk_id"]] for r in rows)
            ):
                raise KnowledgeBranchError("knowledge_release_collection_incomplete")
        binding = (
            version,
            store,
            HybridAuthorityReader(self._factory, version.version_id, version.profile),
        )
        self._versions[selection.fingerprint] = binding
        return binding

    def validate_selection(self, selection):
        with self._locked():
            return self._version(selection)[0]

    def _ranker(self):
        if self._reranker is None:
            ranker = LocalHybridReranker(
                self.retrieval_profile, cache_dir=self.cache_dir
            )
            self._resources.callback(ranker.close)
            self._reranker = ranker
        return self._reranker

    def count_query_tokens(self, text):
        with self._locked():
            return self._ranker().count_query_tokens(text)

    def retrieve(self, selection, query, *, bm25_text, timeout_seconds=60):
        with self._locked(timeout_seconds) as waited:
            version, store, authority = self._version(selection)
            if self._dense is None:
                dense = HybridEmbeddingAdapter(
                    version.profile, cache_dir=self.cache_dir
                )
                try:
                    self._ranker()
                except BaseException as exc:
                    try:
                        dense.close()
                    except Exception as cleanup:  # noqa: BLE001 -- retain the primary initialization failure
                        exc.add_note(f"Dense cleanup failed: {type(cleanup).__name__}")
                    raise
                self._dense = dense
                self._resources.callback(dense.close)
            if selection.fingerprint not in self._retrievers:
                self._retrievers[selection.fingerprint] = HybridKnowledgeRetriever(
                    _DenseBinding(self._dense, version.profile),
                    store,
                    authority,
                    self._ranker(),
                    version.profile,
                    self.retrieval_profile,
                )
            result = self._retrievers[selection.fingerprint].retrieve(
                query, bm25_text=bm25_text
            )
            value = result.model_dump(mode="json")
            value["diagnostics"]["stage_seconds"]["queue"] = waited
            return HybridRetrievalResult.model_validate_json(json.dumps(value))

    def close(self):
        with INFERENCE_LOCK, self._init_lock:
            if self._closed:
                return
            self._closed = True
            for retriever in self._retrievers.values():
                retriever._active = False
            self._resources.close()
            self._factory = self._client = self._dense = self._reranker = None


def default_knowledge_retrievers():
    from resolv.schemas.hybrid_knowledge import HybridRetrievalProfile

    return KnowledgeRetrieverPool(
        knowledge_settings=KnowledgeDatabaseSettings,
        milvus_settings=MilvusSettings,
        retrieval_profile=HybridRetrievalProfile(),
    )
