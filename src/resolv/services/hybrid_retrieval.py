"""Independent query composition, strict failure, without LLM or chat input handling."""

import time

from resolv.schemas.hybrid_knowledge import (
    HybridDiagnostics,
    HybridRetrievalError,
    HybridRetrievalResult,
)
from resolv.services.hybrid_fingerprints import profile_hash
from resolv.services.hybrid_ranking import fuse_rrf
from resolv.services.hybrid_reranker import INFERENCE_LOCK
from resolv.services.hybrid_windows import build_windows, select_evidence


class HybridKnowledgeRetriever:
    def __init__(
        self, embedding, store, authority, reranker, index_profile, retrieval_profile
    ):
        if (
            embedding.profile != index_profile
            or embedding.dimension != 1024
            or store.profile != index_profile
            or authority.profile != index_profile
            or reranker.profile != retrieval_profile
        ):
            raise HybridRetrievalError("hybrid_retrieval_profile_mismatch")
        self.embedding = embedding
        self.store = store
        self.authority = authority
        self.reranker = reranker
        self.index_profile = index_profile
        self.retrieval_profile = retrieval_profile
        self._active = True

    def retrieve(
        self, query: str, *, bm25_text: str | None = None
    ) -> HybridRetrievalResult:
        with INFERENCE_LOCK:
            if not self._active:
                raise HybridRetrievalError("hybrid_runtime_closed")
            if not isinstance(query, str) or not query.strip():
                raise HybridRetrievalError("invalid_query")
            if bm25_text is not None and (
                not isinstance(bm25_text, str) or not bm25_text.strip()
            ):
                raise HybridRetrievalError("invalid_bm25_text")
            started = time.perf_counter()
            times = {}

            def stage(name, operation):
                begin = time.perf_counter()
                value = operation()
                times[name] = time.perf_counter() - begin
                return value

            try:
                vector = stage("embedding", lambda: self.embedding.embed_query(query))
                version = self.authority.version_id
                dense = stage(
                    "dense",
                    lambda: self.store.search_dense(
                        vector,
                        index_version=version,
                        limit=self.retrieval_profile.dense_limit,
                    ),
                )
                bm25 = stage(
                    "bm25",
                    lambda: self.store.search_bm25(
                        query if bm25_text is None else bm25_text,
                        index_version=version,
                        limit=self.retrieval_profile.bm25_limit,
                    ),
                )
                fused = stage(
                    "fusion",
                    lambda: fuse_rrf(dense, bm25, k=self.retrieval_profile.rrf_k),
                )
                centers = ()
                chunks = {}
                windows = ()
                selected = ()
                bodies = {}
                scores = ()
                invalid = 0
                if fused:
                    centers, chunks = stage(
                        "authority", lambda: self.authority.prepare(fused)
                    )
                    invalid = self.authority.invalid_count
                    windows = stage("windows", lambda: build_windows(centers, chunks))
                    if windows:
                        scores = stage(
                            "rerank", lambda: self.reranker.score(query, windows)
                        )
                        selected, bodies = stage(
                            "selection",
                            lambda: select_evidence(
                                windows,
                                scores,
                                chunks,
                                limit=self.retrieval_profile.output_limit,
                            ),
                        )
                times["total"] = time.perf_counter() - started
                return HybridRetrievalResult(
                    index_version=version,
                    index_profile_hash=profile_hash(self.index_profile),
                    retrieval_profile=self.retrieval_profile,
                    windows=selected,
                    chunks=bodies,
                    diagnostics=HybridDiagnostics(
                        dense_count=len(dense),
                        bm25_count=len(bm25),
                        fused_count=len(fused),
                        invalid_count=invalid,
                        center_count=len(centers),
                        windows_before_merge=len(centers),
                        windows_after_merge=len(windows),
                        scored_count=len(scores),
                        selected_count=len(selected),
                        stage_seconds=times,
                        query_tokens=self.reranker.query_tokens if windows else 0,
                        pair_tokens=self.reranker.pair_tokens if windows else (),
                    ),
                )
            except Exception as exc:
                if isinstance(exc, HybridRetrievalError):
                    raise
                raise HybridRetrievalError("hybrid_retrieval_failed") from exc
