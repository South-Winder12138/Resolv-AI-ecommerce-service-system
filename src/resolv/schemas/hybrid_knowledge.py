"""Independent hybrid contracts. No storage, model loading, or network effects."""
import json
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from resolv.schemas.knowledge import Document, KnowledgeChunk, KnowledgeFileConfig

Hash64 = Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]
Revision = Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{40}$')]
Name = Annotated[str, StringConstraints(pattern=r'^[A-Za-z_][A-Za-z0-9_]{0,254}$')]
Version = Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9_][A-Za-z0-9_-]{0,127}$')]
Count = Annotated[int, Field(ge=0)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


class HybridRetrievalError(RuntimeError):
    """Safe error code; retain dependency details only in the exception chain."""
    def __init__(self, code: str, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(code)


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)


class FrozenDocument(Document):
    model_config = FrozenContract.model_config


class FrozenChunk(KnowledgeChunk):
    model_config = FrozenContract.model_config


class FrozenFileConfig(KnowledgeFileConfig):
    model_config = FrozenContract.model_config


def _freeze(value, target):
    if isinstance(value, BaseModel):
        payload = value.model_dump()
        result = target.model_validate(payload)
        if result.model_dump() != payload:
            raise ValueError('Payload normalization would hide a source change')
        return result
    return value


class HybridIndexProfile(FrozenContract):
    collection_name: Name
    model_name: Literal['BAAI/bge-m3'] = 'BAAI/bge-m3'
    model_revision: Revision = '5617a9f61b028005a4858fdac845db406aefb181'
    dimension: Literal[1024] = 1024
    normalize_embeddings: Literal[True] = True
    device: Literal['cpu'] = 'cpu'
    precision: Literal['fp32'] = 'fp32'
    use_fp16: Literal[False] = False
    passage_max_length: int = Field(default=512, gt=0, le=8192)
    text_rule: Literal['category-questions-answer-v1'] = 'category-questions-answer-v1'
    schema_version: Literal['knowledge-v3'] = 'knowledge-v3'
    dense_index_type: Literal['FLAT'] = 'FLAT'
    dense_metric: Literal['COSINE'] = 'COSINE'
    sparse_index_type: Literal['SPARSE_INVERTED_INDEX'] = 'SPARSE_INVERTED_INDEX'
    sparse_metric: Literal['BM25'] = 'BM25'
    analyzer: Literal['chinese'] = 'chinese'
    bm25_k1: Literal[1.2] = 1.2
    bm25_b: Literal[0.75] = 0.75
    section_rule: Literal['document-ordinal-v1'] = 'document-ordinal-v1'


class HybridRetrievalProfile(FrozenContract):
    reranker_name: Literal['BAAI/bge-reranker-v2-m3'] = 'BAAI/bge-reranker-v2-m3'
    reranker_revision: Revision = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
    device: Literal['mps'] = 'mps'
    precision: Literal['fp32'] = 'fp32'
    batch_size: Literal[4] = 4
    query_max_tokens: Literal[512] = 512
    pair_max_tokens: Literal[3072] = 3072
    dense_limit: Literal[50] = 50
    bm25_limit: Literal[50] = 50
    center_limit: Literal[50] = 50
    output_limit: Literal[10] = 10
    rrf_k: Literal[60] = 60
    rrf_rule: Literal['equal-original-ranks-v1'] = 'equal-original-ranks-v1'
    window_rule: Literal['same-section-adjacent-one-v1'] = 'same-section-adjacent-one-v1'
    serialization_rule: Literal['title-boundaries-v1'] = 'title-boundaries-v1'
    sort_rule: Literal['logit-best-rrf-window-id-v1'] = 'logit-best-rrf-window-id-v1'


class HybridSourceDocument(FrozenContract):
    document: FrozenDocument
    ingest_hash: Hash64
    chunks: tuple[FrozenChunk, ...]

    @field_validator('document', mode='before')
    @classmethod
    def freeze_document(cls, value):
        return _freeze(value, FrozenDocument)

    @field_validator('chunks', mode='before')
    @classmethod
    def freeze_chunks(cls, value):
        return tuple(_freeze(x, FrozenChunk) for x in value) if isinstance(value, tuple) else value


class HybridManifestChunk(FrozenContract):
    chunk: FrozenChunk
    content_hash: Hash64
    section_instance_id: Hash64
    expandable: bool

    @field_validator('chunk', mode='before')
    @classmethod
    def freeze_chunk(cls, value):
        return _freeze(value, FrozenChunk)


class HybridManifestDocument(FrozenContract):
    document_id: Hash64
    document_hash: Hash64
    ingest_hash: Hash64
    file_config: FrozenFileConfig
    max_chars: int = Field(gt=0)
    overlap: Count
    chunks: tuple[HybridManifestChunk, ...]

    @field_validator('file_config', mode='before')
    @classmethod
    def freeze_config(cls, value, info):
        # A before validator hands nested dictionaries back as Python input.
        # Restore JSON validation for strict tuple fields on persisted JSON.
        if info.mode == 'json' and isinstance(value, dict):
            return FrozenFileConfig.model_validate_json(json.dumps(value))
        return _freeze(value, FrozenFileConfig)

    @model_validator(mode='after')
    def validate_chunks(self) -> Self:
        if self.overlap >= self.max_chars:
            raise ValueError('Invalid overlap')
        for i, entry in enumerate(self.chunks):
            if entry.chunk.document_id != self.document_id or entry.chunk.chunk_index != i:
                raise ValueError('Manifest chunks must be complete and in document order')
        if len({x.chunk.chunk_id for x in self.chunks}) != len(self.chunks):
            raise ValueError('Duplicate manifest chunk')
        return self


class HybridCorpusManifest(FrozenContract):
    documents: tuple[HybridManifestDocument, ...]
    manifest_rule: Literal['explicit-sql-corpus-v1'] = 'explicit-sql-corpus-v1'

    @model_validator(mode='after')
    def validate_documents(self) -> Self:
        ids = [x.document_id for x in self.documents]
        if ids != sorted(set(ids)):
            raise ValueError('Documents must be unique and sorted')
        return self


class HybridIndexEntry(FrozenContract):
    index_version: Version
    index_hash: Hash64
    manifest_chunk: HybridManifestChunk

    def milvus_projection(self) -> dict[str, str]:
        c = self.manifest_chunk.chunk
        return {'chunk_id': c.chunk_id, 'document_id': c.document_id,
                    'index_version': self.index_version, 'index_hash': self.index_hash, 'text': c.text}


class HybridIndexSnapshot(FrozenContract):
    entry: HybridIndexEntry
    document_hash: Hash64
    content_hash: Hash64
    ingest_hash: Hash64


class HybridCandidate(FrozenContract):
    index_version: Version
    chunk_id: Hash64
    document_id: Hash64
    index_hash: Hash64
    route: Literal['dense', 'bm25']
    rank: int = Field(ge=1, le=50)
    raw_score: Finite

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.index_version, self.chunk_id, self.index_hash


class FusedHybridCandidate(FrozenContract):
    index_version: Version
    chunk_id: Hash64
    document_id: Hash64
    index_hash: Hash64
    dense: HybridCandidate | None = None
    bm25: HybridCandidate | None = None
    rrf_score: Finite = Field(gt=0)

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.index_version, self.chunk_id, self.index_hash

    @property
    def best_rank(self) -> int:
        return min(x.rank for x in (self.dense, self.bm25) if x is not None)

    @model_validator(mode='after')
    def validate_routes(self) -> Self:
        if self.dense is None and self.bm25 is None:
            raise ValueError('Fused candidate needs a route')
        for route in ('dense', 'bm25'):
            value = getattr(self, route)
            if value is not None and (value.route != route or value.identity != self.identity
                                      or value.document_id != self.document_id):
                raise ValueError('Inconsistent candidate identity')
        return self


class VerifiedHybridChunk(HybridManifestChunk):
    document_hash: Hash64
    index_hash: Hash64


class VerifiedHybridCenter(FrozenContract):
    verified_chunk: VerifiedHybridChunk
    fused_candidate: FusedHybridCandidate

    @model_validator(mode='after')
    def validate_center(self) -> Self:
        c, f = self.verified_chunk, self.fused_candidate
        if (c.chunk.chunk_id, c.chunk.document_id, c.index_hash) != (f.chunk_id, f.document_id, f.index_hash):
            raise ValueError('Center identity mismatch')
        return self


class WindowReferences(FrozenContract):
    window_id: Hash64
    document_id: Hash64
    section_instance_id: Hash64
    chunk_ids: tuple[Hash64, ...] = Field(min_length=1, max_length=3)
    center_chunk_ids: tuple[Hash64, ...] = Field(min_length=1)
    center_candidates: tuple[FusedHybridCandidate, ...] = Field(min_length=1)
    best_rrf: Finite = Field(gt=0)

    @model_validator(mode='after')
    def validate_references(self) -> Self:
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError('Duplicate window chunk')
        if len(set(self.center_chunk_ids)) != len(self.center_chunk_ids):
            raise ValueError('Duplicate center')
        if not set(self.center_chunk_ids).issubset(self.chunk_ids):
            raise ValueError('Center outside window')
        if tuple(x.chunk_id for x in self.center_candidates) != self.center_chunk_ids:
            raise ValueError('Center provenance mismatch')
        if any(x.document_id != self.document_id for x in self.center_candidates):
            raise ValueError('Cross-document center')
        if self.best_rrf != max(x.rrf_score for x in self.center_candidates):
            raise ValueError('Incorrect window RRF')
        return self


class PreparedWindow(WindowReferences):
    body_text: str = Field(min_length=1)


class HybridEvidenceWindow(WindowReferences):
    rerank_score: Finite


class HybridVersionSnapshot(FrozenContract):
    version_id: Version
    profile: HybridIndexProfile
    manifest: HybridCorpusManifest
    state: Literal['building', 'ready']


class HybridVersionSummary(FrozenContract):
    version_id: Version
    collection_name: Name
    profile_hash: Hash64
    manifest_hash: Hash64
    state: Literal['building', 'ready']
    expected_count: Count
    pending_count: Count
    done_count: Count

    @model_validator(mode='after')
    def counts_match(self) -> Self:
        if self.pending_count + self.done_count != self.expected_count:
            raise ValueError('Version counts mismatch')
        if self.state == 'ready' and self.pending_count:
            raise ValueError('Ready version has pending entries')
        return self


class HybridBuildResult(FrozenContract):
    summary: HybridVersionSummary
    attempted_count: Count
    indexed_count: Count
    batch_count: Count


class HybridDiagnostics(FrozenContract):
    dense_count: int = Field(default=0, ge=0, le=50)
    bm25_count: int = Field(default=0, ge=0, le=50)
    fused_count: int = Field(default=0, ge=0, le=100)
    invalid_count: int = Field(default=0, ge=0, le=100)
    center_count: int = Field(default=0, ge=0, le=50)
    windows_before_merge: int = Field(default=0, ge=0, le=50)
    windows_after_merge: int = Field(default=0, ge=0, le=50)
    scored_count: int = Field(default=0, ge=0, le=50)
    selected_count: int = Field(default=0, ge=0, le=10)
    stage_seconds: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(default_factory=dict)
    query_tokens: Count = 0
    pair_tokens: tuple[Count, ...] = ()


class HybridRetrievalResult(FrozenContract):
    index_version: Version
    index_profile_hash: Hash64
    retrieval_profile: HybridRetrievalProfile
    windows: tuple[HybridEvidenceWindow, ...] = Field(max_length=10)
    chunks: dict[str, VerifiedHybridChunk]
    diagnostics: HybridDiagnostics

    @model_validator(mode='after')
    def validate_closure(self) -> Self:
        if len({w.window_id for w in self.windows}) != len(self.windows):
            raise ValueError('Duplicate evidence window')
        refs = {cid for w in self.windows for cid in w.chunk_ids}
        if refs != set(self.chunks) or len(self.chunks) > 30:
            raise ValueError('Body references must be closed and contain no orphans')
        for key, chunk in self.chunks.items():
            if key != chunk.chunk.chunk_id:
                raise ValueError('Body key mismatch')
        for w in self.windows:
            if any(self.chunks[cid].chunk.document_id != w.document_id
                   or self.chunks[cid].section_instance_id != w.section_instance_id for cid in w.chunk_ids):
                raise ValueError('Window body identity mismatch')
            if any(c.index_version != self.index_version for c in w.center_candidates):
                raise ValueError('Evidence version mismatch')
        return self
