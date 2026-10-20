"""Plain knowledge contracts, independent of storage and query execution."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints

_NonblankText = Annotated[
    str, StringConstraints(strip_whitespace=True, strict=True, min_length=1)
]

BGE_M3_DIMENSION = 1024
_Hash64 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]


class KnowledgeIndexProfile(BaseModel):
    """Declared immutable indexing target; does not load or verify a model."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    collection_name: str = Field(
        default="resolv_knowledge_v2", pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,254}$"
    )
    passage_max_length: int = Field(default=512, gt=0, le=8192)
    model_name: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    dimension: int = Field(default=BGE_M3_DIMENSION, ge=BGE_M3_DIMENSION, le=BGE_M3_DIMENSION)
    normalize_embeddings: Literal[True] = True
    use_fp16: Literal[False] = False
    text_rule: Literal["category-questions-answer-v1"] = "category-questions-answer-v1"
    schema_version: Literal["knowledge-v2"] = "knowledge-v2"


class KnowledgeStageResult(BaseModel):
    """Plain committed staging outcome, independent of Session lifetime."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    document_id: str
    source: str
    content_type: str
    ingest_hash: str
    outcome: Literal["created", "unchanged"]
    chunk_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    done_count: int = Field(ge=0)


class Document(BaseModel):
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str


class Chunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)


class KnowledgeFileConfig(BaseModel):
    """Explicit per-source business type and literal body keyword rules."""

    model_config = ConfigDict(extra="forbid")

    content_type: _NonblankText
    key_clause_keywords: tuple[_NonblankText, ...] = ()


class KnowledgeChunk(Chunk):
    """Structured ingestion payload; query scores and SQL state live elsewhere."""

    model_config = ConfigDict(extra="forbid")

    category: str
    questions: str
    answer: _NonblankText
    section_path: str
    content_type: _NonblankText
    is_key_clause: StrictInt = Field(ge=0, le=1)


class SearchHit(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    score: float = Field(allow_inf_nan=False)


class KnowledgeIndexEntry(BaseModel):
    """Explicit v2 scalar projection; dense vectors are supplied separately."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    chunk_id: _Hash64
    document_id: _Hash64
    chunk_index: StrictInt = Field(ge=0, lt=2**63)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section_path: str
    content_type: str = Field(min_length=1)
    is_key_clause: StrictInt = Field(ge=0, le=1)
    index_hash: _Hash64


class KnowledgeCandidate(BaseModel):
    """Opaque index candidate; authoritative text must be read from SQL."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    chunk_id: _Hash64
    document_id: _Hash64
    index_hash: _Hash64
    score: float = Field(allow_inf_nan=False)


class KnowledgeSearchHit(KnowledgeChunk):
    score: float = Field(allow_inf_nan=False)


class KnowledgeIndexSnapshot(KnowledgeChunk):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    index_hash: _Hash64


class KnowledgeIndexResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    attempted_count: int = Field(ge=0)
    indexed_count: int = Field(ge=0)
    batch_count: int = Field(ge=0)
