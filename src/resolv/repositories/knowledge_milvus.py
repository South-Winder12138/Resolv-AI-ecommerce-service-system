"""Milvus v2 scalar/vector storage reusing the existing dense lifecycle."""
from collections.abc import Sequence
from typing import ClassVar

from pymilvus import DataType

from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, VectorStoreError
from resolv.repositories.milvus import MilvusVectorStore
from resolv.schemas.knowledge import (
    BGE_M3_DIMENSION,
    KnowledgeCandidate,
    KnowledgeIndexEntry,
)


class KnowledgeMilvusVectorStore(MilvusVectorStore):
    _varchar_limits: ClassVar[dict[str, int]] = {
        "chunk_id": 64, "document_id": 64, "text": 8192, "source": 4096,
        "section_path": 4096, "content_type": 255, "index_hash": 64,
    }
    _integer_fields: ClassVar[dict[str, DataType]] = {"chunk_index": DataType.INT64, "is_key_clause": DataType.INT8}

    @property
    def collection_name(self) -> str:
        return self._collection

    def ensure_collection(self, dimension: int = BGE_M3_DIMENSION) -> None:
        if type(dimension) is not int or dimension != BGE_M3_DIMENSION:
            raise ValueError("knowledge v2 requires the fixed dense dimension")
        super().ensure_collection(dimension)

    def _validate_schema(self, description: dict, dimension: int | None) -> int:
        actual = super()._validate_schema(description, dimension)
        if actual != BGE_M3_DIMENSION:
            raise ValueError("knowledge v2 dense dimension mismatch")
        return actual

    def upsert(
        self, entries: Sequence[KnowledgeIndexEntry], vectors: Sequence[Sequence[float]],
    ) -> int:
        if len(entries) != len(vectors):
            raise ValueError("Entry and vector counts must match")
        if not entries:
            return 0
        rows, ids = [], set()
        for entry in entries:
            if not isinstance(entry, KnowledgeIndexEntry):
                raise ValueError("upsert requires explicit knowledge index entries")  # noqa: TRY004
            validated = KnowledgeIndexEntry.model_validate(entry.model_dump())
            for name, limit in self._varchar_limits.items():
                value = getattr(validated, name)
                if (name != "section_path" and not value.strip()) or len(value.encode("utf-8")) > limit:
                    raise ValueError(f"{name} exceeds its nonblank/UTF-8 capacity contract")
            if validated.chunk_id in ids:
                raise ValueError("duplicate chunk_id in one upsert batch")
            ids.add(validated.chunk_id)
            rows.append(validated.model_dump())
        values = self._vectors(vectors, len(rows))
        for row, vector in zip(rows, values, strict=True):
            row["dense_vector"] = vector
        try:
            response = self._client.upsert(
                collection_name=self._collection, data=rows, timeout=MILVUS_TIMEOUT_SECONDS,
            )
            count = response["upsert_count"]
            if type(count) is not int or not 0 <= count <= len(rows):
                raise ValueError("Milvus returned an invalid upsert count")
            return count
        except Exception as exc:
            raise VectorStoreError("Knowledge v2 upsert failed.") from exc

    def search(self, vector: Sequence[float], top_k: int) -> list[KnowledgeCandidate]:
        if type(top_k) is not int or not 0 < top_k < 16384:
            raise ValueError("top_k must be a positive Milvus-supported integer")
        values = self._vectors([vector], 1)
        try:
            response = self._client.search(
                collection_name=self._collection, data=values, anns_field="dense_vector",
                limit=top_k, output_fields=["chunk_id", "document_id", "index_hash"],
                search_params={"metric_type": "COSINE", "params": {}},
                consistency_level="Strong", timeout=MILVUS_TIMEOUT_SECONDS,
            )
            if (
                not isinstance(response, list) or len(response) != 1
                or not isinstance(response[0], list) or len(response[0]) > top_k
            ):
                raise ValueError("Milvus returned an invalid candidate container")
            candidates = []
            for raw in response[0]:
                primary = raw["chunk_id"] if "chunk_id" in raw else raw["id"]
                entity = raw["entity"]
                if ("id" in raw and raw["id"] != primary) or entity.get("chunk_id", primary) != primary:
                    raise ValueError("Milvus candidate primary key mismatch")
                candidates.append(KnowledgeCandidate(
                    chunk_id=primary, document_id=entity["document_id"],
                    index_hash=entity["index_hash"], score=raw["distance"],
                ))
            return candidates
        except Exception as exc:
            raise VectorStoreError("Knowledge v2 search or candidate mapping failed.") from exc
