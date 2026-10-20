"""Knowledge schema and dense vector storage; the caller owns the client."""

import re
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from pymilvus import DataType, MilvusClient

from resolv.core.config import MILVUS_COLLECTION_PATTERN
from resolv.db.milvus import MILVUS_TIMEOUT_SECONDS, VectorStoreError
from resolv.schemas.knowledge import Chunk, SearchHit

_VARCHAR_LIMITS = {"chunk_id": 64, "document_id": 64, "text": 8192, "source": 4096}
_VECTOR_FIELD = "dense_vector"
_INDEX_NAME = "dense_cosine"


class MilvusVectorStore:
    _varchar_limits: ClassVar[dict[str, int]] = _VARCHAR_LIMITS
    _integer_fields: ClassVar[dict[str, DataType]] = {"chunk_index": DataType.INT64}

    def __init__(self, client: MilvusClient, collection: str) -> None:
        if not isinstance(collection, str) or not re.fullmatch(
            MILVUS_COLLECTION_PATTERN, collection
        ):
            raise ValueError("collection must be a valid Milvus name")
        self._client = client
        self._collection = collection
        self._dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._dimension = None
        try:
            if not self._client.has_collection(
                collection_name=self._collection, timeout=MILVUS_TIMEOUT_SECONDS
            ):
                schema = self._client.create_schema(
                    auto_id=False, enable_dynamic_field=False
                )
                for name, limit in self._varchar_limits.items():
                    schema.add_field(
                        field_name=name,
                        datatype=DataType.VARCHAR,
                        is_primary=name == "chunk_id",
                        max_length=limit,
                    )
                for name, datatype in self._integer_fields.items():
                    schema.add_field(field_name=name, datatype=datatype)
                schema.add_field(
                    field_name=_VECTOR_FIELD,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=dimension,
                )
                self._client.create_collection(
                    collection_name=self._collection,
                    schema=schema,
                    timeout=MILVUS_TIMEOUT_SECONDS,
                )
            self._prepare_existing(dimension, create_index=True)
        except Exception as exc:
            raise VectorStoreError(
                "Milvus collection setup failed; check schema and index compatibility."
            ) from exc

    def _validate_schema(self, description: dict, dimension: int | None) -> int:
        if (
            description["auto_id"] is not False
            or description["enable_dynamic_field"] is not False
        ):
            raise ValueError(
                "knowledge schema requires explicit IDs and no dynamic fields"
            )
        fields = description["fields"]
        expected = {*self._varchar_limits, *self._integer_fields, _VECTOR_FIELD}
        if len(fields) != len(expected) or {f["name"] for f in fields} != expected:
            raise ValueError("knowledge schema fields do not match")
        actual_dimension = None
        for field in fields:
            name = field["name"]
            primary = field.get("is_primary", False)
            if primary != (name == "chunk_id") or field.get("nullable", False):
                raise ValueError("knowledge schema primary key or nullability mismatch")
            if name in self._varchar_limits:
                if (
                    field["type"] != DataType.VARCHAR
                    or int(field["params"]["max_length"]) != self._varchar_limits[name]
                ):
                    raise ValueError("knowledge string field type or capacity mismatch")
            elif name in self._integer_fields:
                if field["type"] != self._integer_fields[name]:
                    raise ValueError("knowledge integer field type mismatch")
            else:
                if field["type"] != DataType.FLOAT_VECTOR:
                    raise ValueError("dense_vector must be FLOAT_VECTOR")
                actual_dimension = int(field["params"]["dim"])
                if actual_dimension <= 0 or (
                    dimension is not None and actual_dimension != dimension
                ):
                    raise ValueError("dense dimension mismatch")
        return actual_dimension

    def _prepare_existing(self, dimension: int | None, *, create_index: bool) -> None:
        description = self._client.describe_collection(
            collection_name=self._collection, timeout=MILVUS_TIMEOUT_SECONDS
        )
        actual_dimension = self._validate_schema(description, dimension)
        indexes = self._client.list_indexes(
            collection_name=self._collection,
            field_name=_VECTOR_FIELD,
            timeout=MILVUS_TIMEOUT_SECONDS,
        )
        if not indexes and create_index:
            params = self._client.prepare_index_params()
            params.add_index(
                field_name=_VECTOR_FIELD,
                index_name=_INDEX_NAME,
                index_type="FLAT",
                metric_type="COSINE",
            )
            self._client.create_index(
                collection_name=self._collection,
                index_params=params,
                timeout=MILVUS_TIMEOUT_SECONDS,
            )
            indexes = [_INDEX_NAME]
        if len(indexes) != 1:
            raise ValueError("knowledge dense index is missing or ambiguous")
        index = self._client.describe_index(
            collection_name=self._collection,
            index_name=indexes[0],
            timeout=MILVUS_TIMEOUT_SECONDS,
        )
        if (
            index["field_name"] != _VECTOR_FIELD
            or index["index_type"] != "FLAT"
            or index["metric_type"] != "COSINE"
        ):
            raise ValueError("knowledge index must use FLAT/COSINE on dense_vector")
        self._client.load_collection(
            collection_name=self._collection, timeout=MILVUS_TIMEOUT_SECONDS
        )
        self._dimension = actual_dimension

    def _existing_dimension(self) -> int:
        if self._dimension is None:
            try:
                self._prepare_existing(None, create_index=False)
            except Exception as exc:
                raise VectorStoreError(
                    "Milvus knowledge collection is missing or incompatible."
                ) from exc
        return self._dimension

    def _vectors(
        self, vectors: Sequence[Sequence[float]], count: int
    ) -> list[list[float]]:
        try:
            values = np.asarray(vectors)
            if (
                values.dtype.kind not in "fiu"
                or values.ndim != 2
                or values.shape[0] != count
            ):
                raise ValueError(
                    "vectors must be a matching batch of real numeric vectors"
                )
            with np.errstate(over="raise", invalid="raise"):
                values = values.astype(np.float32)
            if not np.isfinite(values).all() or not np.any(values != 0, axis=1).all():
                raise ValueError("vectors must be finite and nonzero")
        except (TypeError, ValueError, FloatingPointError) as exc:
            raise ValueError(
                "vectors must be a matching batch of finite real nonzero vectors"
            ) from exc
        if values.shape[1] != self._existing_dimension():
            raise ValueError("vector dimension does not match the collection schema")
        return values.tolist()

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match")
        if not chunks:
            return 0
        rows = []
        ids = set()
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                # Keep the repository's input validation contract uniform.
                raise ValueError("upsert requires Chunk objects")  # noqa: TRY004
            for name, limit in _VARCHAR_LIMITS.items():
                value = getattr(chunk, name)
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or len(value.encode("utf-8")) > limit
                ):
                    raise ValueError(
                        f"{name} must be nonblank and within its UTF-8 byte limit"
                    )
            if type(chunk.chunk_index) is not int or not 0 <= chunk.chunk_index < 2**63:
                raise ValueError("chunk_index must be a nonnegative INT64")
            if chunk.chunk_id in ids:
                raise ValueError("duplicate chunk_id in one upsert batch")
            ids.add(chunk.chunk_id)
            rows.append(chunk.model_dump())
        dense_vectors = self._vectors(vectors, len(chunks))
        for row, vector in zip(rows, dense_vectors, strict=True):
            row[_VECTOR_FIELD] = vector
        try:
            response = self._client.upsert(
                collection_name=self._collection,
                data=rows,
                timeout=MILVUS_TIMEOUT_SECONDS,
            )
            count = response["upsert_count"]
            if type(count) is not int or not 0 <= count <= len(chunks):
                raise ValueError("Milvus returned an invalid upsert count")
            return count
        except Exception as exc:
            raise VectorStoreError("Milvus knowledge upsert failed.") from exc

    def search(self, vector: Sequence[float], top_k: int) -> list[SearchHit]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        data = self._vectors([vector], 1)
        try:
            response = self._client.search(
                collection_name=self._collection,
                data=data,
                anns_field=_VECTOR_FIELD,
                limit=top_k,
                output_fields=[*_VARCHAR_LIMITS, "chunk_index"],
                search_params={"metric_type": "COSINE", "params": {}},
                consistency_level="Strong",
                timeout=MILVUS_TIMEOUT_SECONDS,
            )
            if (
                not isinstance(response, list)
                or len(response) != 1
                or not isinstance(response[0], list)
            ):
                raise ValueError("Milvus returned an invalid search result container")
            hits = []
            for raw in response[0]:
                entity = raw["entity"]
                primary_key = raw["chunk_id"] if "chunk_id" in raw else raw["id"]
                if (
                    ("id" in raw and raw["id"] != primary_key)
                    or entity.get("chunk_id", primary_key) != primary_key
                ):
                    raise ValueError("Milvus hit primary key mismatch")
                hits.append(
                    SearchHit(
                        chunk_id=primary_key,
                        document_id=entity["document_id"],
                        text=entity["text"],
                        source=entity["source"],
                        chunk_index=entity["chunk_index"],
                        score=raw["distance"],
                    )
                )
            return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
        except Exception as exc:
            raise VectorStoreError(
                "Milvus knowledge search or result mapping failed."
            ) from exc
