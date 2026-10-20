"""Caller-owned BGE-M3 dense inference; importing this module loads no model."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from resolv.core.config import EmbeddingSettings
from resolv.schemas.knowledge import BGE_M3_DIMENSION

NORMALIZATION_TOLERANCE = 1e-4


class EmbeddingError(RuntimeError):
    """Model loading, inference, output validation, or cleanup failed."""


def _load_model(settings: EmbeddingSettings) -> Any:
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(
        settings.model_name,
        devices=settings.device,
        use_fp16=settings.use_fp16,
        normalize_embeddings=True,
        batch_size=settings.batch_size,
        query_max_length=settings.query_max_length,
        passage_max_length=settings.passage_max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        query_instruction_for_retrieval=None,
        trust_remote_code=False,
    )


class EmbeddingService:
    """Create once at an explicit composition point and reuse synchronously."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings.model_copy(deep=True)
        self._model: Any = None
        try:
            self._model = _load_model(self._settings)
        except Exception as exc:
            raise EmbeddingError("Embedding model loading failed.") from exc

    @property
    def dimension(self) -> int:
        return BGE_M3_DIMENSION

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._require_model()
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            # Invalid text types use the same public validation boundary as blanks.
            raise ValueError("documents must be a sequence of nonblank strings")  # noqa: TRY004
        return self._embed(list(texts), query=False)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], query=True)[0]

    def _require_model(self) -> Any:
        if self._model is None:
            raise EmbeddingError("Embedding service is closed.")
        return self._model

    def _embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        model = self._require_model()
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"text at index {index} must be a nonblank string")
        if not texts:
            return []

        max_length = (
            self._settings.query_max_length
            if query
            else self._settings.passage_max_length
        )
        try:
            tokens = model.tokenizer(
                texts, truncation=False, padding=False, add_special_tokens=True
            )["input_ids"]
            if len(tokens) != len(texts):
                raise ValueError("tokenizer returned a different batch size")
            lengths = [len(ids) for ids in tokens]
        except Exception as exc:
            raise EmbeddingError("Embedding tokenization failed.") from exc
        for index, length in enumerate(lengths):
            if length > max_length:
                raise ValueError(
                    f"text at index {index} has {length} tokens; limit is {max_length}"
                )

        try:
            encode = model.encode_queries if query else model.encode_corpus
            result = encode(
                texts,
                batch_size=self._settings.batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            with np.errstate(over="raise", invalid="raise"):
                vectors = np.asarray(result["dense_vecs"], dtype=np.float32)
            if vectors.shape != (len(texts), self.dimension):
                raise ValueError("dense output has an invalid count or dimension")
            if not np.isfinite(vectors).all():
                raise ValueError("dense output has nonfinite values")
            norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
            if np.any(np.abs(norms - 1.0) > NORMALIZATION_TOLERANCE):
                raise ValueError("dense output is not normalized")
            return vectors.tolist()
        except Exception as exc:
            raise EmbeddingError(
                "Dense embedding inference or output validation failed."
            ) from exc

    def close(self) -> None:
        model, self._model = self._model, None
        if model is not None:
            try:
                model.stop_self_pool()
            except Exception as exc:
                raise EmbeddingError("Embedding model cleanup failed.") from exc
