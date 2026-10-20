"""Bind the existing dense embedder to a locally cached immutable HF snapshot."""
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from resolv.core.config import EmbeddingSettings
from resolv.schemas.knowledge import KnowledgeIndexProfile
from resolv.services.embedding import EmbeddingError, EmbeddingService


def _snapshot_download(**kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _resolve_snapshot(profile: KnowledgeIndexProfile, cache_dir: Path | None) -> Path:
    try:
        directory = Path(_snapshot_download(
            repo_id=profile.model_name, revision=profile.model_revision,
            cache_dir=cache_dir, local_files_only=True,
        )).resolve()
        if (
            directory.name != profile.model_revision or directory.parent.name != "snapshots"
            or directory.parent.parent.name != "models--" + profile.model_name.replace("/", "--")
        ):
            raise ValueError("Cached snapshot identity does not match its declared profile")
        required = (
            "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "sentencepiece.bpe.model", "pytorch_model.bin", "colbert_linear.pt", "sparse_linear.pt",
        )
        if not all((directory / name).is_file() for name in required):
            raise ValueError("Cached snapshot is incomplete")
        return directory
    except Exception as exc:
        raise EmbeddingError("Pinned BGE-M3 snapshot is unavailable or incomplete locally.") from exc


class KnowledgeEmbeddingService:
    def __init__(
        self, profile: KnowledgeIndexProfile, *, batch_size: int = 8,
        query_max_length: int = 512, cache_dir: Path | None = None,
    ) -> None:
        self._profile = KnowledgeIndexProfile.model_validate(profile.model_dump())
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(query_max_length) is not int or not 0 < query_max_length <= 8192:
            raise ValueError("query_max_length must be a supported positive integer")
        directory = _resolve_snapshot(self._profile, cache_dir)
        settings = EmbeddingSettings(
            _env_file=None, BGE_M3_MODEL=str(directory),
            device="cpu", use_fp16=False, batch_size=batch_size,
            query_max_length=query_max_length, passage_max_length=self._profile.passage_max_length,
        )
        self._service = EmbeddingService(settings)

    @property
    def profile(self) -> KnowledgeIndexProfile:
        return self._profile

    @property
    def dimension(self) -> int:
        return self._service.dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._service.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._service.embed_query(text)

    def close(self) -> None:
        self._service.close()


@contextmanager
def pinned_knowledge_embedding(
    profile: KnowledgeIndexProfile, *, batch_size: int = 8,
    query_max_length: int = 512, cache_dir: Path | None = None,
) -> Iterator[KnowledgeEmbeddingService]:
    service = KnowledgeEmbeddingService(
        profile, batch_size=batch_size, query_max_length=query_max_length, cache_dir=cache_dir
    )
    primary_error: BaseException | None = None
    try:
        yield service
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            service.close()
        except Exception as exc:
            if primary_error is None:
                raise EmbeddingError("Pinned knowledge embedding cleanup failed.") from exc
            primary_error.add_note("Pinned knowledge embedding cleanup also failed.")
