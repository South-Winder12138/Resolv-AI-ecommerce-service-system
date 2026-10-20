"""Pinned local encoder logits with explicit lengths and one forward per batch."""

import gc
import json
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from threading import RLock

from resolv.schemas.hybrid_knowledge import (
    HybridRetrievalError,
    HybridRetrievalProfile,
    PreparedWindow,
)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "model.safetensors",
)
INFERENCE_LOCK = RLock()


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _dependencies():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return torch, AutoTokenizer, AutoModelForSequenceClassification


def validate_revision(revision):
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise HybridRetrievalError("reranker_revision_invalid")


def resolve_reranker_snapshot(revision: str, cache_dir: Path | None = None) -> Path:
    validate_revision(revision)
    try:
        path = Path(
            _snapshot_download(
                repo_id=MODEL_NAME,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        ).resolve()
        if (
            path.name != revision
            or path.parent.name != "snapshots"
            or path.parent.parent.name != "models--BAAI--bge-reranker-v2-m3"
        ):
            raise ValueError("Snapshot identity mismatch")
        if not all(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in REQUIRED_FILES
        ):
            raise ValueError("Incomplete snapshot")
        config = json.loads((path / "config.json").read_text())
        if (
            config.get("model_type") != "xlm-roberta"
            or config.get("max_position_embeddings", 0)
            - config.get("pad_token_id", 1)
            - 1
            < 3072
        ):
            raise ValueError("Unsupported model capacity")
        return path
    except Exception as exc:
        raise HybridRetrievalError("reranker_snapshot_unavailable") from exc


class LocalHybridReranker:
    def __init__(
        self, profile: HybridRetrievalProfile, *, cache_dir: Path | None = None
    ):
        self.profile = HybridRetrievalProfile.model_validate(profile.model_dump())
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._closed = False
        self.query_tokens = 0
        self.pair_tokens = ()
        self.snapshot = resolve_reranker_snapshot(
            self.profile.reranker_revision, cache_dir
        )
        try:
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
                raise HybridRetrievalError("reranker_cpu_fallback_enabled")
            torch, tokenizers, models = _dependencies()
            self._torch = torch
            if not torch.backends.mps.is_available():
                raise HybridRetrievalError("reranker_mps_unavailable")
            self._tokenizer = tokenizers.from_pretrained(
                str(self.snapshot), local_files_only=True
            )
            if self._tokenizer.model_max_length < self.profile.pair_max_tokens:
                raise HybridRetrievalError("reranker_tokenizer_capacity")
            self._model = models.from_pretrained(
                str(self.snapshot), local_files_only=True, dtype=torch.float32
            )
            config = self._model.config
            if (
                config.num_labels != 1
                or config.max_position_embeddings - config.pad_token_id - 1
                < self.profile.pair_max_tokens
            ):
                raise HybridRetrievalError("reranker_model_capacity")
            self._model.to(self.profile.device, dtype=torch.float32)
            self._model.eval()
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_error:  # noqa: BLE001 -- preserve primary initialization failure
                exc.add_note(
                    f"Reranker initialization cleanup also failed ({type(cleanup_error).__name__})."
                )
            if isinstance(exc, HybridRetrievalError):
                raise
            raise HybridRetrievalError("reranker_load_failed") from exc

    def count_query_tokens(self, query: str) -> int:
        with INFERENCE_LOCK:
            if self._closed:
                raise HybridRetrievalError("reranker_closed")
            if not isinstance(query, str) or not query.strip():
                raise HybridRetrievalError("invalid_query")
            try:
                count = len(
                    self._tokenizer(query, add_special_tokens=True, truncation=False)[
                        "input_ids"
                    ]
                )
            except Exception as exc:
                raise HybridRetrievalError("reranker_token_count_failed") from exc
            if count > self.profile.query_max_tokens:
                raise HybridRetrievalError("query_too_long")
            return count

    def score(self, query: str, windows: Sequence[PreparedWindow]) -> tuple[float, ...]:
        with INFERENCE_LOCK:
            if self._closed:
                raise HybridRetrievalError("reranker_closed")
            if not isinstance(query, str) or not query.strip():
                raise HybridRetrievalError("invalid_query")
            if len(windows) > 50:
                raise HybridRetrievalError("reranker_window_count")
            self.query_tokens = 0
            self.pair_tokens = ()
            try:
                q = self._tokenizer(query, add_special_tokens=True, truncation=False)
                self.query_tokens = len(q["input_ids"])
                if self.query_tokens > self.profile.query_max_tokens:
                    raise HybridRetrievalError("query_too_long")
                rows = []
                for w in windows:
                    w = PreparedWindow.model_validate(w.model_dump())
                    row = self._tokenizer(
                        query, w.body_text, add_special_tokens=True, truncation=False
                    )
                    if len(row["input_ids"]) > self.profile.pair_max_tokens:
                        raise HybridRetrievalError("pair_too_long")
                    rows.append(row)
                self.pair_tokens = tuple(len(row["input_ids"]) for row in rows)
                scores = []
                with self._torch.inference_mode():
                    for start in range(0, len(rows), self.profile.batch_size):
                        batch = rows[start : start + self.profile.batch_size]
                        padded = self._tokenizer.pad(
                            batch, padding=True, return_tensors="pt"
                        )
                        inputs = {
                            key: value.to(self.profile.device)
                            for key, value in padded.items()
                        }
                        logits = self._model(**inputs).logits
                        if tuple(logits.shape) != (len(batch), 1):
                            raise HybridRetrievalError("reranker_score_count")
                        values = logits.reshape(-1).float().cpu().tolist()
                        if len(values) != len(batch) or any(
                            not math.isfinite(x) for x in values
                        ):
                            raise HybridRetrievalError("reranker_invalid_score")
                        scores.extend(values)
                return tuple(scores)
            except Exception as exc:
                if isinstance(exc, HybridRetrievalError):
                    raise
                raise HybridRetrievalError("reranker_inference_failed") from exc

    def close(self) -> None:
        with INFERENCE_LOCK:
            if self._closed:
                return
            self._closed = True
            owned_model = self._model is not None
            self._model = None
            self._tokenizer = None
            gc.collect()
            if owned_model and self._torch is not None:
                try:
                    self._torch.mps.empty_cache()
                except Exception as exc:
                    raise HybridRetrievalError("reranker_close_failed") from exc
