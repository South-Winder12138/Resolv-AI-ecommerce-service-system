"""Explicit download preparation for a pinned reranker; normal inference is local."""

import argparse
from pathlib import Path

from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.services.hybrid_reranker import (
    MODEL_NAME,
    _snapshot_download,
    resolve_reranker_snapshot,
    validate_revision,
)


def prepare(revision: str, cache_dir: Path | None = None) -> Path:
    validate_revision(revision)
    try:
        return resolve_reranker_snapshot(revision, cache_dir)
    except HybridRetrievalError:
        _snapshot_download(
            repo_id=MODEL_NAME,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        return resolve_reranker_snapshot(revision, cache_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        path = prepare(args.revision, args.cache_dir)
    except Exception as exc:  # noqa: BLE001 -- suppress arbitrary external credential text
        raise SystemExit(f"Reranker preparation failed: {type(exc).__name__}") from None
    print(f"Prepared {MODEL_NAME} at revision {args.revision}: {path}")


if __name__ == "__main__":
    main()
