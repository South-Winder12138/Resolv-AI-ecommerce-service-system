"""Preview structured Markdown/TXT chunks; no models or databases are opened."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from resolv.knowledge import (
    DEFAULT_SOURCE_ROOT,
    chunk_knowledge_document,
    get_knowledge_file_config,
    load_document,
    load_knowledge_file_configs,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Markdown/TXT path inside source-root")
    parser.add_argument("--config", type=Path, required=True, help="TOML per-file metadata")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        configs = load_knowledge_file_configs(args.config, source_root=args.source_root)
        document = load_document(args.path, source_root=args.source_root)
        file_config = get_knowledge_file_config(configs, document.source)
        chunks = chunk_knowledge_document(
            document, file_config, max_chars=args.max_chars, overlap=args.overlap
        )
    except (ValueError, OSError) as exc:
        raise SystemExit(
            "Knowledge preview failed. Check file paths/source-root, UTF-8 TOML "
            "file mapping, content_type, keywords and chunk budgets."
        ) from exc
    print(json.dumps({
        "document_id": document.document_id,
        "source": document.source,
        "content_type": file_config.content_type,
        "max_chars": args.max_chars,
        "overlap": args.overlap,
        "chunk_count": len(chunks),
        "chunks": [chunk.model_dump() for chunk in chunks],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
