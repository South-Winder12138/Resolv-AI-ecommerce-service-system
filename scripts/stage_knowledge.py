"""Stage one configured Markdown/TXT document into authoritative SQL as pending."""
import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from resolv.core.config import KnowledgeDatabaseSettings
from resolv.db.engine import create_session_factory
from resolv.db.knowledge import KnowledgeStoreError, knowledge_engine
from resolv.knowledge import (
    DEFAULT_SOURCE_ROOT,
    get_knowledge_file_config,
    load_document,
    load_knowledge_file_configs,
)
from resolv.schemas.knowledge import KnowledgeIndexProfile
from resolv.services.knowledge import KnowledgeConflictError, KnowledgeIngestionService


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Markdown/TXT path inside source-root")
    parser.add_argument("--config", type=Path, required=True, help="TOML per-file metadata")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--model-revision", required=True, help="Pinned lowercase 40-character HF commit")
    parser.add_argument("--collection", default="resolv_knowledge_v2", help="Declared future index target")
    parser.add_argument("--passage-max-length", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        configs = load_knowledge_file_configs(args.config, source_root=args.source_root)
        document = load_document(args.path, source_root=args.source_root)
        file_config = get_knowledge_file_config(configs, document.source)
        profile = KnowledgeIndexProfile(
            model_revision=args.model_revision, collection_name=args.collection,
            passage_max_length=args.passage_max_length,
        )
        settings = KnowledgeDatabaseSettings()
        with knowledge_engine(settings) as engine:
            service = KnowledgeIngestionService(create_session_factory(engine))
            result = service.stage_document(
                document, file_config, profile, max_chars=args.max_chars, overlap=args.overlap
            )
    except ValidationError as exc:
        raise SystemExit(
            "Knowledge staging configuration invalid. Check file metadata, pinned "
            "model revision, collection and KNOWLEDGE_DATABASE_URL."
        ) from exc
    except KnowledgeConflictError as exc:
        raise SystemExit(str(exc)) from exc
    except KnowledgeStoreError as exc:
        raise SystemExit(
            "Knowledge staging failed. Check MySQL, explicit migrations and existing data; "
            "inspect state before retrying."
        ) from exc
    except (ValueError, OSError) as exc:
        raise SystemExit(
            "Knowledge staging input invalid. Check file paths/source-root, UTF-8 "
            "TOML mappings and chunk budgets."
        ) from exc
    print(json.dumps({
        **result.model_dump(), "index_profile": profile.model_dump(),
        "inference_performed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
