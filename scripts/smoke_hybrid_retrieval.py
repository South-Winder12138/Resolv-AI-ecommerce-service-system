"""Retrieve explicitly from a selected ready version; output evidence after cleanup."""

import argparse
import json
from pathlib import Path

from resolv.core.config import KnowledgeDatabaseSettings
from resolv.db.knowledge import knowledge_engine
from resolv.db.knowledge_hybrid import consistent_read_factory
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from resolv.schemas.hybrid_knowledge import (
    HybridRetrievalError,
    HybridRetrievalProfile,
)
from resolv.services.hybrid_runtime import hybrid_retrieval_runtime


def run(args):
    with (
        knowledge_engine(KnowledgeDatabaseSettings()) as engine,
        consistent_read_factory(engine)() as session,
    ):
        snapshot = HybridIndexRepository(session).get_version(args.version)
        if (
            snapshot.state != "ready"
            or snapshot.profile.collection_name != args.collection
        ):
            raise HybridRetrievalError("hybrid_cli_version_mismatch")
    with hybrid_retrieval_runtime(
        version_id=args.version,
        index_profile=snapshot.profile,
        retrieval_profile=HybridRetrievalProfile(
            reranker_revision=args.reranker_revision
        ),
        cache_dir=args.cache_dir,
    ) as service:
        result = service.retrieve(args.query)
    return result.model_dump(mode="json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("version", "collection", "reranker-revision", "query"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.report_dir.exists():
        raise SystemExit("Hybrid retrieval failed: hybrid_report_dir_exists")
    try:
        report = run(args)
        args.report_dir.mkdir(parents=True)
        (args.report_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        print(json.dumps(report, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 -- query and arbitrary dependency credentials never go into error output
        args.report_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "passed": False,
            "code": getattr(exc, "code", "hybrid_cli_failed"),
            "type": type(exc).__name__,
        }
        (args.report_dir / "failure.json").write_text(json.dumps(failure, indent=2))
        raise SystemExit("Hybrid retrieval failed: " + failure["code"]) from None


if __name__ == "__main__":
    main()
