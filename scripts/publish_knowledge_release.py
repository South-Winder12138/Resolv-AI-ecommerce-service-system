"""Publish only an explicit immutable formal manifest and new index target."""

import argparse
from pathlib import Path

from resolv.services.knowledge_release import publish_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "report-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-database", "version", "collection"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = publish_release(
            args.manifest,
            expected_database=args.expected_database,
            version=args.version,
            collection=args.collection,
            report_dir=args.report_dir,
        )
    except Exception as exc:  # noqa: BLE001 -- CLI errors must not expose credentials
        raise SystemExit(
            "Release publication failed: " + getattr(exc, "code", type(exc).__name__)
        ) from None
    print(receipt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
