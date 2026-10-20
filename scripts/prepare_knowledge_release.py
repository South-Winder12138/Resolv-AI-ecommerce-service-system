"""Prepare the six approved sources from an explicit committed snapshot."""

import argparse
from pathlib import Path

from resolv.services.knowledge_release import prepare_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare_release(args.revision, output_dir=args.output_dir)
    except Exception as exc:  # noqa: BLE001 -- CLI errors must not expose credentials
        raise SystemExit("Release preparation failed: " + type(exc).__name__) from None
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
