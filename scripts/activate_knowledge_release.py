"""Explicit activation after all acceptance evidence, or explicit first-release disable."""

import argparse
from pathlib import Path

from resolv.services.knowledge_release import activate_release, disable_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--expected-previous-sha256", required=True)
    parser.add_argument("--disable", action="store_true")
    args = parser.parse_args(argv)
    if args.disable and (args.manifest or args.acceptance):
        parser.error("disable cannot be combined with activation inputs")
    if not args.disable and (not args.manifest or not args.acceptance):
        parser.error("activation requires manifest and acceptance")
    try:
        if args.disable:
            result = disable_release(
                selector_path=args.selector,
                expected_previous_hash=args.expected_previous_sha256,
            )
        else:
            result = activate_release(
                args.manifest,
                args.acceptance,
                selector_path=args.selector,
                expected_previous_hash=args.expected_previous_sha256,
            )
    except Exception as exc:  # noqa: BLE001 -- CLI errors must not expose credentials
        raise SystemExit(
            "Release selection failed: " + getattr(exc, "code", type(exc).__name__)
        ) from None
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
