"""Explicit task-owned DDL entry point; never called by an app lifespan."""

import argparse
from pathlib import Path

from scripts.logistics_gate_support import prepare_logistics_demo


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--workspace", required=True, type=Path)
    value.add_argument("--database", required=True)
    return value


def main():
    args = parser().parse_args()
    value = prepare_logistics_demo(args.workspace, args.database)
    print("Prepared logistics resources: " + value["resource_root"])


if __name__ == "__main__":
    main()
