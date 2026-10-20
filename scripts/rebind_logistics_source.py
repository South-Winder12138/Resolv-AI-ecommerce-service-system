"""Explicitly move existing task resources to another verified code export."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from scripts.logistics_gate_support import (
    SOURCE_ROOT,
    rebind_logistics_source,
    source_revision,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--previous-revision", required=True)
    parser.add_argument("--previous-source-root", required=True)
    parser.add_argument("--allow-terminal-budget", action="store_true", help="Code-only handoff; never reopens the terminal ledger")
    args = parser.parse_args()
    report = json.loads((SOURCE_ROOT.parents[1] / "report.json").read_bytes())
    if report.get("passed") is not True or report.get("candidate_commit") != source_revision() or report.get("exported_source") != str(SOURCE_ROOT):
        raise ValueError("Source handoff requires a passed exact offline export")
    rebind_logistics_source(args.ownership, previous_revision=args.previous_revision, previous_source_root=args.previous_source_root, allow_terminal_budget=args.allow_terminal_budget)
    print("Source rebound; existing task data and unused model budget preserved")


if __name__ == "__main__":
    main()
