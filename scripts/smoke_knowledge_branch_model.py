"""Explicit real-provider Gate K1; ordinary pytest never calls run_calls."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import monotonic

from langchain_core.messages import HumanMessage, SystemMessage

from resolv.core.config import ContextSettings, Settings, UnderstandingSettings
from resolv.core.knowledge_branch_model import (
    KnowledgeModelClient,
    make_knowledge_model_profile,
)
from resolv.core.knowledge_branch_prompts import KnowledgeQueryOutput
from resolv.schemas.knowledge_branch import SufficiencyResult, canonical_json

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / ".superpowers/sdd/2026-10-15-resolv-knowledge-branch"
SOURCES = (
    "src/resolv/core/knowledge_branch_model.py",
    "src/resolv/core/knowledge_branch_prompts.py",
    "src/resolv/schemas/knowledge_branch.py",
    "scripts/smoke_knowledge_branch_model.py",
    "pyproject.toml",
    "uv.lock",
)


def validate_model_gate(report):
    if (
        report.get("business_tool_calls") != 0
        or report.get("passed") is not True
        or not report.get("source_fingerprints")
    ):
        raise ValueError("Incomplete K1 gate")
    calls = report.get("calls", [])
    if len(calls) != 3 or {c.get("purpose") for c in calls} != {
        "knowledge_query",
        "knowledge_check",
        "knowledge_answer",
    }:
        raise ValueError("Missing K1 purpose")
    for call in calls:
        audit = call.get("audit", {})
        if (
            audit.get("outcome") != "succeeded"
            or not call.get("structure_valid")
            or call.get("tool_calls") != 0
            or type(audit.get("input_tokens")) is not int
            or type(audit.get("output_tokens")) is not int
            or audit["input_tokens"] > audit.get("input_estimate", -1)
            or audit["input_estimate"] > call.get("input_limit", -1)
            or audit["output_tokens"] > call.get("output_limit", -1)
        ):
            raise ValueError("Missing K1 usage/budget/structure evidence")


def run_calls(*, audit_hook=None):
    settings = Settings()
    ref = ContextSettings().model_dump()
    overhead = UnderstandingSettings().provider_input_overhead_tokens
    ref["provider_input_overhead_tokens"] = (
        overhead if overhead is not None else ref["provider_input_overhead_tokens"]
    )
    # Accepted customer-service transport is Chat Completions with thinking disabled.
    answer_ref = ref | {
        "model_api_mode": "chat_completions",
        "enable_thinking": False,
        "top_p": 0.1,
    }
    client = KnowledgeModelClient(settings)
    calls, audits = [], []

    def record(audit):
        audits.append(audit)
        if audit_hook is not None:
            audit_hook(audit)

    primary = "普通档用户未满99元的配送基础运费是多少？"
    messages = {
        "knowledge_query": canonical_json({"primary": primary, "bindings": []}),
        "knowledge_check": canonical_json(
            {
                "primary": primary,
                "standard": primary,
                "evidence_hash": "b" * 64,
                "windows": [
                    {"window_id": "a" * 64, "text": "普通档用户未满99元基础运费10元。"}
                ],
            }
        ),
        "knowledge_answer": canonical_json(
            {
                "primary": primary,
                "citations": [
                    {
                        "citation_number": 1,
                        "window_id": "a" * 64,
                        "text": "普通档用户未满99元基础运费10元。",
                    }
                ],
            }
        ),
    }
    start = monotonic()
    for purpose, body in messages.items():
        p = make_knowledge_model_profile(
            settings,
            purpose=purpose,
            reference_profile=answer_ref if purpose == "knowledge_answer" else ref,
        )
        sent = [SystemMessage(content=p.system_prompt), HumanMessage(content=body)]
        if purpose == "knowledge_answer":
            output = client.invoke_answer(sent, profile=p, audit_hook=record)
            valid = "[1]" in output.content
            result = {"text": output.content}
        else:
            schema = (
                KnowledgeQueryOutput
                if purpose == "knowledge_query"
                else SufficiencyResult
            )
            output = client.invoke_json(
                purpose, sent, schema, profile=p, audit_hook=record
            )
            valid = purpose == "knowledge_query" or (
                output.useful and output.supporting_window_ids == ("a" * 64,)
            )
            result = output.model_dump(mode="json")
        calls.append(
            {
                "purpose": purpose,
                "model_name": p.model_name,
                "protocol": p.model_api_mode,
                "profile_hash": p.fingerprint,
                "tokenizer": p.tokenizer_profile,
                "input_limit": p.input_limit,
                "output_limit": p.reserved_output_tokens,
                "tool_calls": 0,
                "structure_valid": valid,
                "audit": audits[-1].model_dump(mode="json"),
                "output": result,
            }
        )
    report = {
        "gate": "K1",
        "passed": True,
        "business_tool_calls": 0,
        "calls": calls,
        "audits": [a.model_dump(mode="json") for a in audits],
        "elapsed_seconds": monotonic() - start,
        "source_fingerprints": {
            name: sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
        },
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    target = args.report_dir.resolve()
    if (
        target.exists()
        or not target.is_relative_to(WORKSPACE.resolve())
        or target == WORKSPACE.resolve()
    ):
        print("report_directory_rejected")
        return 2
    target.mkdir(parents=True, exist_ok=False)
    audits = []
    report = None

    def record(audit):
        audits.append(audit.model_dump(mode="json"))
        with (target / "audits.jsonl").open("a") as stream:
            stream.write(json.dumps(audits[-1], ensure_ascii=False) + "\n")

    try:
        report = run_calls(audit_hook=record)
        validate_model_gate(report)
    except Exception as exc:  # noqa: BLE001 -- credentials and reasoning never enter reports
        failure = {
            "gate": "K1",
            "passed": False,
            "error_code": getattr(exc, "code", "knowledge_gate_failed"),
            "error_type": type(exc).__name__,
            "business_tool_calls": 0,
            "audits": audits,
            "partial_report": report,
        }
        (target / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2)
        )
        print(
            json.dumps(
                {k: failure[k] for k in ("gate", "passed", "error_code", "error_type")}
            )
        )
        return 1
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(target / "report.json"),
                "calls": len(report["calls"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
