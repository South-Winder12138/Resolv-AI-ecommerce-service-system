"""Explicit real-provider transport probe. Never executes a business tool."""

import argparse
import json
from pathlib import Path
from time import monotonic

from langchain_core.messages import HumanMessage, SystemMessage

from resolv.core.business_model import (
    get_business_model,
    invoke_business_model,
    make_business_config,
)
from resolv.core.config import Settings
from resolv.schemas.conversation import ConversationError
from resolv.services.context import TokenCounter
from resolv.services.routed_runtime import close_model


def run_probe(client, config, counter, *, question, require_tool=False):
    started = monotonic()
    audits = []
    decision = invoke_business_model(
        client,
        [SystemMessage(content=config.system_prompt), HumanMessage(content=question)],
        config=config,
        counter=counter,
        model_step=1,
        attempt=1,
        audit_hook=audits.append,
        require_tool=require_tool,
    )
    return {
        **decision.audit.model_dump(mode="json"),
        "elapsed_seconds": monotonic() - started,
        "text": decision.message.text,
        "tool_calls": decision.message.tool_calls,
        "business_executions": 0,
        "accuracy_evidence": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "passed": False,
        "accuracy_evidence": False,
        "business_executions": 0,
        "probes": [],
    }
    client = None
    try:
        settings = Settings()
        config = make_business_config(settings)
        report["profile"] = config.model_dump(mode="json")
        client = get_business_model(settings, config)
        counter = TokenCounter(config.tokenizer_profile)
        for index, question in enumerate(
            [
                "你好，请只回答您好。",
                "请查询订单 MH_TRANSPORT_001 的真实状态，必须先调用query_order，不执行取消。",
            ]
        ):
            report["probes"].append(
                run_probe(
                    client, config, counter, question=question, require_tool=index == 1
                )
            )
        calls = report["probes"][1]["tool_calls"]
        if (
            not report["probes"][0]["text"]
            or len(calls) != 1
            or calls[0]["name"] != "query_order"
            or calls[0]["args"] != {"order_id": "MH_TRANSPORT_001"}
        ):
            raise ConversationError("transport_probe_invalid")
        report["passed"] = True
    except Exception as caught:  # noqa: BLE001 — explicit safe transport failure, no protocol fallback
        report["error_code"] = (
            caught.code
            if isinstance(caught, ConversationError)
            else "transport_unavailable"
        )
    finally:
        if client is not None:
            try:
                close_model(client)
            except ConversationError:
                report["passed"], report["error_code"] = False, "model_close_failed"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "probes": len(report["probes"]),
                "error_code": report.get("error_code"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
