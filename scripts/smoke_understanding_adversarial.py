"""Explicit six-case RF2 real-model smoke; separate from the immutable 140 gold."""

import argparse
import asyncio
import json
import os
import re
from hashlib import sha256
from pathlib import Path

from scripts.smoke_understanding_accuracy import evaluate_one
from sqlalchemy.engine import make_url

from resolv.core.config import (
    ContextSettings,
    ConversationDatabaseSettings,
    Settings,
    UnderstandingSettings,
)
from resolv.core.understanding_model import (
    get_understanding_model,
    understanding_model_profile,
)
from resolv.core.understanding_prompts import (
    INTENT_PROMPT_V1,
    INTENT_VERSION,
    RESOLUTION_PROMPT_V1,
    RESOLUTION_VERSION,
)
from resolv.db.conversation import conversation_engine
from resolv.db.engine import create_session_factory
from resolv.repositories.conversation import fingerprint
from resolv.schemas.understanding import UnderstandingError
from resolv.services.context import TokenCounter
from resolv.services.understanding_context import UnderstandingContext
from resolv.services.understanding_evaluation import load_cases, score_cases

ROOT = Path(__file__).resolve().parents[1]


def isolated_database(value):
    url = make_url(value)
    if url.get_backend_name() != "mysql" or not re.fullmatch(
        r"resolv_conversations_smoke_[A-Za-z0-9_]+", url.database or ""
    ):
        raise UnderstandingError("isolated_mysql_required")
    return ConversationDatabaseSettings(_env_file=None, conversation_database_url=value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-env", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    database = isolated_database(os.environ.get(args.database_env, ""))
    fixture = ROOT / "tests/fixtures/understanding_adversarial.json"
    cases = load_cases(fixture)
    settings, budgets, options = Settings(), ContextSettings(), UnderstandingSettings()
    counter = TokenCounter(budgets.tokenizer_profile)
    context = UnderstandingContext(
        budgets,
        counter,
        model_name=settings.openai_model,
        api_address_fingerprint=fingerprint(settings.openai_base_url),
        resolution_prompt=RESOLUTION_PROMPT_V1,
        resolution_version=RESOLUTION_VERSION,
        intent_prompt=INTENT_PROMPT_V1,
        intent_version=INTENT_VERSION,
        overhead_override=options.provider_input_overhead_tokens,
        **understanding_model_profile(settings, options.reasoning_effort),
    )
    client = get_understanding_model(settings, options.timeout_seconds)
    results = []
    report = {
        "scope": "supplemental RF2, not the fixed 140 accuracy gate",
        "dataset_sha256": sha256(fixture.read_bytes()).hexdigest(),
        "model": settings.openai_model,
        "config": context.config.model_dump(mode="json"),
        "business_tool_calls": 0,
        "results": results,
    }
    try:
        with conversation_engine(database) as engine:
            factory = create_session_factory(engine)
            for case in cases:
                result = evaluate_one(case, factory, database, context, client, counter)
                results.append(result)
                print(
                    json.dumps(
                        {
                            "case": case.case_id,
                            "status": result["status"],
                            "error": result["error_code"],
                        }
                    ),
                    flush=True,
                )
            scores = score_cases(cases, results)
            report["scores"] = scores.model_dump(mode="json")
            passed = not (
                scores.failures
                or scores.wrong_binding_count
                or scores.constraint_violations
                or scores.token_underestimates
            )
            report["result"] = "PASS" if passed else "FAIL"
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            if not passed:
                raise UnderstandingError("adversarial_acceptance_failed")
    finally:
        client.root_client.close()
        asyncio.run(client.root_async_client.close())


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 — never print credentials/provider error text
        raise SystemExit(
            "Supplemental RF2 smoke FAILED; inspect the local report."
        ) from None
