"""Fixed real-model acceptance via JSON HTTP and isolated real MySQL context."""

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url

from resolv.api.understandings import router
from resolv.core.config import (
    ContextSettings,
    ConversationDatabaseSettings,
    Settings,
    UnderstandingSettings,
)
from resolv.core.understanding_model import (
    JsonStageModel,
    get_understanding_model,
    understanding_model_profile,
)
from resolv.core.understanding_prompts import (
    INTENT_PROMPT_V1,
    INTENT_VERSION,
    RESOLUTION_PROMPT_V1,
    RESOLUTION_VERSION,
)
from resolv.db.checkpoints import checkpoint_session
from resolv.db.conversation import conversation_engine
from resolv.db.engine import create_session_factory
from resolv.models.conversation import (
    Conversation,
    ConversationAction,
    ConversationMessage,
    ConversationRun,
    ConversationSummaryRecord,
)
from resolv.repositories.conversation import ConversationRepository, fingerprint
from resolv.repositories.understanding import UnderstandingRepository
from resolv.schemas.understanding import (
    ClassificationResult,
    ResolutionResult,
    StageAudit,
    UnderstandingError,
    UnderstandingInput,
)
from resolv.services.context import TokenCounter
from resolv.services.intent_classification import IntentClassifier
from resolv.services.reference_resolution import ReferenceResolver, active_business_ids
from resolv.services.understanding import UnderstandingService
from resolv.services.understanding_context import UnderstandingContext
from resolv.services.understanding_evaluation import (
    assert_acceptance,
    load_cases,
    score_cases,
)
from resolv.services.understanding_safety import CLAIMS, validate_request_semantics

ROOT = Path(__file__).resolve().parents[1]


class ObservedBoundModel:
    """Record transport shape only; never copy provider error text or reasoning."""

    def __init__(self, bound, diagnostics):
        self.bound, self.diagnostics = bound, diagnostics

    def invoke(self, messages):
        try:
            reply = self.bound.invoke(messages)
        except Exception as error:
            self.diagnostics.append(
                {
                    "exception_type": type(error).__name__,
                    "empty_stream": isinstance(error, ValueError)
                    and str(error) == "No generations found in stream.",
                }
            )
            raise
        self.diagnostics.append(
            {
                "status": reply.response_metadata.get("status"),
                "usage_present": bool(reply.usage_metadata),
                "block_types": [
                    b.get("type") for b in reply.content if isinstance(b, dict)
                ]
                if isinstance(reply.content, list)
                else ["string"],
            }
        )
        return reply


class ObservedModel:
    def __init__(self, client, diagnostics):
        self.client, self.diagnostics = client, diagnostics

    def bind(self, **parameters):
        return ObservedBoundModel(self.client.bind(**parameters), self.diagnostics)


class AuditedStageModel(JsonStageModel):
    def __init__(self, client, counter=None):
        self.sdk_diagnostics = []
        super().__init__(ObservedModel(client, self.sdk_diagnostics), counter)
        self.parsed_outputs = {}

    def invoke(self, *args, **kwargs):
        output, audit = super().invoke(*args, **kwargs)
        self.parsed_outputs[audit.invocation_id] = output.model_dump(mode="json")
        return output, audit


def evaluate_one(case, factory, database, context, client, counter):
    model = AuditedStageModel(client, counter)
    service = UnderstandingService(
        factory,
        partial(checkpoint_session, database),
        context,
        ReferenceResolver(model, context),
        IntentClassifier(model, context),
    )
    c = seed_history(factory, case.history, case.summary)
    if case.foreign_history:
        seed_history(factory, case.foreign_history)
    task_keys = seed_pending(service, c, case.pending)
    app = FastAPI()
    app.state.understanding_runtime = service
    app.include_router(router)
    with TestClient(app) as http:
        started = time.monotonic()
        response = http.post(
            f"/conversations/{c}/understandings",
            json={"request_id": uuid4().hex, "text": case.text},
        )
        elapsed = time.monotonic() - started
        value = response.json()
        if response.status_code != 200:
            error = value.get("detail", {})
            id = error.get("understanding_id")
            value = (
                service.get(c, id).model_dump(mode="json")
                if id
                else {"status": "failed", "error_code": error.get("code")}
            )
        record = (
            service.record(c, value["understanding_id"])
            if value.get("understanding_id")
            else None
        )
        result = prediction(case, value, record, task_keys, elapsed)
        result["parsed_outputs"] = model.parsed_outputs
        result["sdk_diagnostics"] = model.sdk_diagnostics
        raw_resolution = next(
            (v for v in model.parsed_outputs.values() if "resolved_text" in v), None
        )
        if raw_resolution is not None:
            result["binding_values"] = [b["value"] for b in raw_resolution["bindings"]]
        return result


def seed_history(
    factory, history, summary=None, title="understanding-accuracy-v1", summary_meta=None
):
    from resolv.db.conversation_base import utc_now

    def timestamp(value):
        return datetime.fromisoformat(value) if value else utc_now()

    c = uuid4().hex
    with factory.begin() as session:
        session.add(
            Conversation(
                conversation_id=c,
                thread_id=c,
                title=title,
                next_message_seq=2 * len(history) + 1,
            )
        )
        session.flush()
        runs = []
        for i, turn in enumerate(history):
            run, q, a = uuid4().hex, uuid4().hex, uuid4().hex
            runs.append(run)
            session.add(
                ConversationRun(
                    run_id=run,
                    conversation_id=c,
                    request_id=uuid4().hex,
                    request_fingerprint="f" * 64,
                    status="completed",
                    question_message_id=q,
                    answer_message_id=a,
                )
            )
            session.flush()
            session.add_all(
                [
                    ConversationMessage(
                        message_id=q,
                        conversation_id=c,
                        run_id=run,
                        seq=2 * i + 1,
                        role="user",
                        text=turn["question"],
                        created_at=timestamp(turn.get("user_observed_at")),
                    ),
                    ConversationMessage(
                        message_id=a,
                        conversation_id=c,
                        run_id=run,
                        seq=2 * i + 2,
                        role="assistant",
                        text=turn["answer"],
                        created_at=timestamp(turn.get("assistant_observed_at")),
                    ),
                ]
            )
            for index, result in enumerate(turn.get("actions", [])):
                session.add(
                    ConversationAction(
                        action_id=uuid4().hex,
                        run_id=run,
                        model_step=index,
                        tool_call_id=uuid4().hex,
                        tool_name="query_order",
                        effect="read",
                        arguments={},
                        arguments_fingerprint="f" * 64,
                        approval_status="not_required",
                        execution_status="completed",
                        result=result["result"] if "observed_at" in result else result,
                        completed_at=timestamp(result["observed_at"])
                        if "observed_at" in result
                        else datetime(2026, 10, 1, tzinfo=UTC),
                    )
                )
        if summary:
            meta = summary_meta or {}
            session.add(
                ConversationSummaryRecord(
                    conversation_id=c,
                    version=meta.get("version", 1),
                    content=summary,
                    covered_until_seq=2 * meta.get("covered_turns", len(history)),
                    model_name=meta.get("model_name", "gold-fixture"),
                    created_at=timestamp(meta.get("created_at")),
                    source_run_id=runs[-1],
                )
            )
    return c


def seed_pending(service, conversation_id, tasks):
    """Hand-labelled setup records; target acceptance input always uses real model stages."""
    keys = {}
    for task in tasks:
        original = "查订单，" + task["text"]
        row, _ = service.accept(
            conversation_id, UnderstandingInput(request_id=uuid4().hex, text=original)
        )

        def ref(fragment, original=original):
            start = original.index(fragment)
            return {
                "kind": "input",
                "source_id": "current",
                "span": {"start": start, "end": start + len(fragment)},
                "json_path": [],
                "quote": fragment,
            }

        requests = [
            {
                "text": fragment,
                "intent": intent,
                "evidence_refs": [ref(fragment)],
                "binding_indexes": [],
                "explicit_priority": None,
                "priority_evidence_refs": [],
                "ambiguities": [],
            }
            for fragment, intent in (
                ("查订单", "order"),
                (task["text"], task["intent"]),
            )
        ]
        now = datetime.now(UTC)
        resolution = ResolutionResult(
            resolved_text=original,
            bindings=[],
            unresolved_references=[],
            continuation_selection=None,
        )
        classified = ClassificationResult.model_validate({"requests": requests})
        with (
            service.checkpoint_factory(
                conversation_id="understanding:" + row.understanding_id
            ),
            service.session_factory.begin() as session,
        ):
            repo = UnderstandingRepository(session)
            owned = repo.claim(row.understanding_id, uuid4().hex)
            for stage, output in (
                ("resolution", resolution),
                ("intents", classified),
            ):
                audit = StageAudit(
                    stage=stage,
                    invocation_id="gold-setup-" + uuid4().hex,
                    attempt=1,
                    input_estimate=0,
                    input_tokens=None,
                    output_tokens=None,
                    started_at=now,
                    finished_at=now,
                    outcome="succeeded",
                    verified_refs=[],
                )
                repo.save_stage(
                    row.understanding_id,
                    owned.execution_epoch,
                    owned.execution_owner_id,
                    stage,
                    output,
                    audit,
                )
            repo.release(
                row.understanding_id,
                owned.execution_epoch,
                owned.execution_owner_id,
            )
        # Drive the real graph using persisted setup stages, saving an actual END.
        view = service.run(conversation_id, row.understanding_id)
        keys[view.pending_requests[0].task_id] = task["key"]
    return keys


def prediction(case, view, record, task_keys, elapsed):
    stage_outputs = view.get("stage_results", {})
    classification = stage_outputs.get("intents") or {}
    requests = classification.get("output", {}).get("requests", [])

    def item(value):
        if not value:
            return None
        refs = value["evidence_refs"]
        key = task_keys.get(value["task_id"])
        if key:
            refs = next(
                (
                    r["evidence_refs"]
                    for r in requests
                    if any(
                        ref["kind"] == "task" and ref["source_id"] == value["task_id"]
                        for ref in r["evidence_refs"]
                    )
                ),
                refs,
            )
        return {
            "intent": value["intent"],
            "entities": sorted(
                {b["value"] for b in value["bindings"]}
                | set(active_business_ids(value["text"]))
            ),
            "evidence_spans": [
                r["span"]
                for r in refs
                if r["kind"] == "input" and r["source_id"] == "current" and r["span"]
            ],
            "task_key": key,
        }

    resolution = view.get("resolution") or {}
    violations = []
    if view.get("primary_request"):
        from resolv.understanding_rules import INTENT_BRANCHES_V1

        if view.get("branch") != INTENT_BRANCHES_V1.get(
            view["primary_request"]["intent"]
        ):
            violations.append("fixed_mapping")
    for text_ in [resolution.get("resolved_text", ""), *(r["text"] for r in requests)]:
        if CLAIMS.search(text_):
            try:
                validate_request_semantics(text_, [getattr(case, "text", "")])
            except UnderstandingError:
                violations.append("false_business_completion")
    audits = []
    if record:
        audits = [
            a.model_dump(mode="json")
            | {
                "input_limit": record.snapshot.config.input_limit,
                "estimate_minus_actual": a.input_estimate - a.input_tokens
                if a.input_tokens is not None
                else None,
            }
            for a in record.invocations
        ]
    return {
        "case_id": case.case_id,
        "status": view.get("status"),
        "primary": item(view.get("primary_request")),
        "secondary": [item(r) for r in view.get("pending_requests", [])],
        "binding_values": [b["value"] for b in resolution.get("bindings", [])],
        "needs_clarification": view.get("status") == "needs_clarification",
        "violations": violations,
        "audits": audits,
        "error_code": view.get("error_code"),
        "elapsed_seconds": elapsed,
        "understanding_id": view.get("understanding_id"),
        "view": view,
    }


def read_existing(settings, id):
    # Server-enforced read-only transaction, SELECT-only ORM operations.
    with conversation_engine(settings) as engine, engine.connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        from sqlalchemy.orm import Session

        with Session(bind=connection) as session:
            repo = ConversationRepository(session)
            conversation = repo.conversation(id)
            turns = repo.completed_turns(id)
            summary = repo.summary(id)
            history = [
                {
                    "question": t.user.text,
                    "answer": t.assistant.text,
                    "user_observed_at": t.user.created_at.isoformat(),
                    "assistant_observed_at": t.assistant.created_at.isoformat(),
                    "actions": [
                        {
                            "result": a.result,
                            "observed_at": a.completed_at.isoformat()
                            if a.completed_at
                            else None,
                        }
                        for a in t.actions
                        if a.execution_status == "completed" and a.result is not None
                    ],
                }
                for t in turns
            ]
            proof = fingerprint(
                {
                    "conversation": conversation.conversation_id,
                    "history": history,
                    "summary": summary.model_dump(mode="json") if summary else None,
                }
            )
        connection.rollback()
        meta = (
            {
                "version": summary.version,
                "model_name": summary.model_name,
                "created_at": summary.created_at.isoformat(),
                "covered_turns": sum(
                    t.assistant.seq <= summary.covered_until_seq for t in turns
                ),
            }
            if summary
            else None
        )
        return history, summary.content.model_dump() if summary else None, proof, meta


def run_accuracy(database, args):
    cases = load_cases(args.cases)
    dataset_sha = sha256(args.cases.read_bytes()).hexdigest()
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
    report_data = {
        "dataset_sha256": dataset_sha,
        "model": settings.openai_model,
        "real_model_concurrency": args.concurrency,
        "versions": {
            name: version(name)
            for name in (
                "langgraph",
                "langgraph-checkpoint",
                "langgraph-checkpoint-mysql",
                "langchain-openai",
                "tiktoken",
            )
        },
        "config": context.config.model_dump(mode="json"),
        "results": results,
        "existing_source": None,
    }
    try:
        with conversation_engine(database) as engine:
            factory = create_session_factory(engine)
            model = JsonStageModel(client, counter)
            service = UnderstandingService(
                factory,
                partial(checkpoint_session, database),
                context,
                ReferenceResolver(model, context),
                IntentClassifier(model, context),
            )
            app = FastAPI()
            app.state.understanding_runtime = service
            app.include_router(router)
            with TestClient(app) as http:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [
                        pool.submit(
                            evaluate_one,
                            case,
                            factory,
                            database,
                            context,
                            client,
                            counter,
                        )
                        for case in cases
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        report_data["scores"] = score_cases(cases, results).model_dump(
                            mode="json"
                        )
                        args.report.parent.mkdir(parents=True, exist_ok=True)
                        args.report.write_text(
                            json.dumps(report_data, ensure_ascii=False, indent=2)
                        )
                        print(
                            json.dumps(
                                {
                                    "case": result["case_id"],
                                    "status": result["status"],
                                    "error": result["error_code"],
                                    "elapsed": round(result["elapsed_seconds"], 2),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                if args.existing_conversation_id:
                    source_settings = ConversationDatabaseSettings(
                        _env_file=None,
                        conversation_database_url=os.environ.get(
                            args.source_database_env, ""
                        ),
                    )
                    history, summary, before, meta = read_existing(
                        source_settings, args.existing_conversation_id
                    )
                    if not history:
                        raise UnderstandingError("existing_completed_context_required")
                    copied = seed_history(
                        factory,
                        history,
                        summary,
                        title="readonly-source-copy",
                        summary_meta=meta,
                    )
                    response = http.post(
                        f"/conversations/{copied}/understandings",
                        json={"request_id": uuid4().hex, "text": "继续讨论刚才的订单"},
                    )
                    _, _, after, _ = read_existing(
                        source_settings, args.existing_conversation_id
                    )
                    assert before == after
                    report_data["existing_source"] = {
                        "conversation_id": args.existing_conversation_id,
                        "source_hash_before": before,
                        "source_hash_after": after,
                        "completed_turns": len(history),
                        "http_status": response.status_code,
                        "copy_conversation_id": copied,
                    }
            report = score_cases(cases, results)
            report_data["scores"] = report.model_dump(mode="json")
            report_data["calls_by_stage"] = {
                stage: sum(a["stage"] == stage for r in results for a in r["audits"])
                for stage in ("resolution", "intents")
            }
            args.report.write_text(
                json.dumps(report_data, ensure_ascii=False, indent=2)
            )
            assert_acceptance(report)
            if (
                args.existing_conversation_id
                and report_data["existing_source"]["http_status"] != 200
            ):
                raise UnderstandingError("existing_context_acceptance_failed")
            print(
                json.dumps(
                    {
                        "result": "PASS",
                        "scores": report.model_dump(exclude={"per_case"}),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        client.root_client.close()
        asyncio.run(client.root_async_client.close())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-env", required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "tests/fixtures/understanding_acceptance.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-database-env")
    parser.add_argument("--existing-conversation-id")
    args = parser.parse_args(argv)
    if bool(args.source_database_env) != bool(args.existing_conversation_id):
        parser.error(
            "Existing source requires both explicit source environment and conversation ID"
        )
    try:
        database = ConversationDatabaseSettings(
            _env_file=None,
            conversation_database_url=os.environ.get(args.database_env, ""),
        )
        name = make_url(database.conversation_database_url.get_secret_value()).database
        if not name or not __import__("re").fullmatch(
            r"resolv_conversations_smoke_[A-Za-z0-9_]+", name
        ):
            raise ValueError
    except ValueError:
        parser.error(
            "Explicit isolated resolv_conversations_smoke_* MySQL URL required"
        )
    run_accuracy(database, args)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 — credentials remain private; detailed safe results live in report
        raise SystemExit(
            "Understanding accuracy acceptance FAILED; inspect the local report."
        ) from None
