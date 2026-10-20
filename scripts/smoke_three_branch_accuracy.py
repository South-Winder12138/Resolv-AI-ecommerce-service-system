"""Opt-in 60-case real model acceptance through the actual HTTP endpoints."""

import argparse
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager, contextmanager
from importlib import import_module
from pathlib import Path
from time import monotonic
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from resolv.api.agent import router as agent_router
from resolv.api.chat import router as chat_router
from resolv.api.conversations import router as conversations_router
from resolv.api.understandings import router as understandings_router
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db
from resolv.models import Order, Shipment
from resolv.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
)
from resolv.models.tool_execution import ToolExecutionReceipt
from resolv.schemas.conversation import ConversationError
from resolv.services.branch_evaluation import (
    BLOCKERS,
    assert_branch_acceptance,
    load_branch_cases,
    score_branch_cases,
)
from resolv.services.branch_replies import render_reply
from resolv.services.routed_runtime import routed_conversation_runtime

gate = import_module(
    "scripts.smoke_three_branch_recovery"
    if __package__
    else "smoke_three_branch_recovery"
)
DATABASE, ROOT, WORKSPACE = gate.DATABASE, gate.ROOT, gate.WORKSPACE
assert_gate_report, settings_for_smoke = (
    gate.assert_gate_report,
    gate.settings_for_smoke,
)
OWNER = "resolv-threebranches-accuracy-v1"


def validate_accuracy_isolation(database, business_db):
    path = business_db.resolve()
    if (
        database != DATABASE
        or path.parent != (ROOT / WORKSPACE).resolve()
        or not re.fullmatch(r"accuracy[-A-Za-z0-9_]*\.db", path.name)
    ):
        raise ValueError(
            "accuracy must use the isolated logical database and a task-owned SQLite file"
        )


def assert_gold_hash(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("frozen gold hash changed")


def require_recovery_gate():
    records = [
        json.loads(line)
        for line in (
            ROOT
            / "docs/superpowers/acceptance/2026-10-14-three-branches/mysql-gate.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    last = next(
        r for r in reversed(records) if r.get("mode") == "matrix" and r.get("passed")
    )
    assert_gate_report(last)


@contextmanager
def isolated_environment(database, business_db):
    validate_accuracy_isolation(database, business_db)
    marker = business_db.with_suffix(".owner.json")
    expected = {
        "owner": OWNER,
        "database": database,
        "path": str(business_db.resolve()),
    }
    if (
        not business_db.is_file()
        or not marker.is_file()
        or json.loads(marker.read_text()) != expected
    ):
        raise ValueError("accuracy business store ownership missing")
    settings = settings_for_smoke(database)
    with patch.dict(
        os.environ,
        {
            "CONVERSATION_DATABASE_URL": settings.conversation_database_url.get_secret_value(),
            "DATABASE_URL": f"sqlite:///{business_db.resolve()}",
            "KNOWLEDGE_CHAT_ENABLED": "false",
        },
    ):
        yield


def create_isolated_app(database, business_db):
    @asynccontextmanager
    async def lifespan(app):
        with isolated_environment(database, business_db):
            owner = routed_conversation_runtime()
            service = await run_in_threadpool(owner.__enter__)
            try:
                app.state.conversation_runtime, app.state.understanding_runtime = (
                    service,
                    service.understanding,
                )
                yield
            finally:
                await run_in_threadpool(owner.__exit__, None, None, None)

    app = FastAPI(lifespan=lifespan)
    for router in (
        understandings_router,
        conversations_router,
        chat_router,
        agent_router,
    ):
        app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def prepare_business(database, path, cases):
    validate_accuracy_isolation(database, path)
    if path.exists() or path.with_suffix(".owner.json").exists():
        raise ValueError(
            "each accuracy attempt requires a new owned SQLite file; never reset previous results"
        )
    engine = create_db_engine(f"sqlite:///{path.resolve()}")
    try:
        init_db(engine)
        factory = create_session_factory(engine)
        with factory.begin() as session:
            for spec in {
                order["id"]: order for case in cases for order in case["orders"]
            }.values():
                row = Order(order_no=spec["id"], status=spec["status"])
                if spec["tracking"]:
                    row.shipment = Shipment(
                        tracking_no=spec["tracking"],
                        status="in_transit",
                        location=spec["location"],
                    )
                session.add(row)
    finally:
        engine.dispose()
    path.with_suffix(".owner.json").write_text(
        json.dumps({"owner": OWNER, "database": database, "path": str(path.resolve())})
    )


def seed_history(service, conversation_id, history):
    with service.session_factory.begin() as session:
        c = session.get(Conversation, conversation_id)
        for turn in history:
            run_id, q, a = (uuid4().hex for _ in range(3))
            session.add(
                ConversationRun(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    request_id="fixture-" + run_id,
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
                        conversation_id=conversation_id,
                        run_id=run_id,
                        seq=c.next_message_seq,
                        role="user",
                        text=turn["question"],
                    ),
                    ConversationMessage(
                        message_id=a,
                        conversation_id=conversation_id,
                        run_id=run_id,
                        seq=c.next_message_seq + 1,
                        role="assistant",
                        text=turn["answer"],
                    ),
                ]
            )
            c.next_message_seq += 2


def public_response(response):
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        events = [
            json.loads(frame.split("data: ", 1)[1])
            for frame in response.text.strip().split("\n\n")
            if "data: " in frame
        ]
        error = next((event for event in events if event["type"] == "error"), None)
        if error:
            return {"status": "error", **error}
        final = next(
            (
                event
                for event in reversed(events)
                if event["type"] in {"message", "approval_required"}
            ),
            {},
        )
        return {
            **final,
            "status": "approval_required"
            if final.get("type") == "approval_required"
            else "completed",
            "answer": final.get("text", ""),
        }
    body = response.json()
    return (
        body
        if response.status_code == 200
        else {"status": "error", **body.get("detail", {})}
    )


def count_model_audit(audit, counters):
    if audit.get("error_code") is not None:
        counters["technical_errors"] += 1
    if audit["actual_input"] is None or audit["actual_output"] is None:
        counters["usage_unknown"] += 1
    elif audit["actual_input"] > audit["estimate"]:
        counters["input_underestimates"] += 1
    if audit["estimate"] > 12000:
        counters["input_budget_violations"] += 1
    if (audit["actual_output"] or 0) > 2000:
        counters["output_budget_violations"] += 1


def count_requested_tool(call, case, counters):
    if call["name"] not in case["expect"]["allowed_tools"]:
        counters["out_of_scope_tools"] += 1
    args = call["args"]
    if "order_id" in args and args["order_id"] not in case["expect"]["allowed_orders"]:
        counters["wrong_entity_bindings"] += 1
    if "tracking_no" in args and args["tracking_no"] not in [
        spec["tracking"]
        for spec in case["orders"]
        if spec["id"] in case["expect"]["allowed_orders"]
    ]:
        counters["wrong_entity_bindings"] += 1


@contextmanager
def observe_stage_outputs(service):
    outputs = {}
    resolver, classifier = (
        service.understanding.resolver,
        service.understanding.classifier,
    )
    previous = resolver.model, classifier.model

    class Observer:
        def __init__(self, model):
            self.model = model

        def invoke(self, *args, **kwargs):
            output, audit = self.model.invoke(*args, **kwargs)
            outputs[audit.invocation_id] = output.model_dump(mode="json")
            return output, audit

    resolver.model, classifier.model = (Observer(model) for model in previous)
    try:
        yield outputs
    finally:
        resolver.model, classifier.model = previous


def evaluate_case(client, service, case, index):
    counters = dict.fromkeys(BLOCKERS, 0)
    start = monotonic()
    conversation_id = client.post("/conversations").json()["conversation_id"]
    seed_history(service, conversation_id, case["history"])
    path = "/agent/chat" if index % 2 == 0 else "/agent/stream"
    body = public_response(
        client.post(
            path,
            json={
                "session_id": conversation_id,
                "request_id": case["id"],
                "question": case["input"],
            },
        )
    )
    run_id = body.get("run_id")
    approval_decision, repeat_confirmations = None, 0
    if body.get("status") == "approval_required" and case["expect"]["approval"] in {
        "approve",
        "reject",
    }:
        approval_decision = case["expect"]["approval"] == "approve"
        approval_path = "/agent/approvals/" + body["approval_id"]
        decision = {"session_id": conversation_id, "approved": approval_decision}
        body = public_response(client.post(approval_path, json=decision))
        duplicate = public_response(client.post(approval_path, json=decision))
        repeat_confirmations = 1
        if duplicate != body:
            counters["duplicate_cancellations"] += 1
    if body.get("status") == "error" or not run_id:
        counters["technical_errors"] += 1
    detail = client.get(f"/conversations/{conversation_id}").json()
    current = next((run for run in detail["runs"] if run["run_id"] == run_id), None)
    result = {
        "case_id": case["id"],
        "conversation_id": conversation_id,
        "run_id": run_id,
        "counters": counters,
        "http_result": body,
        "elapsed_seconds": monotonic() - start,
        "approval_decision": approval_decision,
        "repeat_confirmations": repeat_confirmations,
    }
    if current is None:
        return result
    run = service.get_run(run_id)
    record = service.understanding.record(conversation_id, run.understanding_id)
    view = service.understanding.view(record)
    actions = [action for action in detail["actions"] if action["run_id"] == run_id]
    tasks = client.get(
        f"/conversations/{conversation_id}/understanding-tasks?limit=100"
    ).json()["requests"]
    with service.write_service.business_session_factory() as session:
        receipts = session.scalars(
            select(ToolExecutionReceipt).where(ToolExecutionReceipt.run_id == run_id)
        ).all()
        states = {
            order.order_no: order.status for order in session.scalars(select(Order))
        }
    primary_intent = view.primary_request.intent if view.primary_request else None
    metadata = run.result_json
    with service.checkpoint_factory(conversation_id=conversation_id) as owner:
        checkpoint = owner.saver.get_tuple(
            {"configurable": {"thread_id": conversation_id, "checkpoint_ns": ""}}
        )
        requested = (
            [
                call
                for message in checkpoint.checkpoint["channel_values"].get(
                    "messages", []
                )
                if message.type == "ai"
                for call in message.tool_calls
            ]
            if checkpoint
            else []
        )
    result.update(
        intent=primary_intent,
        branch=view.branch,
        status=run.status,
        reply_kind=metadata["reply_kind"] if metadata else None,
        answer=body.get("answer", ""),
        primary=view.primary_request.model_dump(mode="json")
        if view.primary_request
        else None,
        actions=actions,
        requested_tools=requested,
        pending=tasks,
        result_json=metadata,
        business_audits=run.model_audits,
        understanding_audits=[
            audit.model_dump(mode="json") for audit in record.invocations
        ],
        order_states={spec["id"]: states[spec["id"]] for spec in case["orders"]},
        receipt_count=len(receipts),
        receipts=[
            {"action_id": receipt.action_id, "result": receipt.result}
            for receipt in receipts
        ],
    )
    names = {action["tool_name"] for action in actions}
    for call in requested:
        count_requested_tool(call, case, counters)
    counters["missing_query_results"] += len(
        set(case["expect"]["required_tools"]) - names
    )
    if receipts and approval_decision is not True:
        counters["unapproved_cancellations"] += len(receipts)
    counters["duplicate_cancellations"] += max(0, len(receipts) - 1)
    answer = result["answer"]
    if re.search(
        r"已(?:经)?(?:为您|帮您|给您)?(?:成功)?(?:转接|转人工|受理|提交工单|创建工单)",
        answer,
    ):
        counters["false_escalation_claims"] += 1
    if re.search(r"已(?:经)?(?:成功)?取消", answer) and not any(
        receipt.result.get("success") is True for receipt in receipts
    ):
        counters["fabricated_results"] += 1
    for spec in case["orders"]:
        changed = any(
            receipt.result.get("order_id") == spec["id"]
            and receipt.result.get("changed") is True
            for receipt in receipts
        )
        if states[spec["id"]] != ("cancelled" if changed else spec["status"]):
            counters["inconsistent_business_state"] += 1
    for order_id in case["expect"]["pending_orders"]:
        if not any(
            order_id in task["text"] and task["queue_status"] == "pending"
            for task in tasks
        ):
            counters["inconsistent_business_state"] += 1
    fixed = (
        render_reply(run_id, view, mode="agent")
        if view.status in {"ready", "needs_clarification"}
        else None
    )
    if fixed is not None and (
        answer != fixed.answer or metadata != fixed.metadata.model_dump(mode="json")
    ):
        counters["route_errors"] += 1
    if view.branch != "business" or primary_intent == "human":
        counters["out_of_scope_tools"] += len(actions) + len(run.model_audits)
    if case["expect"]["approval"] == "none" and any(
        action["approval_id"] for action in actions
    ):
        counters["out_of_scope_tools"] += 1
    for audit in run.model_audits:
        count_model_audit(audit, counters)
    return result


def summary_probe(client, service):
    c = client.post("/conversations").json()["conversation_id"]
    clause = "这是历史咨询数据，仅讨论包装和配送时间，没有授权任何取消或退款。"
    question = clause
    while service.context_builder.counter.count_text(question) < 350:
        question += clause
    seed_history(
        service, c, [{"question": question, "answer": question} for _ in range(10)]
    )
    with service.write_service.business_session_factory.begin() as session:
        session.add(Order(order_no="MHA_SUM01", status="processing"))
    body = public_response(
        client.post(
            "/agent/chat",
            json={
                "session_id": c,
                "request_id": "summary-probe",
                "question": "查询订单 MHA_SUM01 的状态",
            },
        )
    )
    run_id = body.get("run_id")
    run = service.get_run(run_id) if run_id else None
    audits = [
        audit
        for audit in (run.model_audits if run else [])
        if audit.get("purpose") == "summary"
    ]
    passed = bool(
        run
        and run.status == "completed"
        and run.summary_version > 0
        and audits
        and all(
            a["error_code"] is None
            and a["actual_input"] is not None
            and a["actual_input"] <= a["estimate"] <= 12000
            and a["actual_output"] <= 2000
            for a in audits
        )
    )
    return {
        "passed": passed,
        "run_id": run_id,
        "summary_version": run.summary_version if run else None,
        "audits": audits,
        "http_result": body,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--business-db", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "tests/fixtures/branch_execution_acceptance.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8014)
    args = parser.parse_args(argv)
    validate_accuracy_isolation(args.database, args.business_db)
    require_recovery_gate()
    expected_hash = (
        (ROOT / "docs/superpowers/acceptance/2026-10-14-three-branches/gold.sha256")
        .read_text()
        .split()[0]
    )
    assert_gold_hash(args.fixture, expected_hash)
    cases = load_branch_cases(args.fixture)
    if args.serve:
        import uvicorn

        uvicorn.run(
            create_isolated_app(args.database, args.business_db),
            host="127.0.0.1",
            port=args.port,
        )
        return 0
    if args.report.exists():
        raise ValueError(
            "preserve previous acceptance reports; select a new report path"
        )
    prepare_business(args.database, args.business_db, cases)
    outcomes = []
    with TestClient(create_isolated_app(args.database, args.business_db)) as client:
        service = client.app.state.conversation_runtime
        for index, case in enumerate(cases):
            with observe_stage_outputs(service) as outputs:
                try:
                    outcome = evaluate_case(client, service, case, index)
                except Exception as caught:  # noqa: BLE001 — keep every failed sample, no technical retries
                    outcome = {
                        "case_id": case["id"],
                        "counters": {
                            **dict.fromkeys(BLOCKERS, 0),
                            "technical_errors": 1,
                        },
                        "error_code": caught.code
                        if isinstance(caught, ConversationError)
                        else "acceptance_case_failed",
                        "error_type": type(caught).__name__,
                    }
                outcome["stage_outputs"] = outputs
            outcomes.append(outcome)
            with args.report.with_suffix(".jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {"gold_sha256": expected_hash, **outcome}, ensure_ascii=False
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "case_id": case["id"],
                        "intent": outcome.get("intent"),
                        "status": outcome.get("status"),
                        "errors": {k: v for k, v in outcome["counters"].items() if v},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            assert_gold_hash(args.fixture, expected_hash)
        try:
            summary = summary_probe(client, service)
        except Exception as caught:  # noqa: BLE001 — separate independent summary Gate
            summary = {
                "passed": False,
                "error_type": type(caught).__name__,
                "error_code": "summary_probe_failed",
            }
    report = score_branch_cases(cases, outcomes) | {
        "gold_sha256": expected_hash,
        "outcomes": outcomes,
        "summary_probe": summary,
        "passed": False,
    }
    try:
        assert_branch_acceptance(report)
        report["passed"] = summary["passed"]
    except ValueError:
        pass
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "outcomes"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
