"""Real isolated MySQL/SQLite crash gate; fake inference proves no model accuracy."""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import event, func, inspect, select, text
from sqlalchemy.engine import make_url

from resolv.core.config import ContextSettings, ConversationDatabaseSettings
from resolv.core.understanding_model import JsonStageModel
from resolv.core.understanding_prompts import INTENT_PROMPT_V1, RESOLUTION_PROMPT_V1
from resolv.db.checkpoints import (
    CheckpointError,
    checkpoint_session,
    setup_checkpoint_tables,
)
from resolv.db.conversation import conversation_engine
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db
from resolv.models import Order
from resolv.models.conversation import (
    ConversationMessage,
    ConversationRun,
)
from resolv.models.understanding import ConversationUnderstanding
from resolv.schemas.conversation import ConversationError
from resolv.services.context import ContextBuilder
from resolv.services.conversation import ConversationService
from resolv.services.conversation_routing import assert_no_legacy_inflight
from resolv.services.intent_classification import IntentClassifier
from resolv.services.reference_resolution import ReferenceResolver
from resolv.services.tool_execution import WriteExecutionService
from resolv.services.understanding import UnderstandingService
from resolv.services.understanding_context import UnderstandingContext

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ".superpowers/sdd/2026-10-14-resolv-three-branch-integration"
DATABASE = "resolv_conversations_smoke_threebranches_20261014"
OWNER = "resolv-threebranches-recovery-v1"
WINDOWS = (
    "after_understanding_terminal",
    "after_approval_pause",
    "after_sqlite_commit",
    "after_final_result",
    "after_sixth_model_step",
)
REQUIRED_SCENARIOS = (
    "atomic_accept",
    "locks_and_fencing",
    "after_understanding_terminal",
    "approval_process_recovery",
    "after_sqlite_commit",
    "after_final_result",
    "after_sixth_model_step",
    "legacy_and_migrations",
)


def marker_path(path):
    return Path(str(path) + ".resolv-smoke.json")


def validate_isolation(database: str, business_db: Path) -> None:
    path = business_db.resolve()
    if database != DATABASE or path != (ROOT / WORKSPACE / "business.db").resolve():
        raise ValueError(
            "only this task's isolated database and business path are allowed"
        )
    marker = marker_path(path)
    expected = {"owner": OWNER, "database": database, "path": str(path)}
    if path.exists() and (
        not marker.is_file() or json.loads(marker.read_text()) != expected
    ):
        raise ValueError("existing business database ownership does not match")
    if marker.exists() and json.loads(marker.read_text()) != expected:
        raise ValueError("business database ownership does not match")


def validate_state(state, database, business_db):
    if (
        state.get("owner") != OWNER
        or state.get("database") != database
        or state.get("business_db") != str(business_db.resolve())
        or state.get("window") not in WINDOWS
    ):
        raise ValueError("state belongs to another recovery gate")


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def settings_for_smoke(database):
    url = make_url(
        ConversationDatabaseSettings().conversation_database_url.get_secret_value()
    ).set(database=database)
    return ConversationDatabaseSettings(
        _env_file=None,
        conversation_database_url=url.render_as_string(hide_password=False),
    )


def bootstrap(settings, factory):
    config = Config(str(ROOT / "alembic.conversation.ini"))
    with factory.kw["bind"].begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    setup_checkpoint_tables(settings)


@contextmanager
def owned_business(path, database, *, initialize=False):
    validate_isolation(database, path)
    if initialize:
        path.parent.mkdir(parents=True, exist_ok=True)
        marker_path(path).write_text(
            json.dumps(
                {"owner": OWNER, "database": database, "path": str(path.resolve())}
            )
        )
    if not path.exists() and not initialize:
        raise ValueError("business database has not been initialized by matrix")
    engine = create_db_engine(f"sqlite:///{path.resolve()}")
    try:
        if initialize:
            init_db(engine)
        yield create_session_factory(engine)
    finally:
        engine.dispose()


class Counter:
    def count_text(self, value):
        return len(value) // 4

    def count(self, messages, schemas):
        return (
            sum(len(str(message.content)) // 4 for message in messages)
            + len(json.dumps(schemas)) // 4
        )


class StageClient:
    def __init__(self):
        self.calls = 0

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        payload = json.loads(messages[1].content.split("\n", 1)[1])
        question = payload["original_text"]
        if "resolution" not in payload:
            result = {
                "resolved_text": question,
                "bindings": [],
                "unresolved_references": [],
                "continuation_selection": None,
            }
        else:
            result = {
                "requests": [
                    {
                        "text": question,
                        "intent": "complaint" if "投诉" in question else "order",
                        "evidence_refs": [
                            {
                                "kind": "input",
                                "source_id": "current",
                                "span": {"start": 0, "end": len(question)},
                                "quote": question,
                                "json_path": [],
                            }
                        ],
                        "binding_indexes": [],
                        "explicit_priority": None,
                        "priority_evidence_refs": [],
                        "ambiguities": [],
                    }
                ]
            }
        return AIMessage(content=json.dumps(result, ensure_ascii=False))


class BusinessModel:
    model_name = "fake-threebranches-gate"

    def __init__(self, window, order_id):
        self.window, self.order_id, self.calls = window, order_id, 0

    def bind_tools(self, tools):
        assert [tool.name for tool in tools] == [
            "query_order",
            "query_logistics",
            "cancel_order",
        ]
        return self

    def bind(self, **kwargs):
        self.require_tool = kwargs.get("tool_choice") == "required"
        return self

    def stream(self, messages):
        self.calls += 1
        step = 1 + sum(message.type == "ai" for message in messages)
        cancel = self.window in {"after_sqlite_commit", "after_approval_pause"}
        calls = []
        if (
            self.window == "after_sixth_model_step"
            or cancel
            and not any(message.type == "tool" for message in messages)
            or getattr(self, "require_tool", False)
        ):
            calls = [
                {
                    "id": f"gate-call-{step}",
                    "name": "cancel_order" if cancel else "query_order",
                    "args": {"order_id": self.order_id},
                }
            ]
        yield AIMessageChunk(
            content="" if calls else "fake inference final reply",
            tool_calls=calls,
            response_metadata={"finish_reason": "tool_calls" if calls else "stop"},
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
        )


def make_service(
    settings, factory, business, window="after_final_result", order_id="GATE001"
):
    from resolv.core.business_model import BUSINESS_SYSTEM_PROMPT
    from resolv.repositories.conversation import fingerprint
    from resolv.schemas.branch_execution import BusinessRunConfig
    from resolv.services.business_policy import business_tool_schemas

    business_config = BusinessRunConfig(
        model_name=BusinessModel.model_name,
        api_address_fingerprint="f" * 64,
        system_prompt=BUSINESS_SYSTEM_PROMPT,
        tool_schema_fingerprint=fingerprint(business_tool_schemas()),
        decision_policy_version="require-observations-v1",
    )
    counter = Counter()
    context_settings = ContextSettings(
        _env_file=None,
        model_context_window_tokens=256000,
        tokenizer_profile="o200k_base",
        knowledge_context_max_tokens=1,
        provider_input_overhead_tokens=business_config.provider_input_overhead_tokens,
    )
    context = UnderstandingContext(
        context_settings,
        counter,
        model_name="fake-gate-understanding",
        api_address_fingerprint="f" * 64,
        resolution_prompt=RESOLUTION_PROMPT_V1,
        intent_prompt=INTENT_PROMPT_V1,
    )
    stages = JsonStageModel(StageClient(), counter)
    checkpoints = partial(checkpoint_session, settings)
    understanding = UnderstandingService(
        factory,
        checkpoints,
        context,
        ReferenceResolver(stages, context),
        IntentClassifier(stages, context),
    )
    return ConversationService(
        factory,
        checkpoints,
        orchestration_version="routed-v2",
        business_run_config=business_config,
        understanding=understanding,
        context_builder=ContextBuilder(context_settings, counter),
        model=BusinessModel(window, order_id),
        write_service=WriteExecutionService(factory, business),
    )


def verify_state(service, state, *, business_db):
    run = service.get_run(state["run_id"])
    understanding = service.understanding.record(
        state["conversation_id"], state["understanding_id"]
    )
    assert run.status == "completed" and run.result_json is not None
    assert run.understanding_id == understanding.understanding_id
    assert (
        understanding.status == "ready"
        and understanding.snapshot.snapshot_hash == state["snapshot_hash"]
    )
    assert len(understanding.invocations) == 2
    ids = []
    for thread_id in (run.conversation_id, "understanding:" + run.understanding_id):
        with service.checkpoint_factory(conversation_id=thread_id) as checkpoints:
            saved = checkpoints.saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            )
            assert (
                saved is not None
                and saved.checkpoint["channel_values"]["phase"] == "end"
            )
            ids.append(saved.config["configurable"]["thread_id"])
            if thread_id == run.conversation_id:
                parent = saved.checkpoint["channel_values"]
    assert ids == [run.conversation_id, "understanding:" + run.understanding_id]
    with sqlite3.connect(
        f"file:{business_db.resolve()}?mode=ro", uri=True
    ) as connection:
        receipts = connection.execute(
            "SELECT result FROM tool_execution_receipts WHERE run_id=?", (run.run_id,)
        ).fetchall()
        order = connection.execute(
            "SELECT status FROM orders WHERE order_no=?", (state["order_id"],)
        ).fetchone()
    cancel = state["window"] in {"after_sqlite_commit", "after_approval_pause"}
    assert len(receipts) == int(cancel)
    if cancel:
        assert order[0] == "cancelled" and json.loads(receipts[0][0])["changed"] is True
    if state["window"] == "after_sixth_model_step":
        assert parent["model_step"] == 6 and parent["remaining_iterations"] == 0
        assert (
            len(run.model_audits) == 6 and run.result_json["reply_kind"] == "step_limit"
        )
        actions = service.detail(run.conversation_id).actions
        assert len(actions) == 5 and all(
            action.approval_id is None for action in actions
        )
    elif not cancel:
        assert len(run.result_json["suggested_actions"]) == 2 and run.model_audits == []
    if state.get("saved_result"):
        assert run.result_json == state["saved_result"]
    return {
        "passed": True,
        "fake_inference": True,
        "accuracy_evidence": False,
        "readonly_business": True,
        "window": state["window"],
        "run_id": run.run_id,
        "understanding_id": run.understanding_id,
        "action_id": state.get("action_id"),
        "snapshot_hash": understanding.snapshot.snapshot_hash,
        "checkpoint_threads": ids,
        "receipt_count": len(receipts),
        "changed_count": sum(
            json.loads(row[0]).get("changed") is True for row in receipts
        ),
        "model_audit_count": len(run.model_audits),
        "model_step": parent["model_step"],
    }


def pause(service, business, state_file, window, database, business_db):
    order_id = service.model.order_id
    with business.begin() as session:
        session.add(Order(order_no=order_id, status="processing"))
    conversation = service.create()
    question = (
        "取消订单 "
        if window in {"after_sqlite_commit", "after_approval_pause"}
        else "查订单 "
        if window == "after_sixth_model_step"
        else "投诉配送问题 "
    ) + order_id
    run, _ = service.accept(
        conversation.conversation_id, uuid4().hex, question, "agent"
    )
    record = service.understanding.record(run.conversation_id, run.understanding_id)
    state = {
        "owner": OWNER,
        "database": database,
        "business_db": str(business_db.resolve()),
        "window": window,
        "run_id": run.run_id,
        "conversation_id": run.conversation_id,
        "understanding_id": run.understanding_id,
        "snapshot_hash": record.snapshot.snapshot_hash,
        "order_id": order_id,
    }
    save_state(state_file, state)

    def crash(actual):
        if actual == window:
            state["paused"] = True
            state["saved_result"] = service.get_run(run.run_id).result_json
            save_state(state_file, state)
            os._exit(23)

    service.fault_hook = crash
    events = list(service.run(run.run_id))
    if window in {"after_sqlite_commit", "after_approval_pause"}:
        approval = next(
            item.data for item in events if item.type == "approval_required"
        )
        state.update(
            action_id=approval["action_id"], approval_id=approval["approval_id"]
        )
        save_state(state_file, state)
        if window == "after_approval_pause":
            crash(window)
        service.decide(
            approval["approval_id"], run.conversation_id, True, require_pause=True
        )
        list(service.resume(run.conversation_id, run.run_id))
    raise RuntimeError("required crash window was not reached")


def run_window(
    window, *, database, business_db, state_file, report, runner=subprocess.run
):
    exit_codes, last = [], {}
    for mode, expected in (("pause", 23), ("resume", 0), ("verify", 0)):
        command_line = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--database",
            database,
            "--business-db",
            str(business_db),
            "--state-file",
            str(state_file),
            "--window",
            window,
            "--report",
            str(report),
        ]
        result = runner(command_line, capture_output=True, text=True)
        exit_codes.append(result.returncode)
        try:
            last = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            last = {}
        if result.returncode != expected:
            return {
                "passed": False,
                "exit_codes": exit_codes,
                "window": window,
                "child": last,
            }
    return {**last, "passed": last.get("passed") is True, "exit_codes": exit_codes}


def assert_gate_report(report):
    if (
        report.get("fake_inference") is not True
        or set(report.get("scenarios", {})) != set(REQUIRED_SCENARIOS)
        or any(
            result.get("passed") is not True for result in report["scenarios"].values()
        )
    ):
        raise ValueError("all eight real storage scenarios are required")


def atomic_probe(service):
    c = service.create()

    def failure(*args):
        raise RuntimeError("injected atomic acceptance failure")

    event.listen(ConversationMessage, "before_insert", failure)
    try:
        try:
            service.accept(c.conversation_id, "retry", "投诉", "agent")
        except RuntimeError:
            pass
        else:
            raise AssertionError("accept did not fail")
    finally:
        event.remove(ConversationMessage, "before_insert", failure)
    with service.session_factory() as session:
        for model in (ConversationMessage, ConversationRun, ConversationUnderstanding):
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(model)
                    .where(model.conversation_id == c.conversation_id)
                )
                == 0
            )
    first, created = service.accept(c.conversation_id, "retry", "投诉", "agent")
    second, duplicate = service.accept(c.conversation_id, "retry", "投诉", "agent")
    assert (
        created
        and not duplicate
        and first.run_id == second.run_id
        and first.understanding_id == second.understanding_id
    )
    assert service.understanding.resolver.model.client.calls == 0
    return {
        "passed": True,
        "run_id": first.run_id,
        "understanding_id": first.understanding_id,
        "accept_rollback": True,
        "idempotent": True,
    }


def lock_probe(service):
    c = service.create()

    def accept(number):
        try:
            return service.accept(
                c.conversation_id, f"concurrent-{number}", "投诉", "agent"
            )[0].run_id
        except ConversationError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(accept, [1, 2]))
    assert results.count("conversation_busy") == 1
    run_id = next(value for value in results if value != "conversation_busy")
    with service.execution(run_id) as execution:
        try:
            with service.checkpoint_factory(conversation_id=c.conversation_id):
                raise AssertionError("lock was not exclusive")
        except CheckpointError as error:
            assert error.code == "conversation_busy"
        with service.session_factory.begin() as session:
            row = session.get(ConversationRun, run_id)
            row.execution_epoch += 1
            row.execution_owner_id = "gate-successor"
        try:
            service.commit_answer(run_id, execution.epoch, "stale")
        except ConversationError:
            pass
        else:
            raise AssertionError("old epoch wrote an answer")
    c2 = service.create()
    run2, _ = service.accept(c2.conversation_id, "lost", "投诉", "agent")
    with service.execution(run2.run_id) as execution:
        checkpoint = execution.checkpoint_session
        with checkpoint._connection.cursor() as cursor:
            cursor.execute("SELECT RELEASE_LOCK(%s)", (checkpoint._lock_name,))
        try:
            execution.assert_owned()
        except ConversationError:
            pass
        else:
            raise AssertionError("lost named lock retained execution authority")
    return {
        "passed": True,
        "run_id": run_id,
        "concurrent_accepts": 1,
        "stale_epoch_writes": 0,
        "lost_lock_fenced": True,
    }


def legacy_probe(settings, service):
    factory = service.session_factory
    with factory() as session:
        before = session.scalar(select(func.count()).select_from(ConversationMessage))
    bootstrap(settings, factory)
    bootstrap(settings, factory)
    with factory() as session:
        assert (
            session.scalar(select(func.count()).select_from(ConversationMessage))
            == before
        )
    legacy = ConversationService(
        factory,
        service.checkpoint_factory,
        owner_id="gate-legacy-" + uuid4().hex,
        context_builder=service.context_builder,
        model=BusinessModel("legacy", "unused"),
    )
    c = legacy.create()
    run, _ = legacy.accept(c.conversation_id, "legacy", "old question", "chat")
    with factory() as session:
        try:
            assert_no_legacy_inflight(session, owner_id=legacy.owner_id)
        except ConversationError as error:
            assert error.code == "legacy_run_requires_resolution"
        else:
            raise AssertionError("legacy gate did not block")
    # This is a newly seeded isolated legacy READ-only chat, never a formal run.
    events = list(legacy.run(run.run_id))
    assert events[-1].data.get("status") == "completed"
    with factory() as session:
        assert_no_legacy_inflight(session, owner_id=legacy.owner_id)
    assert (
        legacy.detail(c.conversation_id).messages[-1].text
        == "fake inference final reply"
    )
    return {
        "passed": True,
        "legacy_run_id": run.run_id,
        "old_messages_preserved": before,
        "repeat_migrations": 2,
        "legacy_gate_blocked": True,
    }


def preflight():
    settings = ConversationDatabaseSettings()
    with conversation_engine(settings) as engine, engine.connect() as connection:
        connection.exec_driver_sql("START TRANSACTION READ ONLY")
        columns = {
            column["name"]
            for column in inspect(connection).get_columns("conversation_runs")
        }
        clause = (
            " AND r.orchestration_version='legacy-v1'"
            if "orchestration_version" in columns
            else ""
        )
        rows = (
            connection.execute(
                text(
                    "SELECT r.run_id,r.status FROM conversation_runs r JOIN conversations c ON c.conversation_id=r.conversation_id WHERE r.status NOT IN ('completed','failed')"
                    + clause
                )
            )
            .mappings()
            .all()
        )
        connection.rollback()
    return {
        "passed": not rows,
        "mode": "preflight",
        "read_only": True,
        "legacy_inflight": [dict(row) for row in rows],
        "ddl_calls": 0,
        "model_calls": 0,
        "business_calls": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["matrix", "pause", "resume", "verify", "preflight"],
        required=True,
    )
    parser.add_argument("--database", default=DATABASE)
    parser.add_argument(
        "--business-db", type=Path, default=ROOT / WORKSPACE / "business.db"
    )
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--window", choices=WINDOWS)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "passed": False,
        "mode": args.mode,
        "fake_inference": True,
        "accuracy_evidence": False,
        "database": args.database
        if args.mode != "preflight"
        else "configured_formal_store",
        "versions": {
            name: version(name)
            for name in [
                "langgraph",
                "langgraph-checkpoint",
                "langgraph-checkpoint-mysql",
                "sqlalchemy",
            ]
        },
    }
    try:
        if args.mode == "preflight":
            report.update(preflight())
        else:
            validate_isolation(args.database, args.business_db)
            settings = settings_for_smoke(args.database)
            with conversation_engine(settings) as engine:
                factory = create_session_factory(engine)
                if args.mode == "matrix":
                    bootstrap(settings, factory)
                with owned_business(
                    args.business_db, args.database, initialize=args.mode == "matrix"
                ) as business:
                    if args.mode == "matrix":
                        service = make_service(settings, factory, business)
                        report["scenarios"] = {}
                        report["scenarios"]["atomic_accept"] = atomic_probe(service)
                        report["scenarios"]["locks_and_fencing"] = lock_probe(service)
                        for window in WINDOWS:
                            state_file = (
                                ROOT
                                / WORKSPACE
                                / (window + "-" + uuid4().hex + ".json")
                            )
                            result = run_window(
                                window,
                                database=args.database,
                                business_db=args.business_db,
                                state_file=state_file,
                                report=args.report,
                            )
                            report["scenarios"][
                                "approval_process_recovery"
                                if window == "after_approval_pause"
                                else window
                            ] = result
                            if not result["passed"]:
                                raise AssertionError("process recovery failed")
                        report["scenarios"]["legacy_and_migrations"] = legacy_probe(
                            settings, service
                        )
                        assert_gate_report(report)
                        report["passed"] = True
                    else:
                        if (
                            args.state_file is None
                            or not args.state_file.resolve().is_relative_to(
                                (ROOT / WORKSPACE).resolve()
                            )
                        ):
                            raise ValueError("state file must belong to this task")
                        if args.mode == "pause":
                            if not args.window or args.state_file.exists():
                                raise ValueError(
                                    "pause requires a new state file and a window"
                                )
                            service = make_service(
                                settings,
                                factory,
                                business,
                                args.window,
                                "MHG" + uuid4().hex[:16],
                            )
                            pause(
                                service,
                                business,
                                args.state_file,
                                args.window,
                                args.database,
                                args.business_db,
                            )
                        state = json.loads(args.state_file.read_text())
                        validate_state(state, args.database, args.business_db)
                        service = make_service(
                            settings,
                            factory,
                            business,
                            state["window"],
                            state["order_id"],
                        )
                        if args.mode == "resume":
                            if state["window"] == "after_approval_pause":
                                service.decide(
                                    state["approval_id"],
                                    state["conversation_id"],
                                    True,
                                    require_pause=True,
                                )
                            events = list(
                                service.resume(
                                    state["conversation_id"], state["run_id"]
                                )
                            )
                            assert (
                                events[-1].type == "done"
                                and events[-1].data.get("status") == "completed"
                            )
                        report.update(
                            verify_state(service, state, business_db=args.business_db)
                        )
    except Exception as caught:  # noqa: BLE001 — no raw credentials, SQL parameters or provider errors
        report.update(
            passed=False,
            error_code=caught.code
            if isinstance(caught, ConversationError)
            else "recovery_gate_failed",
            error_type=type(caught).__name__,
            failure_location=[
                {
                    "file": Path(frame.filename).name,
                    "line": frame.lineno,
                    "function": frame.name,
                }
                for frame in traceback.extract_tb(caught.__traceback__)[-6:]
            ],
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("a") as output:
        output.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
